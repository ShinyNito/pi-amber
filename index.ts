import type { Message, Usage } from "@earendil-works/pi-ai";
import {
  convertToLlm,
  type CompactionEntry,
  type ExtensionAPI,
  type ExtensionContext,
  type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { loadConfig, type AmberConfig } from "./config.ts";
import { type SummaryAuth, summarizeConversation } from "./summarizer.ts";
import { detectSummaryLanguage } from "./summaryLanguage.ts";
import { type VerificationSource } from "./validate.ts";
import { SUMMARY_PROMPT_VERSION } from "./policy.ts";
import {
  createCompactionPressure,
  normalizePressure,
  notePressureAfterCompaction,
  type CompactionPressure,
} from "./pressure.ts";

export type AmberCompactionDetails = {
  version: string;
  summaryLanguage?: string;
  payloadTokens: number;
  tokensBefore: number;
  summaryChars: number;
  summarizer?: Usage;
  model: string;
  /** Whether the previous compaction was effective (kept below threshold). */
  effective?: boolean;
  /** Compaction pressure ladder (persisted across iterations). */
  pressure?: CompactionPressure;
};

// --- helpers ----------------------------------------------------------------

const STATUS_KEY = "amber";

function lastCompactionFromBranch(entries: readonly SessionEntry[]): CompactionEntry | undefined {
  return entries.findLast((entry) => entry.type === "compaction") as CompactionEntry | undefined;
}

function toVerificationSources(messages: readonly Message[]): VerificationSource[] {
  const textOf = (content: unknown): string | undefined => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return undefined;
    const parts: string[] = [];
    for (const block of content) {
      if (typeof block === "object" && block !== null && (block as { type?: string }).type === "text") {
        const text = (block as { text?: string }).text;
        if (typeof text === "string") parts.push(text);
      }
    }
    return parts.join("\n") || undefined;
  };

  const toolCallsOf = (content: unknown): string[] | undefined => {
    if (!Array.isArray(content)) return undefined;
    const calls: string[] = [];
    for (const block of content) {
      if (typeof block !== "object" || block === null) continue;
      const record = block as { type?: string; name?: string; arguments?: unknown };
      if (record.type !== "toolCall") continue;
      const args = record.arguments;
      const argsText = typeof args === "string" ? args : args != null ? JSON.stringify(args) : "";
      calls.push(argsText ? `${record.name ?? ""} ${argsText}` : (record.name ?? ""));
    }
    return calls.length > 0 ? calls : undefined;
  };

  return messages.map((message) => {
    if (message.role === "assistant") {
      return {
        text: textOf(message.content) ?? undefined,
        toolCalls: toolCallsOf(message.content),
      };
    }
    if (message.role === "toolResult") {
      const details = (message as { details?: unknown }).details;
      return { details: details != null ? JSON.stringify(details) : undefined };
    }
    return { content: textOf((message as { content?: unknown }).content) };
  });
}

function resolveSummaryModel(ctx: ExtensionContext, config: AmberConfig) {
  if (config.model) {
    const [provider, ...rest] = config.model.split("/");
    const modelId = rest.join("/");
    if (provider && modelId) {
      const found = ctx.modelRegistry.find(provider, modelId);
      if (found) return found;
    }
  }
  return ctx.model ?? undefined;
}

export default function amber(pi: ExtensionAPI) {
  // Remember the latest system prompt so the summarizer can use it for context
  // (the session_before_compact event does not carry it).
  let cachedSystemPrompt = "";

  pi.on("before_agent_start", (event) => {
    cachedSystemPrompt = event.systemPrompt;
  });

  pi.on("session_before_compact", async (event, ctx) => {
    const config = loadConfig();
    if (!config.enabled) return;

    const { preparation, branchEntries, signal } = event;
    const {
      messagesToSummarize,
      turnPrefixMessages,
      firstKeptEntryId,
      tokensBefore,
      previousSummary,
    } = preparation;

    const allMessages = convertToLlm([...messagesToSummarize, ...turnPrefixMessages]);
    if (allMessages.length === 0) {
      return; // nothing to summarize → default behavior
    }

    const model = resolveSummaryModel(ctx, config);
    if (!model) {
      ctx.ui.notify("pi-amber: no summarization model available, using default compaction", "warning");
      return;
    }

    const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
    if (!auth.ok) {
      ctx.ui.notify(`pi-amber: auth failed (${auth.error}), using default compaction`, "warning");
      return;
    }
    const summaryAuth: SummaryAuth = {
      apiKey: auth.apiKey,
      headers: auth.headers,
      env: auth.env,
    };

    // Iterative context: reuse the previous summary (from the event or the branch).
    const prevCompaction = lastCompactionFromBranch(branchEntries);
    const previousSummarySource = previousSummary || prevCompaction?.summary;

    // Escalation ladder: read the persisted pressure from the previous
    // compaction, let it decay if stale, then decide how aggressive to be.
    const prevDetails = prevCompaction?.details as AmberCompactionDetails | undefined;
    const now = Date.now();
    const pressure = normalizePressure(
      prevDetails?.pressure ?? createCompactionPressure(),
      now,
    );
    const escalate = pressure.level >= 1;

    const summaryLanguage = detectSummaryLanguage(allMessages);

    // --- UI: running phase ---
    ctx.ui.setStatus(
      STATUS_KEY,
      `🟠 compacting ${allMessages.length} msgs · ${tokensBefore.toLocaleString()} tokens`,
    );
    ctx.ui.setWorkingMessage(
      escalate ? `🟠 amber: compacting (level ${pressure.level})…` : "🟠 amber: compacting…",
    );

    try {
      const result = await summarizeConversation({
        model,
        auth: summaryAuth,
        input: {
          messages: allMessages,
          previousSummary: previousSummarySource,
          systemPrompt: cachedSystemPrompt,
          fileOps: preparation.fileOps,
        },
        maxTokens: config.maxTokens,
        signal,
        summaryLanguage,
        recentSources: toVerificationSources(allMessages.slice(-6)),
        escalate,
      });

      const effective = result.summaryChars > 0 && result.payloadTokens < tokensBefore;
      const nextPressure = notePressureAfterCompaction(pressure, {
        effective,
        now: Date.now(),
      });
      const details: AmberCompactionDetails = {
        version: SUMMARY_PROMPT_VERSION,
        summaryLanguage,
        payloadTokens: result.payloadTokens,
        tokensBefore,
        summaryChars: result.summaryChars,
        summarizer: result.summarizerUsage,
        model: `${model.provider}/${model.id}`,
        effective,
        pressure: nextPressure,
      };

      // --- UI: done phase ---
      ctx.ui.setWorkingMessage();
      ctx.ui.setStatus(
        STATUS_KEY,
        `🟠 ✓ ${result.summaryChars.toLocaleString()} chars → ${effective ? "compact" : "still large"}`,
      );

      return {
        compaction: {
          summary: result.summaryText,
          firstKeptEntryId,
          tokensBefore,
          usage: result.summarizerUsage,
          details,
        },
      };
    } catch (error) {
      if (signal.aborted) throw error;
      const message = error instanceof Error ? error.message : String(error);
      // --- UI: failed phase ---
      ctx.ui.setWorkingMessage();
      ctx.ui.setStatus(STATUS_KEY, `🟠 ✗ failed`);
      ctx.ui.notify(`pi-amber: compaction failed (${message}), using default compaction`, "warning");
      return; // fall back to pi's default compaction
    }
  });

  pi.on("session_compact", async (event, ctx) => {
    // Restore the default working indicator and clear the status.
    ctx.ui.setWorkingMessage();
    ctx.ui.setStatus(STATUS_KEY, undefined);
    if (!event.fromExtension) return;
    const details = (event.compactionEntry as CompactionEntry<AmberCompactionDetails>).details;
    if (details) {
      ctx.ui.notify(
        `🟠 amber: context checkpointed · ${details.summaryChars.toLocaleString()} chars · ${details.payloadTokens.toLocaleString()} payload tokens`,
        "info",
      );
    }
  });
}
