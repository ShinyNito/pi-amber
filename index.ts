import type { Message, Usage } from "@earendil-works/pi-ai";
import { appendFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  convertToLlm,
  type CompactionEntry,
  type ExtensionAPI,
  type ExtensionContext,
  type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { loadConfig, type AmberConfig } from "./config.ts";
import { type SummaryAuth, summarizeConversation, toSummaryAuth } from "./summarizer.ts";
import { detectSummaryLanguage } from "./summaryLanguage.ts";
import { type VerificationSource } from "./validate.ts";
import { SUMMARY_PROMPT_VERSION } from "./policy.ts";
import {
  createCompactionPressure,
  normalizePressure,
  notePressureAfterCompaction,
  type CompactionPressure,
} from "./pressure.ts";
import { AmberStatus, formatCompact } from "./ui.ts";

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

function lastCompactionFromBranch(entries: readonly SessionEntry[]): CompactionEntry | undefined {
  return entries.findLast((entry) => entry.type === "compaction") as CompactionEntry | undefined;
}

/** Best-effort failure diagnostics: one JSON line per failure, never throws. */
function debugLogFailure(record: Record<string, unknown>): void {
  try {
    appendFileSync(
      join(homedir(), ".pi", "pi-amber-debug.jsonl"),
      JSON.stringify({ at: new Date().toISOString(), ...record }) + "\n",
    );
  } catch {
    // diagnostics must never break compaction
  }
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

  // Shared across events: the footer spinner started in session_before_compact
  // is stopped in session_compact, so the instance must outlive one handler.
  let status: AmberStatus | undefined;
  const statusFor = (ctx: ExtensionContext): AmberStatus => {
    status ??= new AmberStatus(ctx.ui);
    return status;
  };

  pi.on("session_start", (_event, ctx) => {
    statusFor(ctx).idle();
  });

  pi.on("before_agent_start", (event) => {
    cachedSystemPrompt = event.systemPrompt;
  });

  pi.on("session_before_compact", async (event, ctx) => {
    // The entire handler is guarded: any unexpected error falls back to pi's
    // default compaction with a visible notice instead of silently dying.
    const { preparation, branchEntries, signal } = event;
    let diagnostics: Record<string, unknown> = {};
    try {
      const config = loadConfig();
      if (!config.enabled) {
        ctx.ui.notify("amber: disabled in config, using pi default compaction", "warning");
        return;
      }

      const {
        messagesToSummarize,
        turnPrefixMessages,
        firstKeptEntryId,
        tokensBefore,
        previousSummary,
      } = preparation;

      const allMessages = convertToLlm([...messagesToSummarize, ...turnPrefixMessages]);
      if (allMessages.length === 0) {
        ctx.ui.notify("amber: nothing to summarize, using pi default", "warning");
        return; // nothing to summarize → default behavior
      }

      const model = resolveSummaryModel(ctx, config);
      if (!model) {
        ctx.ui.notify("amber: no model available (ctx.model undefined), using pi default", "warning");
        return;
      }

      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
      if (!auth.ok) {
        ctx.ui.notify(`amber: auth failed (${auth.error}), using pi default`, "warning");
        return;
      }
      const summaryAuth: SummaryAuth = toSummaryAuth(auth);
      // OAuth access tokens rotate on short TTLs (Kimi: ~15 min); a token can
      // be invalidated mid-request when another refresh wins the race. Re-read
      // the registry to pick up the latest credential on auth failures.
      const refreshAuth = async (): Promise<SummaryAuth> => {
        const refreshed = await ctx.modelRegistry.getApiKeyAndHeaders(model);
        if (!refreshed.ok) {
          throw new Error(`auth refresh failed: ${refreshed.error}`);
        }
        return toSummaryAuth(refreshed);
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

      diagnostics = {
        model: `${model.provider}/${model.id}`,
        messages: allMessages.length,
        tokensBefore,
        summaryLanguage,
      };

      // --- UI: running phase (footer spinner; pi's own loader covers the chat area) ---
      statusFor(ctx).startCompaction({
        messages: allMessages.length,
        tokens: tokensBefore,
        level: pressure.level,
      });

      const result = await summarizeConversation({
        model,
        auth: summaryAuth,
        refreshAuth,
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
      if (signal?.aborted) {
        statusFor(ctx).idle();
        throw error;
      }
      const message = error instanceof Error ? error.message : String(error);
      debugLogFailure({ ...diagnostics, error: message });
      // --- UI: failed phase ---
      statusFor(ctx).failed();
      ctx.ui.notify(`amber: compaction failed (${message}), using pi default`, "error");
      return; // fall back to pi's default compaction
    }
  });

  pi.on("session_compact", async (event, ctx) => {
    if (!event.fromExtension) {
      // pi's own compaction ran (fallback or amber disabled) → back to idle.
      statusFor(ctx).idle();
      return;
    }
    const details = (event.compactionEntry as CompactionEntry<AmberCompactionDetails>).details;
    if (!details) {
      statusFor(ctx).idle();
      return;
    }
    statusFor(ctx).done({
      payloadTokens: details.payloadTokens,
      tokensBefore: details.tokensBefore,
      effective: details.effective ?? true,
    });
    const effective = details.effective !== false;
    const trend = effective
      ? `${formatCompact(details.tokensBefore)} → ${formatCompact(details.payloadTokens)} tok`
      : `${formatCompact(details.payloadTokens)} tok · still large`;
    ctx.ui.notify(
      `amber: checkpoint · ${formatCompact(details.summaryChars)} chars · ${trend}`,
      effective ? "info" : "warning",
    );
  });
}
