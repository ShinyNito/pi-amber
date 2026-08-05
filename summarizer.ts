import { completeSimple, type Context, type Model, type UserMessage } from "@earendil-works/pi-ai/compat";
import type { Api, Message, Usage } from "@earendil-works/pi-ai";
import { randomUUID } from "node:crypto";
import { convertToLlm, estimateTokens, serializeConversation } from "@earendil-works/pi-coding-agent";
import { buildRepairPromptText, buildSummarySystemPrompt } from "./policy.ts";
import { buildVerificationSignals, type VerificationSource, validateCompactionSummary } from "./validate.ts";
import { estimateTextTokenUnits } from "./tokenLedger.ts";

export type SummarizeResult = {
  summaryText: string;
  responseId?: string;
  timestamp: number;
  summarizerUsage?: Usage;
  /** Serialized conversation token estimate (before any shrink). */
  payloadTokens: number;
  summaryChars: number;
};

export type SummaryAuth = {
  apiKey?: string;
  headers?: Record<string, string>;
  env?: Record<string, string>;
};

export type CompleteFn = (
  model: Model<Api>,
  context: Context,
  options: {
    apiKey?: string;
    headers?: Record<string, string>;
    env?: Record<string, string>;
    maxTokens: number;
    signal?: AbortSignal;
    sessionId: string;
    cacheRetention: "none";
  },
) => Promise<{
  content: { type: string; text?: string }[];
  usage?: Usage;
  responseId?: string;
  timestamp?: number;
}>;

const defaultComplete: CompleteFn = async (model, context, options) => {
  const response = await completeSimple(model, context, {
    apiKey: options.apiKey,
    headers: options.headers,
    env: options.env,
    maxTokens: options.maxTokens,
    signal: options.signal,
    sessionId: options.sessionId,
    cacheRetention: options.cacheRetention,
  });
  return {
    content: response.content as { type: string; text?: string }[],
    usage: response.usage,
    responseId: response.responseId,
    timestamp: response.timestamp,
  };
};

function createCompactionAbortError() {
  const error = new Error("compaction aborted");
  error.name = "AbortError";
  return error;
}

function sleepWithAbort(ms: number, signal?: AbortSignal) {
  if (!Number.isFinite(ms) || ms <= 0) return;
  if (signal?.aborted) throw createCompactionAbortError();

  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      reject(createCompactionAbortError());
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function getRetryDelayMs(attempt: number) {
  return Math.min(1_500, 400 * 2 ** Math.max(0, attempt));
}

function isOverflowError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return /context|token|too long|maximum context|input.*too large|overflow/i.test(message);
}

function isNonRetryableError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return /unauthorized|authentication|invalid api key|quota|rate limit|insufficient|forbidden/i.test(
    message,
  );
}

function isTransientError(error: unknown) {
  if (isNonRetryableError(error)) return false;
  const message = error instanceof Error ? error.message : String(error);
  return /timeout|timed out|network|socket|econn|5\d\d|temporar/i.test(message);
}

export type SummarizeInput = {
  /** Messages to summarize (agent messages, converted via convertToLlm). */
  messages: Message[];
  /** Previous compaction summary, if any (iterative context). */
  previousSummary?: string;
  /** Incoming user text not yet part of the conversation. */
  nextUserMessage?: string;
  /** System prompt of the main session, for context. */
  systemPrompt?: string;
  /** Deterministic file operations extracted by pi (read/written/edited). */
  fileOps?: { read: Set<string>; written: Set<string>; edited: Set<string> };
};

function formatFileOperations(fileOps: NonNullable<SummarizeInput["fileOps"]>): string {
  const sections: string[] = [];
  if (fileOps.read.size > 0) {
    sections.push(`<read-files>\n${[...fileOps.read].join("\n")}\n</read-files>`);
  }
  if (fileOps.written.size > 0) {
    sections.push(`<written-files>\n${[...fileOps.written].join("\n")}\n</written-files>`);
  }
  if (fileOps.edited.size > 0) {
    sections.push(`<edited-files>\n${[...fileOps.edited].join("\n")}\n</edited-files>`);
  }
  return sections.length > 0 ? sections.join("\n\n") : "";
}

/** Serialize the conversation with pi's official serializer. */
export function serializeForSummary(input: SummarizeInput): string {
  const parts: string[] = [];
  if (input.systemPrompt) {
    parts.push(`<system-prompt>
${input.systemPrompt}
</system-prompt>`);
  }
  parts.push(serializeConversation(input.messages));
  if (input.fileOps) {
    const fileBlock = formatFileOperations(input.fileOps);
    if (fileBlock) parts.push(fileBlock);
  }
  if (input.previousSummary) {
    parts.push(`<previous-summary>
${input.previousSummary}
</previous-summary>`);
  }
  if (input.nextUserMessage) {
    parts.push(`<next-user-message>
${input.nextUserMessage}
</next-user-message>`);
  }
  return parts.join("\n\n");
}

/** Estimate tokens of the serialized conversation using pi's estimator. */
export function estimateSummaryInputTokens(input: SummarizeInput): number {
  let total = 0;
  for (const message of input.messages) total += estimateTokens(message);
  if (input.previousSummary) total += Math.ceil(estimateTextTokenUnits(input.previousSummary));
  if (input.nextUserMessage) total += Math.ceil(estimateTextTokenUnits(input.nextUserMessage));
  if (input.systemPrompt) total += Math.ceil(estimateTextTokenUnits(input.systemPrompt));
  return total;
}

type SummarizerRequest = {
  model: Model<Api>;
  auth: SummaryAuth;
  serialized: string;
  maxTokens: number;
  signal?: AbortSignal;
  complete: CompleteFn;
  summaryLanguage?: string;
  repair?: { invalidOutput: string; validationError: string; verificationSignals: string[] };
};

/** Extract joined text content from a completion response. */
function extractText(response: Awaited<ReturnType<CompleteFn>>): string {
  return response.content
    .filter((c) => c.type === "text" && typeof c.text === "string")
    .map((c) => c.text as string)
    .join("\n");
}

/** Keep the tail (recent work) of the messages; drop the head. Never below 6. */
function shrinkInput(input: SummarizeInput): SummarizeInput | null {
  const messages = input.messages;
  if (messages.length <= 6) return null;
  const keepTail = Math.max(4, Math.floor(messages.length / 2));
  return { ...input, messages: messages.slice(messages.length - keepTail) };
}

async function requestSummary(params: SummarizerRequest): Promise<Awaited<ReturnType<CompleteFn>>> {
  const messages: Context["messages"] = [
    {
      role: "user",
      content: `<conversation>\n${params.serialized}\n</conversation>`,
      timestamp: Date.now(),
    } as UserMessage,
  ];
  if (params.repair) {
    messages.push(
      {
        role: "assistant",
        content: [{ type: "text", text: params.repair.invalidOutput }],
        timestamp: Date.now() + 1,
        api: "pi-amber-compaction",
        provider: params.model.provider,
        model: params.model.id,
        stopReason: "stop",
      } as Context["messages"][number],
      {
        role: "user",
        content: buildRepairPromptText(
          params.repair.validationError,
          params.repair.verificationSignals,
        ),
        timestamp: Date.now() + 2,
      } as UserMessage,
    );
  }

  return params.complete(params.model, {
    systemPrompt: buildSummarySystemPrompt(params.summaryLanguage),
    messages,
  }, {
    apiKey: params.auth.apiKey,
    headers: params.auth.headers,
    env: params.auth.env,
    maxTokens: params.maxTokens,
    signal: params.signal,
    // Summaries are one-off prompts unlikely to be reused; skip prompt-cache writes.
    sessionId: randomUUID(),
    cacheRetention: "none",
  });
}

/**
 * Summary request with a recovery pipeline: overflow → shrink input and retry
 * (once); transient error → backoff retry (once); validation failure → feed the
 * invalid output back for one self-repair. All attempts check abort.
 */
export async function summarizeConversation(params: {
  model: Model<Api>;
  auth: SummaryAuth;
  input: SummarizeInput;
  maxTokens: number;
  signal?: AbortSignal;
  complete?: CompleteFn;
  summaryLanguage?: string;
  recentSources: VerificationSource[];
  /** Escalate: previous compaction was ineffective → trim input more aggressively up front. */
  escalate?: boolean;
}): Promise<SummarizeResult> {
  const complete = params.complete ?? defaultComplete;
  const verificationSignals = buildVerificationSignals(params.recentSources);
  const payloadTokens = estimateSummaryInputTokens(params.input);

  // Start from the full input, but when escalated, pre-trim to roughly half to
  // avoid a guaranteed overflow on the first request.
  let input = params.escalate ? (shrinkInput(params.input) ?? params.input) : params.input;
  let serialized = serializeForSummary(input);
  let networkRetryUsed = false;

  const tryShrink = (): boolean => {
    const shrunk = shrinkInput(input);
    if (!shrunk) return false;
    const shrunkText = serializeForSummary(shrunk);
    if (shrunkText.length >= serialized.length) return false;
    input = shrunk;
    serialized = shrunkText;
    return true;
  };

  while (true) {
    let response: Awaited<ReturnType<CompleteFn>>;
    try {
      response = await requestSummary({
        model: params.model,
        auth: params.auth,
        serialized,
        maxTokens: params.maxTokens,
        signal: params.signal,
        complete,
        summaryLanguage: params.summaryLanguage,
      });
    } catch (error) {
      if (params.signal?.aborted) throw error;
      if (isOverflowError(error) && tryShrink()) continue;
      if (!networkRetryUsed && isTransientError(error)) {
        networkRetryUsed = true;
        await sleepWithAbort(getRetryDelayMs(0), params.signal);
        continue;
      }
      throw error;
    }

    const text = extractText(response);

    const finalize = (rawText: string): SummarizeResult => {
      const summaryText = validateCompactionSummary(
        rawText,
        payloadTokens,
        verificationSignals,
      );
      return {
        summaryText,
        responseId: response.responseId,
        timestamp: response.timestamp ?? Date.now(),
        summarizerUsage: response.usage,
        payloadTokens,
        summaryChars: summaryText.length,
      };
    };

    try {
      return finalize(text);
    } catch (validationError) {
      if (params.signal?.aborted) throw validationError;
      try {
        const repaired = await requestSummary({
          model: params.model,
          auth: params.auth,
          serialized,
          maxTokens: params.maxTokens,
          signal: params.signal,
          complete,
          summaryLanguage: params.summaryLanguage,
          repair: {
            invalidOutput: text.trim(),
            validationError:
              validationError instanceof Error ? validationError.message : String(validationError),
            verificationSignals,
          },
        });
        return finalize(extractText(repaired));
      } catch (repairError) {
        if (params.signal?.aborted) throw repairError;
        if (isOverflowError(repairError) && tryShrink()) continue;
        throw repairError;
      }
    }
  }
}
