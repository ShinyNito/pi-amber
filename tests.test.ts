import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { convertToLlm } from "@earendil-works/pi-coding-agent";
import type { Api, Message, Model } from "@earendil-works/pi-ai";
import { estimateTextTokenUnits, estimateTextTokens } from "./tokenLedger.ts";
import { detectSummaryLanguage } from "./summaryLanguage.ts";
import {
  parseCompactionSummaryXml,
  validateCompactionSummary,
  buildVerificationSignals,
  formatSummaryForContext,
} from "./validate.ts";
import {
  serializeForSummary,
  estimateSummaryInputTokens,
  summarizeConversation,
  toSummaryAuth,
  type CompleteFn,
  type SummarizeInput,
} from "./summarizer.ts";
import {
  createCompactionPressure,
  normalizePressure,
  notePressureAfterCompaction,
  PRESSURE_DECAY_WINDOW_MS,
} from "./pressure.ts";

// ---------------------------------------------------------------------------
// tokenLedger (plain-text estimation, used by validation)
// ---------------------------------------------------------------------------

describe("tokenLedger", () => {
  test("estimates latin text at ~chars/4", () => {
    const text = "hello world this is a test message";
    const units = estimateTextTokenUnits(text);
    assert.ok(units > 0);
    assert.ok(units <= text.length);
    assert.ok(Math.abs(units - 8) < 2, `expected ~8 tokens, got ${units}`);
  });

  test("counts CJK denser than latin", () => {
    const cjk = "你好世界，这是一个测试消息".repeat(10);
    const latin = "a".repeat(cjk.length);
    assert.ok(estimateTextTokenUnits(cjk) > estimateTextTokenUnits(latin) * 2);
  });

  test("estimate is additive across splits", () => {
    const a = "hello 世界 " + "x".repeat(50);
    const b = "more text 测试 " + "y".repeat(30);
    assert.equal(estimateTextTokenUnits(a + b), estimateTextTokenUnits(a) + estimateTextTokenUnits(b));
  });

  test("empty text → 0", () => {
    assert.equal(estimateTextTokens("   "), 0);
    assert.equal(estimateTextTokenUnits(""), 0);
  });
});

// ---------------------------------------------------------------------------
// summaryLanguage
// ---------------------------------------------------------------------------

function userMsg(content: string): Message {
  return { role: "user", content, timestamp: Date.now() } as Message;
}

describe("summaryLanguage", () => {
  test("detects chinese from CJK-dominant messages", () => {
    assert.equal(
      detectSummaryLanguage([
        userMsg("帮我重构这个模块，注意保持向后兼容。"),
        userMsg("顺便加个测试。"),
      ]),
      "Chinese",
    );
  });

  test("returns undefined for latin-dominant", () => {
    assert.equal(
      detectSummaryLanguage([userMsg("Please refactor this module.")]),
      undefined,
    );
  });

  test("returns undefined for too little sample", () => {
    assert.equal(detectSummaryLanguage([userMsg("好")]), undefined);
  });

  test("detects japanese when kana present", () => {
    assert.equal(
      detectSummaryLanguage([userMsg("このモジュールをリファクタリングしてください。")]),
      "Japanese",
    );
  });

  test("detects korean when hangul dominant", () => {
    assert.equal(
      detectSummaryLanguage([userMsg("이 모듈을 리팩토링하고 테스트도 추가해 주세요.")]),
      "Korean",
    );
  });
});

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

describe("validate", () => {
  const validXml = `<summary>
<task>Refactor the auth module</task>
<constraints>
- keep backward compatibility
</constraints>
<state>Mostly done, tests pending</state>
<artifacts>
- [file] src/auth.ts | modified
- [file] src/auth.test.ts | created
</artifacts>
<decisions>
- use JWT — user requirement
</decisions>
<dead_ends>
- tried bcrypt — too slow
</dead_ends>
<knowledge>
- pnpm workspaces need hoisting
</knowledge>
<open_loops>
- confirm token expiry policy
</open_loops>
<next_steps>
1. write tests
2. run pnpm test
</next_steps>
<breadcrumbs>
- src/auth.ts
</breadcrumbs>
</summary>`;

  test("parses summary xml into sections", () => {
    const parsed = parseCompactionSummaryXml(validXml);
    assert.equal(parsed.task, "Refactor the auth module");
    assert.match(parsed.artifacts, /src\/auth\.ts/);
    assert.match(parsed.next_steps, /write tests/);
  });

  test("strips code fences", () => {
    const fenced = "```xml\n" + validXml + "\n```";
    assert.equal(parseCompactionSummaryXml(fenced).task, "Refactor the auth module");
  });

  test("valid summary passes validation", () => {
    const text = validateCompactionSummary(validXml, 1000, []);
    assert.match(text, /## Task/);
    assert.match(text, /## Next Steps/);
  });

  test("missing required tags fails validation", () => {
    assert.throws(
      () => validateCompactionSummary("<summary><task>x</task></summary>", 1000, []),
      /missing <state>/,
    );
  });

  test("missing technical refs fails when signals required", () => {
    const signals = ["src/auth.ts", "pnpm test"];
    const summaryWithoutRefs = `<summary>
<task>Fix a bug</task>
<state>Done</state>
<next_steps>1. ship it</next_steps>
<artifacts>
- [file] something/else.ts | modified
</artifacts>
</summary>`;
    assert.throws(
      () => validateCompactionSummary(summaryWithoutRefs, 1000, signals),
      /verification pass missing recent technical refs/,
    );
  });

  test("summary containing refs passes signal check", () => {
    const text = validateCompactionSummary(validXml, 1000, ["src/auth.ts"]);
    assert.match(text, /## Task/);
  });

  test("too-short summary rejected on large source", () => {
    assert.throws(
      () =>
        validateCompactionSummary(
          "<summary><task>t</task><state>s</state><next_steps>1</next_steps><artifacts>- [file] a.ts | read</artifacts></summary>",
          5000,
          [],
        ),
      /summary too short/,
    );
  });

  test("buildVerificationSignals extracts paths and commands", () => {
    const signals = buildVerificationSignals([
      { content: "check src/lib/utils.ts and run pnpm test" },
      { text: "modified config.json" },
      { toolCalls: ['read path="src/main.ts"'] },
    ]);
    assert.ok(signals.length > 0);
    assert.ok(signals.some((s) => s.includes("utils.ts")));
  });

  test("formatSummaryForContext renders markdown", () => {
    const parsed = parseCompactionSummaryXml(validXml);
    const text = formatSummaryForContext(parsed);
    assert.match(text, /## Constraints/);
    assert.match(text, /## Dead Ends/);
  });
});

// ---------------------------------------------------------------------------
// summarizer: official serializer integration
// ---------------------------------------------------------------------------

describe("summarizer", () => {
  function makeMessages(count: number): Message[] {
    const messages: Message[] = [];
    for (let i = 0; i < count; i += 1) {
      if (i % 3 === 0) {
        messages.push({
          role: "user",
          content: `user message ${i} — 中文内容测试`,
          timestamp: Date.now() + i,
        } as Message);
      } else if (i % 3 === 1) {
        messages.push({
          role: "assistant",
          content: [{ type: "text", text: `assistant reply ${i}` }],
          timestamp: Date.now() + i,
        } as Message);
      } else {
        messages.push({
          role: "toolResult",
          toolCallId: `call-${i}`,
          toolName: "read",
          isError: false,
          content: [{ type: "text", text: `output ${i}` + "x".repeat(100) }],
          timestamp: Date.now() + i,
        } as Message);
      }
    }
    return messages;
  }

  test("serializeForSummary uses pi's serializer format", () => {
    const input: SummarizeInput = {
      messages: makeMessages(3),
    };
    const text = serializeForSummary(input);
    assert.match(text, /\[User\]: user message 0/);
    assert.match(text, /\[Assistant\]: assistant reply 1/);
    assert.match(text, /\[Tool result\]:/);
  });

  test("serializeForSummary includes previous summary and next message", () => {
    const text = serializeForSummary({
      messages: makeMessages(1),
      previousSummary: "previous context",
      nextUserMessage: "next instruction",
    });
    assert.match(text, /<previous-summary>/);
    assert.match(text, /previous context/);
    assert.match(text, /<next-user-message>/);
    assert.match(text, /next instruction/);
  });

  test("estimateSummaryInputTokens is positive and grows with input", () => {
    const small = estimateSummaryInputTokens({ messages: makeMessages(3) });
    const large = estimateSummaryInputTokens({ messages: makeMessages(30) });
    assert.ok(small > 0);
    assert.ok(large > small);
  });

  test("estimateSummaryInputTokens includes previous summary", () => {
    const base = estimateSummaryInputTokens({ messages: makeMessages(3) });
    const withPrev = estimateSummaryInputTokens({
      messages: makeMessages(3),
      previousSummary: "x".repeat(400),
    });
    assert.ok(withPrev > base);
  });

  test("convertToLlm round-trips agent messages", () => {
    const llm = convertToLlm(makeMessages(3));
    assert.equal(llm.length, 3);
    assert.deepEqual(
      llm.map((m) => m.role),
      ["user", "assistant", "toolResult"],
    );
  });

  test("serializeForSummary includes deterministic file ops", () => {
    const text = serializeForSummary({
      messages: makeMessages(1),
      fileOps: {
        read: new Set(["src/a.ts", "src/b.ts"]),
        written: new Set(["out/result.json"]),
        edited: new Set(["src/a.ts"]),
      },
    });
    assert.match(text, /<read-files>/);
    assert.match(text, /src\/a\.ts/);
    assert.match(text, /<written-files>/);
    assert.match(text, /<edited-files>/);
  });

  test("serializeForSummary omits empty file ops", () => {
    const text = serializeForSummary({
      messages: makeMessages(1),
      fileOps: { read: new Set(), written: new Set(), edited: new Set() },
    });
    assert.doesNotMatch(text, /<read-files>/);
  });

  test("summarizeConversation pre-trims when payload exceeds context window", async () => {
    // A tiny context window forces the pre-trim path: the first request must
    // already be trimmed down (not the full 200-message payload).
    const fakeModel = {
      provider: "test",
      id: "tiny",
      api: {},
      contextWindow: 2_000,
      maxTokens: 1_024,
    } as unknown as Model<Api>;
    const input = { messages: makeMessages(200) };
    const payloadTokens = estimateSummaryInputTokens(input);
    assert.ok(payloadTokens > 2_000, "fixture must exceed the window");

    const complete: CompleteFn = async (model, context, options) => {
      // The serialized payload is embedded in the single user message.
      const serialized = context.messages[0].content as string;
      // Pre-trim keeps the tail (recent work), drops the head.
      assert.ok(serialized.includes("user message 198"), "tail retained");
      assert.ok(!serialized.includes("user message 0"), "head dropped");
      return {
        content: [
          {
            type: "text",
            text: [
              "<summary>",
              "<task>Handle the recent work.</task>",
              "<constraints>",
              "- keep it simple",
              "</constraints>",
              "<state>Tail messages retained.</state>",
              "<artifacts>",
              "- [file] /tmp/a.ts | created",
              "</artifacts>",
              "<decisions>",
              "- proceed",
              "</decisions>",
              "<dead_ends>",
              "- none",
              "</dead_ends>",
              "<knowledge>",
              "- none",
              "</knowledge>",
              "<open_loops>",
              "- none",
              "</open_loops>",
              "<next_steps>",
              "- continue",
              "</next_steps>",
              "<breadcrumbs>",
              "- /tmp",
              "</breadcrumbs>",
              "</summary>",
            ].join("\n"),
          },
        ],
        usage: {
          input: 100,
          output: 50,
          totalTokens: 150,
          cost: { input: 0, output: 0, total: 0 },
        },
      };
    };

    const result = await summarizeConversation({
      model: fakeModel,
      auth: {},
      input,
      maxTokens: 1_024,
      recentSources: [],
      complete,
    });
    assert.ok(result.summaryText.includes("Handle the recent work"));
    // payloadTokens still reflects the FULL conversation (stats口径).
    assert.equal(result.payloadTokens, payloadTokens);
  });

  const validXmlFixture = `<summary>
<task>Continue the refactor.</task>
<state>Halfway done.</state>
<artifacts>
- [file] /tmp/a.ts | modified
</artifacts>
<next_steps>
1. finish the refactor
</next_steps>
</summary>`;

  test("empty output with stopReason length retries with doubled maxTokens", async () => {
    const fakeModel = {
      provider: "test",
      id: "reasoner",
      api: {},
      contextWindow: 128_000,
    } as unknown as Model<Api>;

    const seenMaxTokens: number[] = [];
    const complete: CompleteFn = async (_model, _context, options) => {
      seenMaxTokens.push(options.maxTokens);
      if (seenMaxTokens.length === 1) {
        // Reasoning model burned the whole budget on thinking → empty text.
        return { content: [{ type: "text", text: "" }], stopReason: "length" };
      }
      return { content: [{ type: "text", text: validXmlFixture }], stopReason: "stop" };
    };

    const result = await summarizeConversation({
      model: fakeModel,
      auth: {},
      input: { messages: makeMessages(1) },
      maxTokens: 1_024,
      recentSources: [],
      complete,
    });
    assert.deepEqual(seenMaxTokens, [1_024, 2_048]);
    assert.ok(result.summaryText.includes("Continue the refactor"));
  });

  test("persistent empty output throws an actionable error", async () => {
    const fakeModel = {
      provider: "test",
      id: "broken",
      api: {},
      contextWindow: 128_000,
    } as unknown as Model<Api>;

    const complete: CompleteFn = async () => ({
      content: [{ type: "text", text: "   " }],
      stopReason: "stop",
    });

    await assert.rejects(
      () =>
        summarizeConversation({
          model: fakeModel,
          auth: {},
          input: { messages: makeMessages(1) },
          maxTokens: 1_024,
          recentSources: [],
          complete,
        }),
      /empty output/,
    );
  });

  test("toSummaryAuth promotes bearer header to apiKey", () => {
    const result = toSummaryAuth({ headers: { Authorization: "Bearer jwt-token-123", "X-Other": "keep" } });
    assert.equal(result.apiKey, "jwt-token-123");
    assert.deepEqual(result.headers, { "X-Other": "keep" });
  });

  test("toSummaryAuth leaves explicit apiKey untouched", () => {
    const result = toSummaryAuth({ apiKey: "direct-key", headers: { Authorization: "Bearer other" } });
    assert.equal(result.apiKey, "direct-key");
    assert.deepEqual(result.headers, { Authorization: "Bearer other" });
  });

  test("toSummaryAuth drops empty header map", () => {
    const result = toSummaryAuth({ headers: { authorization: "bearer  spaced-token " } });
    assert.equal(result.apiKey, "spaced-token");
    assert.equal(result.headers, undefined);
  });

  test("provider error responses surface errorMessage and reject", async () => {
    const fakeModel = {
      provider: "test",
      id: "erroneous",
      api: {},
      contextWindow: 128_000,
    } as unknown as Model<Api>;

    let calls = 0;
    const complete: CompleteFn = async () => {
      calls += 1;
      return {
        content: [],
        stopReason: "error",
        errorMessage: "rate limit exceeded, try again later",
      };
    };

    await assert.rejects(
      () =>
        summarizeConversation({
          model: fakeModel,
          auth: {},
          input: { messages: makeMessages(1) },
          maxTokens: 1_024,
          recentSources: [],
          complete,
        }),
      /provider error: rate limit exceeded/,
    );
    // "rate limit" is non-retryable → no retry attempts.
    assert.equal(calls, 1);
  });

  test("401 provider error refreshes auth and retries once", async () => {
    const fakeModel = {
      provider: "test",
      id: "oauth-model",
      api: {},
      contextWindow: 128_000,
    } as unknown as Model<Api>;

    const seenApiKeys: (string | undefined)[] = [];
    let calls = 0;
    const complete: CompleteFn = async (_model, _context, options) => {
      calls += 1;
      seenApiKeys.push(options.apiKey);
      if (calls === 1) {
        return {
          content: [],
          stopReason: "error",
          errorMessage: '401 {"error":{"type":"authentication_error","message":"The API Key appears to be invalid"}}',
        };
      }
      return { content: [{ type: "text", text: validXmlFixture }], stopReason: "stop" };
    };

    let refreshes = 0;
    const result = await summarizeConversation({
      model: fakeModel,
      auth: { apiKey: "stale-token" },
      refreshAuth: async () => {
        refreshes += 1;
        return { apiKey: "fresh-token" };
      },
      input: { messages: makeMessages(1) },
      maxTokens: 1_024,
      recentSources: [],
      complete,
    });
    assert.equal(refreshes, 1);
    assert.equal(calls, 2);
    assert.deepEqual(seenApiKeys, ["stale-token", "fresh-token"]);
    assert.ok(result.summaryText.includes("Continue the refactor"));
  });
});

// ---------------------------------------------------------------------------
// pressure (escalation ladder)
// ---------------------------------------------------------------------------

describe("pressure", () => {
  test("starts at level 0", () => {
    const p = createCompactionPressure();
    assert.equal(p.level, 0);
    assert.equal(p.consecutiveIneffective, 0);
  });

  test("ineffective compaction raises level", () => {
    let p = createCompactionPressure();
    p = notePressureAfterCompaction(p, { effective: false, now: 1000 });
    assert.equal(p.level, 1);
    assert.equal(p.consecutiveIneffective, 1);
    p = notePressureAfterCompaction(p, { effective: false, now: 2000 });
    assert.equal(p.level, 2);
    // level 2 is the cap
    p = notePressureAfterCompaction(p, { effective: false, now: 3000 });
    assert.equal(p.level, 2);
  });

  test("effective compaction resets streak", () => {
    let p = createCompactionPressure();
    p = notePressureAfterCompaction(p, { effective: false, now: 1000 });
    p = notePressureAfterCompaction(p, { effective: true, now: 2000 });
    assert.equal(p.level, 0);
    assert.equal(p.consecutiveIneffective, 0);
  });

  test("pressure decays after idle window", () => {
    let p = createCompactionPressure();
    p = notePressureAfterCompaction(p, { effective: false, now: 1000 });
    const now = 1000 + PRESSURE_DECAY_WINDOW_MS + 1;
    const normalized = normalizePressure(p, now);
    assert.equal(normalized.level, 0);
    assert.equal(normalized.consecutiveIneffective, 0);
  });

  test("fresh pressure is not decayed", () => {
    let p = createCompactionPressure();
    p = notePressureAfterCompaction(p, { effective: false, now: 1000 });
    const normalized = normalizePressure(p, 1000 + PRESSURE_DECAY_WINDOW_MS - 1);
    assert.equal(normalized.level, 1);
  });
});
