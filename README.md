# pi-amber 🟠

**Context compaction with structured summaries, validation, and self-repair** — a [pi](https://pi.dev) extension that replaces the default context compression with an amber-grade checkpoint system.

When your conversation grows too long, pi compacts it into a summary. `pi-amber` makes that summary **structured, verified, and language-aware** — so the next model can continue the work without losing decisions, dead ends, or exact file references.

## Features

- **Structured XML checkpoints** — dense handoff documents with `task / constraints / state / artifacts / decisions / dead_ends / knowledge / open_loops / next_steps / breadcrumbs`
- **Anti-hallucination validation** — summaries must keep required sections and at least one recent technical reference (paths/commands) verbatim; failures trigger an automatic self-repair pass
- **Self-repair pipeline** — overflow → shrink input and retry; transient errors → backoff retry; validation failure → feed the invalid output back for one repair
- **Language-aware summaries** — detects CJK-dominant conversations (Chinese / Japanese / Korean) and writes the summary in the user's language
- **Pressure escalation ladder** — consecutive ineffective compactions raise the trim aggressiveness (persisted across compactions), decaying after 5 idle minutes
- **Deterministic file ledger** — pi's extracted `read/written/edited` file operations are injected into the summarizer context, so artifacts never rely on the model's memory alone
- **Iterative context** — the previous summary is passed to the next compaction
- **Uses pi's own machinery** — `convertToLlm`, `serializeConversation`, `estimateTokens`, model registry auth — no re-implemented infrastructure

## Install

```bash
pi install npm:@shinynito/pi-amber
```

Or run once without installing:

```bash
pi -e npm:@shinynito/pi-amber
```

## Configuration

Optional config at `~/.pi/pi-amber.json`:

```json
{
  "model": "google/gemini-2.5-flash",
  "maxTokens": 8192,
  "enabled": true
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `""` (current session model) | Provider/model used for summarization, e.g. `"google/gemini-2.5-flash"`. Empty uses the active session model. |
| `maxTokens` | `8192` | Max output tokens for the summary request. |
| `enabled` | `true` | Set `false` to fall back to pi's default compaction entirely. |

## How it works

```
session grows past threshold
      │
      ▼
session_before_compact (pi event)
      │
      ├─ convertToLlm + serializeConversation (pi official)
      ├─ inject: system prompt · file ledger · previous summary
      ├─ detect summary language (CJK-aware)
      │
      ▼
summarizer (recovery pipeline)
   overflow → shrink input (keep tail) → retry
   transient → backoff retry
   invalid  → self-repair (feed error back)
      │
      ▼
validate
   ✓ required sections present
   ✓ artifacts format `- [kind] ref | status`
   ✓ recent technical refs preserved
   ✓ not too short (CJK-aware token estimate)
      │
      ▼
CompactionEntry with { summary, usage, details }
   details carry pressure state for the next round
```

## Design notes

- **Why structured XML?** A fixed schema makes validation possible: we can *prove* the summary kept the required sections and didn't hallucinate away recent file references — then ask the model to fix it if it did.
- **Why keep the tail?** When the input must shrink, recent work is the most valuable context for continuing; the head is already covered by the previous summary.
- **Why a pressure ladder instead of a hard cap?** Compaction counts are a poor proxy for health. Tracking *ineffective* compactions and escalating trim aggressiveness keeps the session alive without ever hard-blocking the user.

## Development

```bash
pnpm install
pnpm typecheck        # tsc --noEmit
pnpm test             # node --experimental-strip-types --test tests.test.ts
```

## Files

```
index.ts           # extension entry: before_agent_start + session_before_compact + session_compact
policy.ts          # summarizer system prompt (XML schema + security + language rule)
summarizer.ts      # request pipeline: serialize → recover (shrink/retry/repair)
validate.ts        # XML parsing + structural/technical-ref validation
pressure.ts        # escalation ladder (persisted across compactions)
summaryLanguage.ts # CJK-dominant language detection
tokenLedger.ts     # plain-text CJK-aware token estimation (for validation)
config.ts          # ~/.pi/pi-amber.json loader
```

## License

MIT
