import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

type Ui = ExtensionContext["ui"];

const STATUS_KEY = "amber";

/** Braille frames for the footer progress spinner. */
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_INTERVAL_MS = 120;

/** Compact number for the footer: 842 → "842", 1_250 → "1.3k", 12_500 → "13k". */
export function formatCompact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 10_000) return `${Math.round(value / 1000)}k`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return String(value);
}

/**
 * Themed footer status for pi-amber.
 *
 * The footer is the only extension UI surface visible during compaction:
 * pi shows its own compaction loader (CompactionStatusIndicator), so
 * setWorkingMessage is never displayed while the summarizer runs. All
 * progress and result feedback therefore goes through setStatus here.
 */
export class AmberStatus {
  private readonly ui: Ui;
  private timer: ReturnType<typeof setInterval> | undefined;
  private frame = 0;

  constructor(ui: Ui) {
    this.ui = ui;
  }

  /** Subtle idle badge shown while the session is active. */
  idle(): void {
    this.stopSpinner();
    const t = this.ui.theme;
    this.ui.setStatus(STATUS_KEY, `${this.dot("warning")} ${t.fg("dim", "amber")}`);
  }

  /** Animated progress shown while the summarizer runs. */
  startCompaction(info: { messages: number; tokens: number; level: number }): void {
    this.stopSpinner();
    const render = () => {
      const t = this.ui.theme;
      const spinner = t.fg("warning", SPINNER_FRAMES[this.frame % SPINNER_FRAMES.length]);
      this.frame += 1;
      const details = [`${info.messages} msgs`, `${formatCompact(info.tokens)} tok`];
      const suffix = info.level >= 1 ? ` ${t.fg("warning", `L${info.level}`)}` : "";
      this.ui.setStatus(
        STATUS_KEY,
        `${spinner} ${t.fg("muted", "compacting")} ${t.fg("dim", details.join(" · "))}${suffix}`,
      );
    };
    render();
    this.timer = setInterval(render, SPINNER_INTERVAL_MS);
    this.timer.unref?.();
  }

  /** Persistent result shown after an amber compaction completes. */
  done(info: { payloadTokens: number; tokensBefore: number; effective: boolean }): void {
    this.stopSpinner();
    const t = this.ui.theme;
    const glyph = this.dot(info.effective ? "success" : "warning");
    const trend = info.effective
      ? `${formatCompact(info.tokensBefore)} → ${formatCompact(info.payloadTokens)} tok`
      : `${formatCompact(info.payloadTokens)} tok ${t.fg("warning", "· still large")}`;
    this.ui.setStatus(STATUS_KEY, `${glyph} ${t.fg("dim", trend)}`);
  }

  /** Transient failure state (replaced by idle once pi's fallback runs). */
  failed(): void {
    this.stopSpinner();
    const t = this.ui.theme;
    this.ui.setStatus(STATUS_KEY, `${this.dot("error")} ${t.fg("dim", "compaction failed")}`);
  }

  clear(): void {
    this.stopSpinner();
    this.ui.setStatus(STATUS_KEY, undefined);
  }

  private dot(color: "warning" | "success" | "error"): string {
    return this.ui.theme.fg(color, "●");
  }

  private stopSpinner(): void {
    if (this.timer !== undefined) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }
}
