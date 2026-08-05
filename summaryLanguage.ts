// Detect the dominant language of recent user messages so the summary can be
// written in the user's own language instead of always defaulting to English.
import type { Message } from "@earendil-works/pi-ai";

const MAX_SCANNED_CHARS = 4_000;
// If CJK characters make up at least this share of letter-class characters,
// the conversation is considered CJK-dominant. Technical conversations mix
// lots of Latin identifiers, so the threshold should not be too high.
const CJK_DOMINANCE_THRESHOLD = 0.25;
// With too little sample we keep the default (English) summary.
const MIN_SCANNED_LETTERS = 8;

type ScriptCounts = {
  han: number;
  kana: number;
  hangul: number;
  latin: number;
};

function tallyScripts(text: string, counts: ScriptCounts) {
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    if ((code >= 0x3400 && code <= 0x9fff) || (code >= 0xf900 && code <= 0xfaff)) {
      counts.han += 1;
    } else if (code >= 0x3040 && code <= 0x30ff) {
      counts.kana += 1;
    } else if (
      (code >= 0xac00 && code <= 0xd7af) ||
      (code >= 0x1100 && code <= 0x11ff) ||
      (code >= 0x3130 && code <= 0x318f)
    ) {
      counts.hangul += 1;
    } else if ((code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a)) {
      counts.latin += 1;
    }
  }
}

function userMessageText(message: Message): string | undefined {
  if (message.role !== "user") return undefined;
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
        const text = (block as { text?: string }).text;
        if (typeof text === "string") parts.push(text);
      }
    }
    return parts.join("\n") || undefined;
  }
  return undefined;
}

/**
 * Scan the most recent user messages (up to MAX_SCANNED_CHARS) and return the
 * summary language as an English name ("Chinese", "Japanese", "Korean"), or
 * undefined to keep the default English summary.
 */
export function detectSummaryLanguage(messages: readonly Message[]): string | undefined {
  const counts: ScriptCounts = { han: 0, kana: 0, hangul: 0, latin: 0 };
  let scannedChars = 0;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (scannedChars >= MAX_SCANNED_CHARS) break;
    const text = userMessageText(messages[index]);
    if (!text) continue;
    const slice = text.slice(0, MAX_SCANNED_CHARS - scannedChars);
    scannedChars += slice.length;
    tallyScripts(slice, counts);
  }

  const cjk = counts.han + counts.kana + counts.hangul;
  const letters = cjk + counts.latin;
  if (letters < MIN_SCANNED_LETTERS || cjk / letters < CJK_DOMINANCE_THRESHOLD) {
    return undefined;
  }
  // Japanese prose inevitably mixes kana; Chinese does not. Hangul at or above
  // half the CJK share is treated as Korean.
  if (counts.kana > 0 && counts.kana * 20 >= cjk) return "Japanese";
  if (counts.hangul * 2 >= cjk) return "Korean";
  return "Chinese";
}
