const CHARS_PER_TOKEN = 4;
// CJK text is far denser than Latin: mainstream tokenizers (o200k/cl100k/Claude)
// need roughly 1 token per 1.4-1.7 han characters. A naive chars/4 estimate
// undercounts by ~2.5-3x. 0.7 tokens per CJK char is deliberately conservative.
const CJK_TOKENS_PER_CHAR = 0.7;

// CJK unified ideographs (incl. ext A), kana, hangul, compatibility ideographs
// and full-width punctuation. Single precompiled regex: the hot path counts CJK
// chars via one native replace pass instead of a per-character JS loop.
const CJK_RE = /[\u2e80-\u9fff\uac00-\ud7af\u1100-\u11ff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]/g;

/**
 * Fractional token estimate for arbitrary text (no trim, no rounding).
 * CJK chars count at CJK_TOKENS_PER_CHAR, everything else at 1/CHARS_PER_TOKEN.
 *
 * Note: for whole messages prefer pi's exported `estimateTokens(message)`;
 * this exists only because pi exposes no plain-text estimator.
 */
export function estimateTextTokenUnits(text: string): number {
  if (text.length === 0) return 0;
  const nonCjk = text.replace(CJK_RE, "");
  const cjkChars = text.length - nonCjk.length;
  return nonCjk.length / CHARS_PER_TOKEN + cjkChars * CJK_TOKENS_PER_CHAR;
}

export function estimateTextTokens(text: string): number {
  const normalized = text.trim();
  if (!normalized) return 0;
  return Math.ceil(estimateTextTokenUnits(normalized));
}
