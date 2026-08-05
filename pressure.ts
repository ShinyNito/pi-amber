// Compaction pressure state machine (persisted in CompactionEntry.details).
// Consecutive ineffective compactions push the level up (more aggressive input
// trimming), but never hard-block; pressure decays after an idle window so a
// healthy session resets on its own.

export type PressureLevel = 0 | 1 | 2;

export type CompactionPressure = {
  level: PressureLevel;
  consecutiveIneffective: number;
  compactionsApplied: number;
  lastCompactionAt: number;
};

// A compaction is "ineffective" when it keeps more than this share of the
// pre-compaction threshold. 0.9 = still 90%+ of the way to the limit.
export const INEFFECTIVE_COMPACTION_RATIO = 0.9;
export const MAX_PRESSURE_LEVEL: PressureLevel = 2;
// After this idle window without compactions, pressure resets to zero.
export const PRESSURE_DECAY_WINDOW_MS = 5 * 60_000;

export function createCompactionPressure(): CompactionPressure {
  return { level: 0, consecutiveIneffective: 0, compactionsApplied: 0, lastCompactionAt: 0 };
}

/** Decay pressure if the last compaction was too long ago. */
export function normalizePressure(
  pressure: CompactionPressure,
  now: number,
): CompactionPressure {
  if (
    pressure.lastCompactionAt > 0 &&
    now - pressure.lastCompactionAt > PRESSURE_DECAY_WINDOW_MS &&
    (pressure.level > 0 || pressure.consecutiveIneffective > 0)
  ) {
    return { ...pressure, level: 0, consecutiveIneffective: 0 };
  }
  return pressure;
}

/** Record the outcome of a compaction and advance the ladder. */
export function notePressureAfterCompaction(
  pressure: CompactionPressure,
  params: { effective: boolean; now: number },
): CompactionPressure {
  const consecutiveIneffective = params.effective ? 0 : pressure.consecutiveIneffective + 1;
  return {
    level: Math.min(MAX_PRESSURE_LEVEL, consecutiveIneffective) as PressureLevel,
    consecutiveIneffective,
    compactionsApplied: pressure.compactionsApplied + 1,
    lastCompactionAt: params.now,
  };
}
