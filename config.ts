import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export type AmberConfig = {
  /** Provider/model used for summarization. Empty → current session model. */
  model?: string;
  /** Maximum output tokens for the summary request. */
  maxTokens: number;
  /** Skip auto-compaction and let pi use its default. */
  enabled: boolean;
};

export const DEFAULT_AMBER_CONFIG: AmberConfig = {
  model: "",
  maxTokens: 8192,
  enabled: true,
};

export function getConfigFile(): string {
  return join(homedir(), ".pi", "pi-amber.json");
}

function readJson<T>(path: string, fallback: T): T {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch {
    return fallback;
  }
}

export function loadConfig(): AmberConfig {
  const stored = readJson<Partial<AmberConfig>>(getConfigFile(), {});
  return {
    ...DEFAULT_AMBER_CONFIG,
    ...stored,
  };
}

export function saveConfig(config: AmberConfig): void {
  const file = getConfigFile();
  try {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify(config, null, 2) + "\n");
  } catch (error) {
    console.error(`pi-amber: failed to save config: ${error instanceof Error ? error.message : String(error)}`);
  }
}
