/**
 * Configuration loader for the RuView MCP server.
 *
 * All settings can be overridden via environment variables.  No config file is
 * required — the server is designed to work out of the box with a locally-running
 * sensing-server on the default port.
 */

import os from "node:os";
import path from "node:path";
import { existsSync } from "node:fs";
import type { RuviewConfig } from "./types.js";

function env(key: string): string | undefined {
  return process.env[key];
}

function envOrDefault(key: string, fallback: string): string {
  return env(key) ?? fallback;
}

/**
 * Load the effective RuviewConfig from environment variables.
 *
 * Environment variables:
 *   RUVIEW_SENSING_SERVER_URL   — base URL of the sensing-server  (default: http://localhost:3000)
 *   RUVIEW_API_TOKEN            — Bearer token for /api/v1/* routes (no default; auth disabled when absent)
 *   RUVIEW_POSE_COG_BINARY      — path to cog-pose-estimation binary
 *   RUVIEW_COUNT_COG_BINARY     — path to cog-person-count binary
 *   RUVIEW_JOBS_DIR             — directory for job logs (default: ~/.ruview/jobs)
 */
export function loadConfig(): RuviewConfig {
  return {
    sensingServerUrl: envOrDefault(
      "RUVIEW_SENSING_SERVER_URL",
      "http://localhost:3000"
    ),
    apiToken: env("RUVIEW_API_TOKEN"),
    poseCogBinary: envOrDefault(
      "RUVIEW_POSE_COG_BINARY",
      detectCogBinary("cog-pose-estimation")
    ),
    countCogBinary: envOrDefault(
      "RUVIEW_COUNT_COG_BINARY",
      detectCogBinary("cog-person-count")
    ),
    jobsDir: envOrDefault(
      "RUVIEW_JOBS_DIR",
      path.join(os.homedir(), ".ruview", "jobs")
    ),
  };
}

/**
 * Locate a cog binary in the common appliance install locations, probing each
 * candidate (ADR-264 F8/O7 — the pre-review version built this list and then
 * unconditionally returned the bare name). Falls back to the bare name (PATH
 * resolution at spawn time) when no candidate exists.
 */
function detectCogBinary(name: string): string {
  const id = name.replace("cog-", "");
  // Common install paths for Cognitum cog binaries on Linux/macOS appliances.
  const candidates = [
    `/var/lib/cognitum/apps/${id}/cog-${id}-arm`,
    `/var/lib/cognitum/apps/${id}/cog-${id}-x86_64`,
    `/usr/local/bin/${name}`,
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return name; // bare name — rely on PATH; spawn fails gracefully if absent
}
