/**
 * config.js — Load and merge mycode.config.json with defaults.
 *
 * The config file is looked up from process.cwd() (wherever the user runs
 * `mycode`), so each project can have its own ignore list, model choice, etc.
 * Missing keys fall back to the defaults below — you never need a config file
 * unless you want to override something.
 */

import fs from "fs";
import path from "path";

const DEFAULTS = {
  model: "claude-sonnet-4-6",

  // Folders/patterns to skip when reading the project tree.
  // Uses glob-style matching relative to the project root.
  ignore: [
    "node_modules/**",
    ".git/**",
    "dist/**",
    "build/**",
    ".next/**",
    "coverage/**",
    "*.min.js",
    "*.map",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
  ],

  // Files larger than this (bytes) won't be read into context automatically.
  // The user can still explicitly ask about them.
  maxFileSizeBytes: 100_000,   // 100 KB

  // Maximum total characters sent to the model as file context.
  // Prevents accidentally blowing the context window on giant repos.
  maxContextChars: 80_000,

  // Commands that are always considered destructive and require confirmation
  // even if the user has enabled auto-confirm.
  destructivePatterns: [
    /rm\s/,
    /rmdir/,
    /git\s+reset/,
    /git\s+clean/,
    /git\s+push\s+.*--force/,
    /DROP\s+TABLE/i,
    /format\s+[a-z]:/i,
  ],
};

let _cached = null;

export function loadConfig() {
  if (_cached) return _cached;

  const configPath = path.join(process.cwd(), "mycode.config.json");
  let userConfig = {};

  if (fs.existsSync(configPath)) {
    try {
      userConfig = JSON.parse(fs.readFileSync(configPath, "utf8"));
    } catch (e) {
      console.warn(`Warning: could not parse mycode.config.json — ${e.message}`);
    }
  }

  // Deep-merge: user ignore list REPLACES defaults (don't concat — user knows what they want)
  _cached = { ...DEFAULTS, ...userConfig };
  return _cached;
}

/** Reset cache — useful for tests or when the user edits the config mid-session. */
export function resetConfig() {
  _cached = null;
}
