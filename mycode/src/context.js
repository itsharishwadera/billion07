/**
 * context.js — Project file reader and context builder.
 *
 * The model can't help with YOUR code unless it can see YOUR code.
 * This module:
 *   1. Walks the project tree (respecting the ignore list).
 *   2. Builds a compact directory tree string for orientation.
 *   3. Reads specific files and injects their content into the prompt.
 *   4. Respects maxFileSizeBytes and maxContextChars limits so we don't
 *      accidentally send 500KB of source to the API.
 */

import fs from "fs";
import path from "path";
import { glob } from "glob";
import { loadConfig } from "./config.js";

const ROOT = process.cwd();

// ---------------------------------------------------------------------------
// Directory tree
// ---------------------------------------------------------------------------

/**
 * Returns a compact tree string like:
 *   src/
 *     routes/
 *       leads.js
 *     models/
 *       Lead.js
 *   package.json
 */
export async function buildFileTree(root = ROOT) {
  const cfg = loadConfig();

  // Get all files not matching the ignore list
  const files = await glob("**/*", {
    cwd: root,
    ignore: cfg.ignore,
    nodir: false,
    dot: false,
  });

  // Sort so directories come before their contents
  files.sort();

  // Build tree string
  const lines = [];
  const seenDirs = new Set();

  for (const f of files) {
    const parts = f.split("/");
    // Indent each path segment
    for (let depth = 0; depth < parts.length - 1; depth++) {
      const dirPath = parts.slice(0, depth + 1).join("/") + "/";
      if (!seenDirs.has(dirPath)) {
        seenDirs.add(dirPath);
        lines.push("  ".repeat(depth) + parts[depth] + "/");
      }
    }
    const indent = "  ".repeat(parts.length - 1);
    lines.push(indent + parts[parts.length - 1]);
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// File reading
// ---------------------------------------------------------------------------

/**
 * Read a single file and return its content as a string.
 * Returns null (with a warning) if the file is too large or doesn't exist.
 */
export function readFile(filePath) {
  const cfg = loadConfig();
  const abs = path.isAbsolute(filePath) ? filePath : path.join(ROOT, filePath);

  if (!fs.existsSync(abs)) {
    return { content: null, error: `File not found: ${filePath}` };
  }

  const stat = fs.statSync(abs);
  if (stat.size > cfg.maxFileSizeBytes) {
    return {
      content: null,
      error: `File too large (${(stat.size / 1024).toFixed(1)} KB > ${cfg.maxFileSizeBytes / 1024} KB limit): ${filePath}`,
    };
  }

  try {
    const content = fs.readFileSync(abs, "utf8");
    return { content, error: null };
  } catch (e) {
    return { content: null, error: `Could not read ${filePath}: ${e.message}` };
  }
}

/**
 * Read multiple files and return them formatted as a context block.
 *
 * Output looks like:
 *   ### File: src/routes/leads.js
 *   ```js
 *   // ... file content ...
 *   ```
 */
export function readFiles(filePaths) {
  const cfg = loadConfig();
  const blocks = [];
  let totalChars = 0;

  for (const fp of filePaths) {
    if (totalChars >= cfg.maxContextChars) {
      blocks.push(`\n[Context limit reached — skipped: ${fp}]`);
      continue;
    }

    const { content, error } = readFile(fp);
    if (error) {
      blocks.push(`\n### File: ${fp}\n[ERROR: ${error}]`);
      continue;
    }

    const ext = path.extname(fp).replace(".", "") || "text";
    const block = `\n### File: ${fp}\n\`\`\`${ext}\n${content}\n\`\`\``;
    blocks.push(block);
    totalChars += block.length;
  }

  return blocks.join("\n");
}

// ---------------------------------------------------------------------------
// Context builder for the model
// ---------------------------------------------------------------------------

/**
 * Build the full context string injected at the start of each conversation.
 *
 * Always includes: project root path, file tree.
 * Optionally includes: contents of files mentioned in the task.
 *
 * @param {string[]} filesToRead  - Relative paths to read into context
 * @returns {Promise<string>}
 */
export async function buildContext(filesToRead = []) {
  const tree = await buildFileTree();
  const lines = [
    `Project root: ${ROOT}`,
    "",
    "Directory structure:",
    "```",
    tree,
    "```",
  ];

  if (filesToRead.length > 0) {
    lines.push("\nFile contents:");
    lines.push(readFiles(filesToRead));
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Smart file detection: extract file paths from user's task string
// ---------------------------------------------------------------------------

/**
 * Pull out anything that looks like a file path from the user's task.
 * Examples: "fix routes/leads.js" → ["routes/leads.js"]
 *           "compare Header.jsx and Footer.jsx" → ["Header.jsx", "Footer.jsx"]
 */
export function extractFilePaths(text) {
  // Match path-like tokens: optional leading ./,  word chars + slashes + extension
  const pattern = /(?:\.\/)?[\w\-./]+\.(?:js|jsx|ts|tsx|json|css|html|md|env|yml|yaml|sh|py|sql)\b/g;
  const matches = text.match(pattern) ?? [];
  // Only include paths that actually exist in the project
  return matches.filter((p) => {
    const abs = path.isAbsolute(p) ? p : path.join(ROOT, p);
    return fs.existsSync(abs);
  });
}
