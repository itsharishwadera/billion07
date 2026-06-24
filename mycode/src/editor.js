/**
 * editor.js — Parse proposed file edits from the model, show diffs, apply with confirmation.
 *
 * The model outputs file edits in a custom fenced format:
 *
 *   <<<FILE: path/to/file.js>>>
 *   // new file content
 *   <<<END>>>
 *
 * This module:
 *   1. Parses all such blocks out of the model's response.
 *   2. Generates a unified diff against the existing file (or shows "new file").
 *   3. Shows the diff with colour.
 *   4. Asks for confirmation before writing.
 *   5. Writes only after confirmed.
 *
 * Why this format instead of standard unified diff output from the model?
 * Asking the model to output a full unified diff is fragile — it gets the
 * line numbers wrong constantly.  Asking it to output the FULL new file content
 * is more reliable and easier to parse.  We compute the diff ourselves.
 */

import fs from "fs";
import path from "path";
import { createTwoFilesPatch } from "diff";
import chalk from "chalk";
import { confirm } from "./prompts.js";

const ROOT = process.cwd();

// Regex to find <<<FILE: path>>> ... <<<END>>> blocks in the model response
const FILE_BLOCK_RE = /<<<FILE:\s*(.+?)>>>\n([\s\S]*?)<<<END>>>/g;

/**
 * Parse all proposed file edits from a model response string.
 * Returns an array of { filePath, newContent } objects.
 */
export function parseEdits(modelResponse) {
  const edits = [];
  let match;
  FILE_BLOCK_RE.lastIndex = 0;   // reset regex state
  while ((match = FILE_BLOCK_RE.exec(modelResponse)) !== null) {
    edits.push({
      filePath: match[1].trim(),
      newContent: match[2],      // preserve trailing newline if present
    });
  }
  return edits;
}

/**
 * Generate a coloured unified diff between the current file and proposed content.
 */
function colorDiff(filePath, newContent) {
  const abs = path.isAbsolute(filePath) ? filePath : path.join(ROOT, filePath);
  const oldContent = fs.existsSync(abs) ? fs.readFileSync(abs, "utf8") : "";
  const isNew = !fs.existsSync(abs);

  if (oldContent === newContent) {
    return chalk.dim(`  (no changes to ${filePath})`);
  }

  const patch = createTwoFilesPatch(
    isNew ? "/dev/null" : filePath,
    filePath,
    oldContent,
    newContent,
    isNew ? "" : "current",
    "proposed"
  );

  // Colour the diff lines
  const lines = patch.split("\n").map((line) => {
    if (line.startsWith("+++") || line.startsWith("---")) return chalk.bold(line);
    if (line.startsWith("@@")) return chalk.cyan(line);
    if (line.startsWith("+")) return chalk.green(line);
    if (line.startsWith("-")) return chalk.red(line);
    return chalk.dim(line);
  });

  return lines.join("\n");
}

/**
 * Show all proposed edits as diffs, ask for confirmation per-file, then write.
 *
 * @param {Array<{filePath: string, newContent: string}>} edits
 * @param {boolean} autoConfirm  - skip prompts (e.g. for --yes flag)
 * @returns {Promise<string[]>}  paths of files actually written
 */
export async function applyEdits(edits, autoConfirm = false) {
  if (edits.length === 0) return [];

  const written = [];

  for (const { filePath, newContent } of edits) {
    const abs = path.isAbsolute(filePath) ? filePath : path.join(ROOT, filePath);
    const isNew = !fs.existsSync(abs);

    console.log(
      "\n" +
        chalk.bold.yellow(`━━━ Proposed ${isNew ? "new file" : "edit"}: ${filePath} ━━━`)
    );
    console.log(colorDiff(filePath, newContent));

    const ok = autoConfirm || (await confirm(`Apply this change to ${filePath}?`));

    if (ok) {
      // Create parent directories if needed
      fs.mkdirSync(path.dirname(abs), { recursive: true });
      fs.writeFileSync(abs, newContent, "utf8");
      console.log(chalk.green(`  ✓ Written: ${filePath}`));
      written.push(filePath);
    } else {
      console.log(chalk.dim(`  ✗ Skipped: ${filePath}`));
    }
  }

  return written;
}
