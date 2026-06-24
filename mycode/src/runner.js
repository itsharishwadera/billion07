/**
 * runner.js — Shell command execution with confirmation and destructive-command detection.
 *
 * The model proposes commands using:
 *   <<<CMD: npm install express>>>
 *
 * This module:
 *   1. Parses <<<CMD: ...>>> blocks from the model response.
 *   2. Checks each command against the destructive patterns list in config.
 *   3. Always asks for confirmation; flags destructive commands with a red warning.
 *   4. Spawns the command as a child process with live stdout/stderr streaming
 *      so you see output in real time.
 *
 * Why spawn instead of exec?
 * `exec` buffers ALL output then returns it — bad for long commands like
 * `npm install` or `npm run build` where you want to see progress.
 * `spawn` streams stdout/stderr directly to the terminal.
 */

import { spawn } from "child_process";
import chalk from "chalk";
import { loadConfig } from "./config.js";
import { confirm } from "./prompts.js";

const CMD_BLOCK_RE = /<<<CMD:\s*(.+?)>>>/g;

/**
 * Parse all proposed shell commands from a model response.
 * Returns an array of command strings.
 */
export function parseCommands(modelResponse) {
  const cmds = [];
  let match;
  CMD_BLOCK_RE.lastIndex = 0;
  while ((match = CMD_BLOCK_RE.exec(modelResponse)) !== null) {
    cmds.push(match[1].trim());
  }
  return cmds;
}

/**
 * Check whether a command matches any of the configured destructive patterns.
 */
function isDestructive(cmd) {
  const { destructivePatterns } = loadConfig();
  return destructivePatterns.some((p) =>
    typeof p === "string" ? cmd.includes(p) : p.test(cmd)
  );
}

/**
 * Run a shell command, streaming output to the terminal.
 * Returns a promise that resolves with the exit code.
 */
function runCommand(cmd) {
  return new Promise((resolve) => {
    console.log(chalk.dim(`\n$ ${cmd}\n`));

    // Use cmd /c on Windows, sh -c on Unix
    const isWin = process.platform === "win32";
    const shell = isWin ? "cmd" : "sh";
    const shellFlag = isWin ? "/c" : "-c";

    const child = spawn(shell, [shellFlag, cmd], {
      cwd: process.cwd(),
      stdio: "inherit",   // stream directly to terminal
      env: process.env,
    });

    child.on("close", (code) => {
      if (code === 0) {
        console.log(chalk.green(`\n  ✓ Command exited with code 0`));
      } else {
        console.log(chalk.red(`\n  ✗ Command exited with code ${code}`));
      }
      resolve(code);
    });

    child.on("error", (err) => {
      console.error(chalk.red(`  Error spawning command: ${err.message}`));
      resolve(1);
    });
  });
}

/**
 * Show all proposed commands, ask for confirmation, run approved ones.
 *
 * @param {string[]} commands
 * @param {boolean} autoConfirm  - skip prompts (e.g. --yes flag)
 * @returns {Promise<void>}
 */
export async function runCommands(commands, autoConfirm = false) {
  for (const cmd of commands) {
    const destructive = isDestructive(cmd);

    if (destructive) {
      console.log(
        "\n" +
          chalk.bold.red("⚠  DESTRUCTIVE COMMAND DETECTED") +
          "\n" +
          chalk.red(`   ${cmd}`)
      );
    } else {
      console.log("\n" + chalk.bold.blue(`Shell command: `) + chalk.white(cmd));
    }

    // Destructive commands always prompt, even with --yes
    const ok =
      (!autoConfirm || destructive)
        ? await confirm(
            destructive
              ? chalk.red(`Run this destructive command? Type yes to confirm:`)
              : `Run this command?`,
            destructive   // requireExplicitYes for destructive
          )
        : true;

    if (ok) {
      await runCommand(cmd);
    } else {
      console.log(chalk.dim(`  Skipped: ${cmd}`));
    }
  }
}
