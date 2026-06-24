/**
 * prompts.js — Shared interactive confirmation helpers using inquirer.
 */

import { input, confirm as inquirerConfirm, select } from "@inquirer/prompts";
import chalk from "chalk";

/**
 * Ask a yes/no question.
 * @param {string} message
 * @param {boolean} requireExplicitYes  - if true, user must type "yes" not just press Enter
 * @returns {Promise<boolean>}
 */
export async function confirm(message, requireExplicitYes = false) {
  if (requireExplicitYes) {
    const answer = await input({
      message: `${message} (type "yes" to confirm):`,
    });
    return answer.trim().toLowerCase() === "yes";
  }

  return inquirerConfirm({ message, default: false });
}

/**
 * Prompt the user to choose from a list of options.
 */
export async function choose(message, choices) {
  return select({ message, choices });
}

/**
 * Prompt for free text input.
 */
export async function askText(message, defaultValue = "") {
  return input({ message, default: defaultValue });
}

/**
 * Print a styled section header to the terminal.
 */
export function header(text) {
  console.log("\n" + chalk.bold.cyan(`━━━ ${text} ━━━`));
}

/**
 * Print an info line.
 */
export function info(text) {
  console.log(chalk.dim(`ℹ  ${text}`));
}

/**
 * Print a success line.
 */
export function success(text) {
  console.log(chalk.green(`✓  ${text}`));
}

/**
 * Print an error line.
 */
export function error(text) {
  console.error(chalk.red(`✗  ${text}`));
}
