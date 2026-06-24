#!/usr/bin/env node
/**
 * bin/mycode.js — CLI entry point.
 *
 * The shebang line (#!/usr/bin/env node) tells the OS to run this with Node
 * when you invoke it as a standalone command after `npm link`.
 *
 * Commander parses argv and routes to the right action:
 *
 *   mycode "fix the bug in routes/leads.js"      ← one-shot task
 *   mycode chat                                    ← interactive REPL
 *   mycode plan "refactor the auth middleware"     ← plan mode (explain first)
 *   mycode read src/app.js                         ← just read + summarise a file
 *   mycode config                                  ← print current config
 */

// No API key needed — this tool talks to Ollama running locally.
import { program } from "commander";
import chalk from "chalk";
import readline from "readline";

import { sendMessages } from "../src/client.js";
import { buildContext, extractFilePaths } from "../src/context.js";
import { parseEdits, applyEdits } from "../src/editor.js";
import { parseCommands, runCommands } from "../src/runner.js";
import { Session } from "../src/session.js";
import { loadConfig } from "../src/config.js";
import { header, info, error } from "../src/prompts.js";

const VERSION = "0.1.0";

// ---------------------------------------------------------------------------
// Core task runner — shared by both one-shot and chat modes
// ---------------------------------------------------------------------------

async function runTask(userTask, session, opts = {}) {
  const { planMode = false, autoConfirm = false, files = [] } = opts;

  // Detect any file paths mentioned in the task
  const mentionedFiles = [...new Set([...files, ...extractFilePaths(userTask)])];

  // Build project context (tree + file contents)
  info(`Reading project context${mentionedFiles.length ? ` (${mentionedFiles.length} file(s))` : ""}…`);
  const context = await buildContext(mentionedFiles);

  // Compose the user message
  let userMessage = `${context}\n\n---\nTask: ${userTask}`;
  if (planMode) {
    userMessage += "\n\nBefore doing anything, list a numbered plan of what you intend to do. Wait for my approval before executing.";
  }

  session.addUserMessage(userMessage);

  // Call the model (streams to stdout)
  header("mycode");
  const reply = await sendMessages(session.getMessages(), {
    systemPrompt: undefined, // use the default from client.js
  });
  session.addAssistantMessage(reply);

  // Parse and apply any proposed edits
  const edits = parseEdits(reply);
  if (edits.length > 0) {
    info(`Found ${edits.length} proposed file edit(s).`);
    await applyEdits(edits, autoConfirm);
  }

  // Parse and run any proposed shell commands
  const cmds = parseCommands(reply);
  if (cmds.length > 0) {
    info(`Found ${cmds.length} proposed command(s).`);
    await runCommands(cmds, autoConfirm);
  }

  console.log(chalk.dim(`\n[Session: ${session.summary}]`));
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

program
  .name("mycode")
  .description("Personal CLI coding assistant powered by Claude")
  .version(VERSION);

// ---- Default: one-shot task  -----------------------------------------------
program
  .argument("[task]", "Describe what you want to do")
  .option("-f, --file <paths...>", "Explicitly include these files in context")
  .option("-y, --yes", "Auto-confirm all non-destructive changes")
  .option("--plan", "Plan mode: explain steps before executing")
  .action(async (task, opts) => {
    if (!task) {
      program.help();
      return;
    }
    const session = new Session();
    try {
      await runTask(task, session, {
        planMode: opts.plan ?? false,
        autoConfirm: opts.yes ?? false,
        files: opts.file ?? [],
      });
    } catch (err) {
      error(err.message);
      process.exit(1);
    }
  });

// ---- Interactive chat mode  ------------------------------------------------
program
  .command("chat")
  .description("Start an interactive multi-turn session")
  .option("-y, --yes", "Auto-confirm all non-destructive changes")
  .action(async (opts) => {
    const session = new Session();
    const cfg = loadConfig();

    console.log(chalk.bold.cyan("\nmycode interactive mode"));
    console.log(chalk.dim(`Model: ${cfg.model}  |  Project: ${process.cwd()}`));
    console.log(chalk.dim(`Type your task, or:\n  /clear   — reset conversation\n  /exit    — quit\n`));

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: true,
    });

    const prompt = () => {
      rl.question(chalk.bold.cyan("you> "), async (line) => {
        const task = line.trim();

        if (!task) {
          prompt();
          return;
        }

        if (task === "/exit" || task === "/quit") {
          console.log(chalk.dim("Goodbye."));
          rl.close();
          return;
        }

        if (task === "/clear") {
          session.clear();
          console.log(chalk.dim("Conversation cleared."));
          prompt();
          return;
        }

        if (task === "/history") {
          console.log(chalk.dim(JSON.stringify(session.getMessages(), null, 2)));
          prompt();
          return;
        }

        try {
          await runTask(task, session, {
            planMode: task.startsWith("/plan "),
            autoConfirm: opts.yes ?? false,
          });
        } catch (err) {
          error(err.message);
        }

        prompt();
      });
    };

    prompt();
  });

// ---- Plan mode shortcut  ---------------------------------------------------
program
  .command("plan <task>")
  .description("Explain what would be done before executing anything")
  .option("-f, --file <paths...>", "Explicitly include these files in context")
  .action(async (task, opts) => {
    const session = new Session();
    try {
      await runTask(task, session, {
        planMode: true,
        autoConfirm: false,
        files: opts.file ?? [],
      });
    } catch (err) {
      error(err.message);
      process.exit(1);
    }
  });

// ---- Read a file and ask the model to summarise it  -----------------------
program
  .command("read <file>")
  .description("Read a file and ask Claude to explain or summarise it")
  .option("--ask <question>", "Custom question about the file")
  .action(async (file, opts) => {
    const session = new Session();
    const question = opts.ask ?? "Explain what this file does and highlight anything worth knowing.";
    try {
      await runTask(question, session, { files: [file] });
    } catch (err) {
      error(err.message);
      process.exit(1);
    }
  });

// ---- Show resolved config  -------------------------------------------------
program
  .command("config")
  .description("Print the resolved configuration for this project")
  .action(() => {
    const cfg = loadConfig();
    console.log(JSON.stringify(cfg, null, 2));
  });

program.parse();
