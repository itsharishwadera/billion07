/**
 * client.js — Anthropic API client wrapper.
 *
 * Why a wrapper instead of using the SDK directly everywhere?
 * Centralising the API call here means:
 *   1. Swapping to a different provider (OpenAI, Gemini, local Ollama) means
 *      editing ONE file, not hunting through the codebase.
 *   2. Retry logic, token counting, and error formatting live in one place.
 *   3. The rest of the code only knows about `sendMessages(messages)` → string.
 */

import Anthropic from "@anthropic-ai/sdk";
import { loadConfig } from "./config.js";

let _client = null;

function getClient() {
  if (_client) return _client;

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error(
      "ANTHROPIC_API_KEY is not set.\n" +
        "Add it to a .env file in your project root, or export it in your shell:\n" +
        "  export ANTHROPIC_API_KEY=sk-ant-..."
    );
  }

  _client = new Anthropic({ apiKey });
  return _client;
}

/**
 * Send a conversation to the Claude API and return the assistant's text reply.
 *
 * @param {Array<{role: "user"|"assistant", content: string}>} messages
 * @param {object} opts  - overrides: model, maxTokens, systemPrompt
 * @returns {Promise<string>}  assistant message text
 */
export async function sendMessages(messages, opts = {}) {
  const cfg = loadConfig();
  const client = getClient();

  const model = opts.model ?? cfg.model ?? "claude-sonnet-4-6";
  const maxTokens = opts.maxTokens ?? 8096;
  const system = opts.systemPrompt ?? buildSystemPrompt();

  // Streaming gives faster perceived response for long outputs.
  // We accumulate the stream into a single string before returning.
  let fullText = "";
  process.stdout.write("\x1b[2m");                 // dim — model is "thinking"

  const stream = client.messages.stream({
    model,
    max_tokens: maxTokens,
    system,
    messages,
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta?.type === "text_delta"
    ) {
      process.stdout.write(event.delta.text);
      fullText += event.delta.text;
    }
  }

  process.stdout.write("\x1b[0m\n");               // reset dim
  return fullText;
}

/**
 * Lighter call for structured tasks (planning, one-shot answers).
 * Does NOT stream — just returns the complete response.
 */
export async function ask(prompt, opts = {}) {
  return sendMessages([{ role: "user", content: prompt }], opts);
}

function buildSystemPrompt() {
  return `You are mycode, a personal CLI coding assistant running on the user's machine.

You help with reading, writing, fixing, and refactoring code across real project files.

Rules:
- Be concise. The user is in a terminal — wall-of-text responses are hard to read.
- When proposing file edits, output them in this EXACT format so the tool can parse and apply them:

<<<FILE: path/to/file.ext>>>
<full new file content here>
<<<END>>>

- When proposing shell commands, wrap them like this:

<<<CMD: npm install express>>>

- Before running destructive operations (delete, overwrite, git reset), always explain WHY.
- If you are unsure about something, say so — don't guess at file paths or API shapes.
- In plan mode, list numbered steps before doing anything.`;
}
