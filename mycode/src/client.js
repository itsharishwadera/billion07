/**
 * client.js — Local LLM client via Ollama.
 *
 * Ollama runs a local server at http://localhost:11434 and exposes an
 * OpenAI-compatible REST API. No API key needed — completely free and offline.
 *
 * To swap models: change "model" in mycode.config.json.
 * Popular options for 8GB VRAM:
 *   llama3.1      — best general coding + reasoning (recommended)
 *   mistral       — fast, good at code
 *   codellama     — fine-tuned specifically for code
 *   deepseek-coder — strong code model
 *
 * To pull a model:  ollama pull llama3.1
 * To list models:   ollama list
 */

import { loadConfig } from "./config.js";

const OLLAMA_BASE = "http://localhost:11434";

/**
 * Send a conversation to Ollama and stream the response to stdout.
 * Returns the full assistant reply as a string.
 *
 * @param {Array<{role: "user"|"assistant", content: string}>} messages
 * @param {object} opts
 * @returns {Promise<string>}
 */
export async function sendMessages(messages, opts = {}) {
  const cfg = loadConfig();
  const model = opts.model ?? cfg.model ?? "llama3.1";
  const system = opts.systemPrompt ?? buildSystemPrompt();

  // Prepend system message in the messages array
  const fullMessages = [
    { role: "system", content: system },
    ...messages,
  ];

  let response;
  try {
    response = await fetch(`${OLLAMA_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: fullMessages,
        stream: true,
      }),
    });
  } catch (e) {
    throw new Error(
      `Cannot connect to Ollama at ${OLLAMA_BASE}.\n` +
      `Make sure Ollama is running — start it with:  ollama serve\n` +
      `Original error: ${e.message}`
    );
  }

  if (!response.ok) {
    const text = await response.text();
    // Friendly message if the model isn't downloaded yet
    if (response.status === 404 || text.includes("model") ) {
      throw new Error(
        `Model "${model}" not found in Ollama.\n` +
        `Download it with:  ollama pull ${model}\n` +
        `Or list available models with:  ollama list`
      );
    }
    throw new Error(`Ollama API error ${response.status}: ${text}`);
  }

  // Ollama streams newline-delimited JSON objects
  let fullText = "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  process.stdout.write("\x1b[2m");   // dim while model is generating

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    // Each chunk may contain multiple newline-separated JSON objects
    for (const line of chunk.split("\n")) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        const token = obj?.message?.content ?? "";
        if (token) {
          process.stdout.write(token);
          fullText += token;
        }
      } catch {
        // partial JSON line — skip
      }
    }
  }

  process.stdout.write("\x1b[0m\n");  // reset dim
  return fullText;
}

/**
 * One-shot question, no history.
 */
export async function ask(prompt, opts = {}) {
  return sendMessages([{ role: "user", content: prompt }], opts);
}

function buildSystemPrompt() {
  return `You are mycode, a personal CLI coding assistant running locally on the user's machine.

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
