/**
 * session.js — Conversation history manager.
 *
 * Claude's API is stateless — every request must include the full conversation
 * history. This module manages that history for the duration of one mycode session.
 *
 * A "session" is scoped to a single run of `mycode`. When you Ctrl-C and restart,
 * history is cleared (fresh context). If you want persistent sessions across
 * restarts, that's a future feature (save/load from ~/.mycode/sessions/).
 *
 * Message format matches the Anthropic API exactly:
 *   { role: "user" | "assistant", content: string }
 */

export class Session {
  constructor() {
    /** @type {Array<{role: string, content: string}>} */
    this.messages = [];
    this.turnCount = 0;
  }

  /**
   * Add a user message (task + context) to history.
   */
  addUserMessage(content) {
    this.messages.push({ role: "user", content });
    this.turnCount++;
  }

  /**
   * Add the assistant's reply to history so the next turn has context.
   */
  addAssistantMessage(content) {
    this.messages.push({ role: "assistant", content });
  }

  /**
   * Return a copy of history for sending to the API.
   * We return a copy so callers can't accidentally mutate the session.
   */
  getMessages() {
    return [...this.messages];
  }

  /**
   * Clear history — useful if the user wants to start a fresh task
   * without restarting the process (future: `/clear` command).
   */
  clear() {
    this.messages = [];
    this.turnCount = 0;
  }

  /**
   * Return a compact summary for the status line.
   */
  get summary() {
    const userTurns = this.messages.filter((m) => m.role === "user").length;
    return `${userTurns} turn${userTurns !== 1 ? "s" : ""}`;
  }
}
