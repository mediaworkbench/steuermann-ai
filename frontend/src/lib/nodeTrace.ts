/**
 * Graph nodes that run AFTER the answer is streamed ([DONE]) — the post-response
 * phase (rolling digest, summarization, memory write, cache stats). Their timing/
 * status is captured by the backend during the background drain and written back to
 * the assistant message, so the Inspector can show them once persisted.
 *
 * Shared single source of truth: the Inspector renders them, and the auto-refresh
 * poll uses it to detect when the post-response trace has landed.
 */
export const POST_RESPONSE_NODES = [
  "compress_conversation",
  "summarize",
  "update_memory",
  "cache_stats",
] as const;

const POST_RESPONSE_SET = new Set<string>(POST_RESPONSE_NODES);

/** True when a node id belongs to the post-response phase. */
export function isPostResponseNode(node: string): boolean {
  return POST_RESPONSE_SET.has(node);
}
