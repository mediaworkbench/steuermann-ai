import type { Message, MessageMetrics, NodeTraceEntry } from "./types";

export interface FocusedAnswer {
  /** True when the user has pinned a *past* answer (panel shows that answer). */
  isHistorical: boolean;
  /** The pinned answer's metrics (only meaningful when isHistorical). */
  metrics: MessageMetrics | null;
  /** The pinned answer's node trace (only meaningful when isHistorical). */
  nodeTrace: NodeTraceEntry[];
}

const FOLLOW_LATEST: FocusedAnswer = { isHistorical: false, metrics: null, nodeTrace: [] };

/**
 * Resolve which answer the workspace panel should show.
 *
 * Returns `isHistorical: false` for the default "follow latest" behavior —
 * when no answer is pinned (`focusedAnswerIndex === null`), when the pin equals
 * the latest answer, or when the index is stale/invalid (out of bounds, or no
 * longer an assistant message after a truncating edit-and-resend). The caller
 * then blends in the live latest metrics + (streaming) trace.
 *
 * When `isHistorical` is true, the panel reads the pinned answer's committed
 * `metrics` / `nodeTrace` directly (no live state).
 */
export function pickFocusedAnswer(
  messages: Message[],
  focusedAnswerIndex: number | null,
  lastAssistantIndex: number,
): FocusedAnswer {
  if (focusedAnswerIndex === null || focusedAnswerIndex === lastAssistantIndex) {
    return FOLLOW_LATEST;
  }
  const msg = messages[focusedAnswerIndex];
  if (!msg || msg.role !== "assistant") {
    return FOLLOW_LATEST;
  }
  return {
    isHistorical: true,
    metrics: msg.metrics ?? null,
    nodeTrace: msg.nodeTrace ?? [],
  };
}
