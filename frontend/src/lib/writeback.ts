/**
 * Helpers for the workspace-document writeback stream.
 *
 * In writeback mode the model replies as two sections:
 *
 *   SUMMARY:
 *   <one or two sentences>
 *
 *   DOCUMENT:
 *   <full revised file content>
 *
 * The chat should show only the SUMMARY while the (potentially long) DOCUMENT
 * body streams, then commit the clean confirmation the backend persisted. This
 * mirrors the backend's `_extract_writeback_summary` regex (tolerant of one or
 * more newlines before `DOCUMENT:`).
 */

export interface WritebackSplit {
  /** Text of the SUMMARY section so far (trimmed). */
  summary: string;
  /** True once the `DOCUMENT:` marker has appeared in the stream. */
  inDocument: boolean;
}

const SUMMARY_START = /SUMMARY:\s*\n?/;
const DOCUMENT_MARKER = /\n+DOCUMENT:\s*\n?/;

/** True when `content` looks like the start of a structured writeback response. */
export function looksLikeWriteback(content: string): boolean {
  return /^\s*SUMMARY:/.test(content ?? "");
}

/**
 * Split a (possibly partial) writeback stream into its summary and a flag for
 * whether the document body has started. Returns an empty summary when the
 * `SUMMARY:` header has not appeared yet.
 */
export function splitWritebackStream(content: string): WritebackSplit {
  const text = content ?? "";
  const start = SUMMARY_START.exec(text);
  if (!start) return { summary: "", inDocument: false };

  const afterSummary = text.slice(start.index + start[0].length);
  const doc = DOCUMENT_MARKER.exec(afterSummary);
  if (doc) {
    return { summary: afterSummary.slice(0, doc.index).trim(), inDocument: true };
  }
  return { summary: afterSummary.trim(), inDocument: false };
}
