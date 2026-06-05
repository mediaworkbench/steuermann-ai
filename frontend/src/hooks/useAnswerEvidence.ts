import { useMemo } from "react";
import type { MessageMetrics } from "@/lib/types";
import { deriveAnswerEvidence, type AnswerEvidence } from "@/lib/answerEvidence";

/**
 * Memoized accessor for normalized answer evidence. The single source of truth
 * for the in-stream evidence chips, the workspace evidence tabs, and the
 * per-message MetricsPanel.
 */
export function useAnswerEvidence(metrics: MessageMetrics | null | undefined): AnswerEvidence {
  return useMemo(() => deriveAnswerEvidence(metrics), [metrics]);
}

export type { AnswerEvidence };
