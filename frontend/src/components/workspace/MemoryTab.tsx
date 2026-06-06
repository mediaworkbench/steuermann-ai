"use client";

import { useI18n } from "@/hooks/useI18n";
import { WorkspaceTabState } from "./WorkspaceTabState";
import type { AnswerEvidence } from "@/lib/answerEvidence";

/** Read-only evidence tab: memories recalled for the latest answer. */
export function MemoryTab({ evidence }: { evidence: AnswerEvidence }) {
  const { t } = useI18n();

  if (evidence.memories.length === 0) {
    return (
      <WorkspaceTabState
        icon="memory"
        title={t("workspace.tabMemory")}
        hint={t("workspace.memoryEmpty")}
      />
    );
  }

  return (
    <div className="p-3 space-y-2">
      {evidence.memories.map((mem) => (
        <div key={mem.memory_id} className="rounded-lg border border-gray-200 bg-gray-50/60 p-2.5">
          <p className="text-xs text-evergreen/80">{mem.text || mem.memory_id}</p>
          <p className="text-[11px] text-evergreen/40 mt-1">
            {mem.is_related ? t("memories.related") : t("memories.primary")}
            {typeof mem.importance_score === "number" && (
              <> · {t("workspace.metaScore", { score: mem.importance_score.toFixed(2) })}</>
            )}
            {typeof mem.user_rating === "number" && (
              <> · {t("workspace.metaRated", { rating: mem.user_rating })}</>
            )}
          </p>
        </div>
      ))}
    </div>
  );
}
