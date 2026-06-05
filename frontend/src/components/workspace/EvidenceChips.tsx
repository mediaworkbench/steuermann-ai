"use client";

import { Icon } from "../Icon";
import { useI18n } from "@/hooks/useI18n";
import { useAnswerEvidence } from "@/hooks/useAnswerEvidence";
import type { MessageMetrics } from "@/lib/types";

/**
 * Compact, glanceable evidence summary for a single answer (sources · memory ·
 * tools · docs · map). Rendered at the chat-stream level for the latest answer
 * only; the full drill-down lives in the workspace evidence tabs. Returns null
 * when the answer produced no evidence.
 */
export function EvidenceChips({ metrics }: { metrics?: MessageMetrics }) {
  const { t } = useI18n();
  const evidence = useAnswerEvidence(metrics);
  if (!evidence.hasEvidence) return null;

  const chips = [
    { key: "sources", icon: "menu_book", count: evidence.sourceCount, label: t("workspace.evidenceSources") },
    { key: "memory", icon: "memory", count: evidence.memoryCount, label: t("workspace.evidenceMemory") },
    { key: "tools", icon: "build", count: evidence.toolCount, label: t("workspace.evidenceTools") },
    { key: "docs", icon: "folder_open", count: evidence.documentCount, label: t("workspace.evidenceDocs") },
  ].filter((c) => c.count > 0);

  if (evidence.mapData) {
    chips.push({ key: "map", icon: "map", count: 1, label: t("workspace.mapGenerated") });
  }
  if (chips.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-1 mt-2" aria-label={t("workspace.evidenceSummary")}>
      {chips.map((c) => (
        <span
          key={c.key}
          title={c.label}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium
                     bg-gray-50 text-evergreen/55 border border-gray-200"
        >
          <Icon name={c.icon} size={12} className="text-evergreen/40" />
          {c.key === "map" ? c.label : c.count}
        </span>
      ))}
    </div>
  );
}
