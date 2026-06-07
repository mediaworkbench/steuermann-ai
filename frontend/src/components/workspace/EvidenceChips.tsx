"use client";

import { iconMap } from "@/lib/iconMap";
import { useI18n } from "@/hooks/useI18n";
import { useAnswerEvidence } from "@/hooks/useAnswerEvidence";
import { Button } from "@/components/ui/Button";
import type { MessageMetrics } from "@/lib/types";
import type { WorkspaceTabId } from "./types";

interface EvidenceChipsProps {
  metrics?: MessageMetrics;
  /** When provided, chips become buttons that open the matching workspace tab. */
  onSelect?: (tab: WorkspaceTabId) => void;
}

/**
 * Compact, glanceable evidence summary for a single answer (sources · memory ·
 * tools · docs · map). Rendered at the chat-stream level for the latest answer
 * only; the full drill-down lives in the workspace evidence tabs. Returns null
 * when the answer produced no evidence.
 */
export function EvidenceChips({ metrics, onSelect }: EvidenceChipsProps) {
  const { t } = useI18n();
  const evidence = useAnswerEvidence(metrics);
  if (!evidence.hasEvidence) return null;

  const allChips: { key: string; icon: string; count: number; label: string; tab: WorkspaceTabId }[] = [
    { key: "sources", icon: "menu_book", count: evidence.sourceCount, label: t("workspace.evidenceSources"), tab: "knowledge" },
    { key: "memory", icon: "memory", count: evidence.memoryCount, label: t("workspace.evidenceMemory"), tab: "memory" },
    { key: "tools", icon: "build", count: evidence.toolCount, label: t("workspace.evidenceTools"), tab: "outputs" },
    { key: "docs", icon: "folder_open", count: evidence.documentCount, label: t("workspace.evidenceDocs"), tab: "documents" },
  ];
  const chips = allChips.filter((c) => c.count > 0);

  if (evidence.mapData) {
    chips.push({ key: "map", icon: "map", count: 1, label: t("workspace.mapGenerated"), tab: "outputs" });
  }
  if (chips.length === 0) return null;

  const baseClass =
    "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-surface-muted text-muted-foreground border border-border";

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-1 mt-2" aria-label={t("workspace.evidenceSummary")}>
      {chips.map((c) => {
        const ChipIcon = iconMap[c.icon];
        const body = (
          <>
            <ChipIcon size={12} className="text-muted-foreground" />
            {c.key === "map" ? c.label : c.count}
          </>
        );
        return onSelect ? (
          <Button
            key={c.key}
            type="button"
            title={c.label}
            onClick={() => onSelect(c.tab)}
            variant="ghost"
            size="sm"
            className={`${baseClass} h-auto rounded-full px-2 py-0.5 text-[11px] hover:bg-primary/10 hover:text-primary hover:border-primary/20`}
          >
            {body}
          </Button>
        ) : (
          <span key={c.key} title={c.label} className={baseClass}>
            {body}
          </span>
        );
      })}
    </div>
  );
}
