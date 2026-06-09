"use client";

import { useState } from "react";
import { iconMap } from "@/lib/iconMap";
import { useI18n } from "@/hooks/useI18n";
import { useAnswerEvidence } from "@/hooks/useAnswerEvidence";
import { Button } from "@/components/ui/button";
import type { MessageMetrics, NodeTraceEntry } from "@/lib/types";
import { KnowledgeTab } from "./KnowledgeTab";
import { MemoryTab } from "./MemoryTab";
import { OutputsTab } from "./OutputsTab";
import { InspectorTab } from "./InspectorTab";

/** The four read-only evidence tabs (the Documents tab is editing chrome, excluded here). */
export type EvidenceTabId = "knowledge" | "memory" | "outputs" | "inspector";

const TABS: { id: EvidenceTabId; icon: string; labelKey: string }[] = [
  { id: "knowledge", icon: "menu_book", labelKey: "workspace.tabKnowledge" },
  { id: "memory", icon: "memory", labelKey: "workspace.tabMemory" },
  { id: "outputs", icon: "build", labelKey: "workspace.tabOutputs" },
  { id: "inspector", icon: "account_tree", labelKey: "workspace.tabInspector" },
];

/**
 * Self-contained, read-only bundle of the Knowledge / Memory / Outputs /
 * Inspector evidence tabs with a lightweight local tab switcher — no Documents
 * tab, no editing chrome. Drives the same presentational tab components the live
 * workspace panel uses, but for an arbitrary (e.g. persisted, historical) answer.
 *
 * Feed it from a selected message: `metrics` → `deriveAnswerEvidence`, and the
 * persisted `nodeTrace`. Both already flow through `toUiMessage` for stored
 * conversations, so a reloaded answer renders identically to a live one.
 */
export function WorkspaceEvidenceTabs({
  metrics,
  nodeTrace = [],
  className,
}: {
  metrics: MessageMetrics | null | undefined;
  nodeTrace?: NodeTraceEntry[];
  className?: string;
}) {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState<EvidenceTabId>("knowledge");
  const evidence = useAnswerEvidence(metrics);

  const tabCounts: Record<EvidenceTabId, number> = {
    knowledge: evidence.sourceCount,
    memory: evidence.memoryCount,
    outputs: evidence.toolCount + (evidence.mapData ? 1 : 0),
    inspector: nodeTrace.length,
  };

  return (
    <div className={`flex min-h-0 flex-col ${className ?? ""}`}>
      <div
        role="tablist"
        aria-label={t("workspace.answerEvidence")}
        className="flex items-center gap-1 px-2 py-1.5 border-b border-border shrink-0 overflow-x-auto"
      >
        {TABS.map((tab) => {
          const active = activeTab === tab.id;
          const label = t(tab.labelKey);
          const count = tabCounts[tab.id];
          const TabIcon = iconMap[tab.icon];
          return (
            <Button
              type="button"
              key={tab.id}
              role="tab"
              aria-selected={active}
              aria-label={label}
              title={label}
              onClick={() => setActiveTab(tab.id)}
              variant="ghost"
              size="sm"
              className={`group flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors
                ${active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-surface-muted"}`}
            >
              <TabIcon size={16} />
              <span className="whitespace-nowrap">{label}</span>
              {count > 0 && (
                <span className="rounded-full px-1.5 text-[10px] leading-4 bg-primary/15">
                  {count}
                </span>
              )}
            </Button>
          );
        })}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto flex flex-col">
        {activeTab === "knowledge" && <KnowledgeTab evidence={evidence} />}
        {activeTab === "memory" && <MemoryTab evidence={evidence} />}
        {activeTab === "outputs" && <OutputsTab evidence={evidence} />}
        {activeTab === "inspector" && <InspectorTab nodeTrace={nodeTrace} isStreaming={false} />}
      </div>
    </div>
  );
}
