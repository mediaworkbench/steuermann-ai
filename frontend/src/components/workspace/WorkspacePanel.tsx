"use client";

import { Folder, Info, X } from "lucide-react";
import { iconMap } from "@/lib/iconMap";
import { useI18n } from "@/hooks/useI18n";
import { useAnswerEvidence } from "@/hooks/useAnswerEvidence";
import { useWorkspacePanel } from "@/context/WorkspacePanelContext";
import { Button } from "@/components/ui/button";
import type { WorkspacePanelProps, WorkspaceTabId } from "./types";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { DocumentsTab } from "./DocumentsTab";
import { KnowledgeTab } from "./KnowledgeTab";
import { MemoryTab } from "./MemoryTab";
import { OutputsTab } from "./OutputsTab";
import { InspectorTab } from "./InspectorTab";
import { WorkspacePanelShell } from "./WorkspacePanelShell";
import { WorkspacePanelTopBar } from "./WorkspacePanelTopBar";

const TABS: { id: WorkspaceTabId; icon: string; labelKey: string }[] = [
  { id: "documents", icon: "folder_open", labelKey: "workspace.tabDocuments" },
  { id: "knowledge", icon: "menu_book", labelKey: "workspace.tabKnowledge" },
  { id: "memory", icon: "memory", labelKey: "workspace.tabMemory" },
  { id: "outputs", icon: "build", labelKey: "workspace.tabOutputs" },
  { id: "inspector", icon: "account_tree", labelKey: "workspace.tabInspector" },
];

/**
 * Modular container for the chat-right workspace panel. Renders the shell
 * chrome (mobile toggle/overlay + sliding panel) and a tab bar, then hosts the
 * active section. The Documents tab stays mounted across tab switches (via
 * display:contents) so an in-progress edit and the active-document selection
 * survive peeking at the evidence tabs.
 *
 * Internal tab state is local for now; R1.4 lifts it into WorkspacePanelContext.
 */
export function WorkspacePanel({
  isOpen,
  onToggle,
  conversationId,
  documents,
  isLoading = false,
  onDocumentsRefresh,
  onEnsureConversation,
  onAttachmentUploaded,
  documentsLoading = false,
  documentsError = null,
  onRetryDocuments,
  answerMetrics = null,
  nodeTrace = [],
  isStreaming = false,
  historicalAnswer = false,
  onJumpToLatest,
}: WorkspacePanelProps) {
  const { t } = useI18n();
  const { activeTab, setActiveTab } = useWorkspacePanel();
  const evidence = useAnswerEvidence(answerMetrics);
  // The banner only applies to the per-answer evidence tabs; Documents is
  // conversation-scoped, so a "viewing an earlier answer" note there is misleading.
  const showHistoricalBanner = historicalAnswer && activeTab !== "documents";

  const tabCounts: Record<WorkspaceTabId, number> = {
    documents: documents.length,
    knowledge: evidence.sourceCount,
    memory: evidence.memoryCount,
    outputs: evidence.toolCount + (evidence.mapData ? 1 : 0),
    inspector: nodeTrace.length,
  };

  return (
    <>
      {/* Toggle button (visible on mobile/tablet) */}
      <Tooltip>
        <TooltipTrigger render={
          <Button
            type="button"
            onClick={onToggle}
            variant="primary"
            size="sm"
            className="fixed bottom-20 right-4 z-40 rounded-full p-3 shadow-lg transition-colors md:hidden"
            aria-label={t("workspace.toggleSidebar")}
          >
            {isOpen ? <X size={24} /> : <Folder size={24} />}
          </Button>
        } />
        <TooltipContent>{t("workspace.toggleSidebar")}</TooltipContent>
      </Tooltip>

      {/* Sidebar overlay (mobile) */}
      {isOpen && (
        <div
          className="fixed inset-0 z-10 bg-black/20 md:hidden"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}

      {/* Sidebar panel */}
      <WorkspacePanelShell isOpen={isOpen} onToggle={onToggle}>
        <WorkspacePanelTopBar
          title={t("chat.workspace")}
          onClose={onToggle}
          closeLabel={t("workspace.closeSidebar")}
        />

        {/* Tab bar — icon-forward segmented control; the active tab reveals its
            label so even long localized labels fit the narrow panel. */}
        <div
          role="tablist"
          aria-label={t("chat.workspace")}
          className="flex items-center gap-1 px-2 py-1.5 border-b border-border shrink-0 overflow-x-auto"
        >
          {TABS.map((tab) => {
            const active = activeTab === tab.id;
            const label = t(tab.labelKey);
            const count = tabCounts[tab.id];
            // Count shows only on the active tab so inactive tabs stay compact
            // and all five fit the narrow panel without horizontal scrolling.
            const showCount = active && count > 0;
            const TabIcon = iconMap[tab.icon];
            return (
              <Tooltip key={tab.id}>
                <TooltipTrigger render={
                  <Button
                    type="button"
                    role="tab"
                    aria-selected={active}
                    aria-label={label}
                    onClick={() => setActiveTab(tab.id)}
                    variant="ghost"
                    size="sm"
                    className={`group flex shrink-0 items-center gap-1.5 rounded-lg text-xs font-medium transition-colors
                      ${active
                        ? "bg-primary/10 text-primary px-2.5 py-1.5"
                        : "text-muted-foreground hover:text-foreground hover:bg-surface-muted p-1.5"}`}
                  >
                    <TabIcon size={16} />
                    {active && <span className="whitespace-nowrap">{label}</span>}
                    {showCount && (
                      <span className="rounded-full px-1.5 text-[10px] leading-4 bg-primary/15">
                        {count}
                      </span>
                    )}
                  </Button>
                } />
                <TooltipContent>{label}</TooltipContent>
              </Tooltip>
            );
          })}
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-y-auto flex flex-col">
          {/* Pinned to an earlier answer — the evidence tabs describe that answer. */}
          {showHistoricalBanner && (
            <div className="flex items-center gap-2 px-3 py-1.5 text-xs border-b border-border bg-surface-muted text-muted-foreground shrink-0">
              <Info size={13} className="shrink-0 text-primary" />
              <span className="flex-1 truncate">{t("workspace.viewingEarlierAnswer")}</span>
              {onJumpToLatest && (
                <Button
                  type="button"
                  onClick={onJumpToLatest}
                  variant="ghost"
                  size="sm"
                  className="h-auto shrink-0 rounded-md px-2 py-0.5 text-xs font-medium text-primary hover:bg-primary/10"
                >
                  {t("workspace.jumpToLatest")}
                </Button>
              )}
            </div>
          )}
          {/* Documents stays mounted (display:contents) to preserve editor/active-doc state. */}
          <div className={activeTab === "documents" ? "contents" : "hidden"}>
            <DocumentsTab
              conversationId={conversationId}
              documents={documents}
              isLoading={isLoading}
              onDocumentsRefresh={onDocumentsRefresh}
              onEnsureConversation={onEnsureConversation}
              onAttachmentUploaded={onAttachmentUploaded}
              documentsLoading={documentsLoading}
              documentsError={documentsError}
              onRetryDocuments={onRetryDocuments}
            />
          </div>
          {activeTab === "knowledge" && <KnowledgeTab evidence={evidence} />}
          {activeTab === "memory" && <MemoryTab evidence={evidence} />}
          {activeTab === "outputs" && <OutputsTab evidence={evidence} />}
          {activeTab === "inspector" && (
            <InspectorTab
              nodeTrace={nodeTrace}
              isStreaming={isStreaming}
              // Latest answer → its post-response nodes are running in the background.
              isLatestAnswer={!historicalAnswer}
              // Only offer the Outputs deep-link when there's tool output to view.
              onOpenOutputs={evidence.tools.length > 0 ? () => setActiveTab("outputs") : undefined}
            />
          )}
        </div>
      </WorkspacePanelShell>
    </>
  );
}
