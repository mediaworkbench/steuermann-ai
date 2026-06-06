"use client";

import { Icon } from "../Icon";
import { useI18n } from "@/hooks/useI18n";
import { useAnswerEvidence } from "@/hooks/useAnswerEvidence";
import { useWorkspacePanel } from "@/context/WorkspacePanelContext";
import type { WorkspacePanelProps, WorkspaceTabId } from "./types";
import { DocumentsTab } from "./DocumentsTab";
import { KnowledgeTab } from "./KnowledgeTab";
import { MemoryTab } from "./MemoryTab";
import { OutputsTab } from "./OutputsTab";
import { InspectorTab } from "./InspectorTab";

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
  writebackSavedDocId,
  onActiveDocumentChange,
  documentsLoading = false,
  documentsError = null,
  onRetryDocuments,
  answerMetrics = null,
  nodeTrace = [],
  isStreaming = false,
}: WorkspacePanelProps) {
  const { t } = useI18n();
  const { activeTab, setActiveTab } = useWorkspacePanel();
  const evidence = useAnswerEvidence(answerMetrics);

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
      <button
        onClick={onToggle}
        className="md:hidden fixed bottom-20 right-4 z-40 rounded-full p-3 bg-pacific-blue text-white shadow-lg hover:bg-pacific-blue/90 transition-colors"
        title={t("workspace.toggleSidebar")}
        aria-label={t("workspace.toggleSidebar")}
      >
        <Icon name={isOpen ? "close" : "folder"} size={24} />
      </button>

      {/* Sidebar overlay (mobile) */}
      {isOpen && (
        <div
          className="fixed inset-0 z-10 bg-black/20 md:hidden"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}

      {/* Sidebar panel */}
      <div
        className={`fixed right-0 top-16 h-[calc(100vh-4rem)] z-10
                     md:sticky md:self-start md:top-20 md:h-[calc(100vh-5rem)] md:z-0
                     w-80 ${isOpen ? "md:w-64 lg:w-72" : "md:w-0"} bg-white border-l border-gray-200
                     transition-all duration-200
                     flex flex-col overflow-hidden min-h-0
                     ${isOpen ? "translate-x-0" : "translate-x-full md:translate-x-0 md:border-l-0"}`}
      >
        {/* Header */}
        <div className="px-3 py-2.5 border-b border-gray-200/80 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="grid place-items-center w-7 h-7 rounded-lg bg-pacific-blue/10 text-pacific-blue shrink-0">
              <Icon name="folder_open" size={16} />
            </span>
            <h3 className="font-semibold text-sm text-evergreen tracking-tight truncate">
              {t("chat.workspace")}
            </h3>
          </div>
          <button
            onClick={onToggle}
            className="md:hidden p-1.5 hover:bg-gray-100 rounded-lg text-evergreen/60"
            aria-label={t("workspace.closeSidebar")}
          >
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* Tab bar — icon-forward segmented control; the active tab reveals its
            label so even long localized labels fit the narrow panel. */}
        <div
          role="tablist"
          aria-label={t("chat.workspace")}
          className="flex items-center gap-1 px-2 py-1.5 border-b border-gray-200/80 shrink-0"
        >
          {TABS.map((tab) => {
            const active = activeTab === tab.id;
            const label = t(tab.labelKey);
            const count = tabCounts[tab.id];
            const showCount = count > 0;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={active}
                aria-label={label}
                title={label}
                onClick={() => setActiveTab(tab.id)}
                className={`group flex items-center gap-1.5 rounded-lg text-xs font-medium transition-colors
                  ${active
                    ? "bg-pacific-blue/10 text-pacific-blue px-2.5 py-1.5"
                    : "text-evergreen/45 hover:text-evergreen hover:bg-gray-100 p-1.5"}`}
              >
                <Icon name={tab.icon} size={16} />
                {active && <span className="whitespace-nowrap">{label}</span>}
                {showCount && (
                  <span
                    className={`rounded-full px-1.5 text-[10px] leading-4 ${
                      active ? "bg-pacific-blue/15" : "bg-gray-200 text-evergreen/50"
                    }`}
                  >
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-y-auto flex flex-col">
          {/* Documents stays mounted (display:contents) to preserve editor/active-doc state. */}
          <div className={activeTab === "documents" ? "contents" : "hidden"}>
            <DocumentsTab
              conversationId={conversationId}
              documents={documents}
              isLoading={isLoading}
              onDocumentsRefresh={onDocumentsRefresh}
              onEnsureConversation={onEnsureConversation}
              onAttachmentUploaded={onAttachmentUploaded}
              writebackSavedDocId={writebackSavedDocId}
              onActiveDocumentChange={onActiveDocumentChange}
              documentsLoading={documentsLoading}
              documentsError={documentsError}
              onRetryDocuments={onRetryDocuments}
            />
          </div>
          {activeTab === "knowledge" && <KnowledgeTab evidence={evidence} />}
          {activeTab === "memory" && <MemoryTab evidence={evidence} />}
          {activeTab === "outputs" && <OutputsTab evidence={evidence} />}
          {activeTab === "inspector" && <InspectorTab nodeTrace={nodeTrace} isStreaming={isStreaming} />}
        </div>
      </div>
    </>
  );
}
