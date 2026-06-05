"use client";

import { useState } from "react";
import { Icon } from "../Icon";
import { useI18n } from "@/hooks/useI18n";
import type { WorkspacePanelProps, WorkspaceTabId } from "./types";
import { DocumentsTab } from "./DocumentsTab";
import { KnowledgeTab } from "./KnowledgeTab";
import { MemoryTab } from "./MemoryTab";
import { OutputsTab } from "./OutputsTab";

const TABS: { id: WorkspaceTabId; icon: string; labelKey: string }[] = [
  { id: "documents", icon: "folder_open", labelKey: "workspace.tabDocuments" },
  { id: "knowledge", icon: "menu_book", labelKey: "workspace.tabKnowledge" },
  { id: "memory", icon: "memory", labelKey: "workspace.tabMemory" },
  { id: "outputs", icon: "build", labelKey: "workspace.tabOutputs" },
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
}: WorkspacePanelProps) {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState<WorkspaceTabId>("documents");

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
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Icon name="folder_open" size={20} className="text-pacific-blue" />
            <h3 className="font-semibold text-sm text-evergreen">{t("chat.workspace")}</h3>
          </div>
          <button
            onClick={onToggle}
            className="md:hidden p-1 hover:bg-gray-100 rounded text-evergreen/60"
            aria-label={t("workspace.closeSidebar")}
          >
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* Tab bar */}
        <div
          role="tablist"
          aria-label={t("chat.workspace")}
          className="flex items-stretch gap-0.5 px-1.5 border-b border-gray-200 shrink-0 overflow-x-auto"
        >
          {TABS.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={active}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1 px-2.5 py-2 text-xs font-medium whitespace-nowrap border-b-2 transition-colors -mb-px
                  ${active
                    ? "border-pacific-blue text-pacific-blue"
                    : "border-transparent text-evergreen/50 hover:text-evergreen hover:border-gray-200"}`}
              >
                <Icon name={tab.icon} size={15} />
                <span>{t(tab.labelKey)}</span>
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
            />
          </div>
          {activeTab === "knowledge" && <KnowledgeTab />}
          {activeTab === "memory" && <MemoryTab />}
          {activeTab === "outputs" && <OutputsTab />}
        </div>
      </div>
    </>
  );
}
