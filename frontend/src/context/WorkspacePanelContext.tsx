"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";
import type { WorkspaceTabId } from "@/components/workspace/types";

const VALID_TABS: WorkspaceTabId[] = ["documents", "knowledge", "memory", "outputs"];
const ACTIVE_TAB_KEY = "workspace.activeTab";

interface WorkspacePanelContextValue {
  /** Selected right-panel tab. Persisted to localStorage. */
  activeTab: WorkspaceTabId;
  setActiveTab: (tab: WorkspaceTabId) => void;
  /** Documents tab search/filter text. In-memory only (not persisted). */
  documentQuery: string;
  setDocumentQuery: (query: string) => void;
}

const WorkspacePanelContext = createContext<WorkspacePanelContextValue | null>(null);

export function useWorkspacePanel(): WorkspacePanelContextValue {
  const ctx = useContext(WorkspacePanelContext);
  if (!ctx) throw new Error("useWorkspacePanel must be used within WorkspacePanelProvider");
  return ctx;
}

/**
 * Holds the workspace panel's *internal* view state (active tab + document
 * filter) so it is shared across the panel, the chat evidence chips, and any
 * future consumer — without entangling it with conversation/stream state, which
 * stays in their existing providers. Only the active tab is persisted.
 */
export function WorkspacePanelProvider({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTabState] = useState<WorkspaceTabId>("documents");
  const [documentQuery, setDocumentQuery] = useState("");

  // Restore the persisted tab on mount. Done in an effect (not in the initial
  // state) so the first client render matches SSR and avoids a hydration mismatch.
  useEffect(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_TAB_KEY);
      if (stored && (VALID_TABS as string[]).includes(stored)) {
        setActiveTabState(stored as WorkspaceTabId);
      }
    } catch {
      /* localStorage unavailable — fall back to the default tab */
    }
  }, []);

  const setActiveTab = useCallback((tab: WorkspaceTabId) => {
    setActiveTabState(tab);
    try {
      localStorage.setItem(ACTIVE_TAB_KEY, tab);
    } catch {
      /* ignore persistence failures */
    }
  }, []);

  return (
    <WorkspacePanelContext.Provider value={{ activeTab, setActiveTab, documentQuery, setDocumentQuery }}>
      {children}
    </WorkspacePanelContext.Provider>
  );
}
