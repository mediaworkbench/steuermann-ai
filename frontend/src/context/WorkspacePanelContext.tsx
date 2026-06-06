"use client";

import { createContext, useContext, useState, useEffect, useCallback } from "react";
import type { WorkspaceTabId } from "@/components/workspace/types";

const VALID_TABS: WorkspaceTabId[] = ["documents", "knowledge", "memory", "outputs", "inspector"];
const ACTIVE_TAB_KEY = "workspace.activeTab";

interface WorkspacePanelContextValue {
  /** Selected right-panel tab. Persisted to localStorage. */
  activeTab: WorkspaceTabId;
  setActiveTab: (tab: WorkspaceTabId) => void;
}

const WorkspacePanelContext = createContext<WorkspacePanelContextValue | null>(null);

export function useWorkspacePanel(): WorkspacePanelContextValue {
  const ctx = useContext(WorkspacePanelContext);
  if (!ctx) throw new Error("useWorkspacePanel must be used within WorkspacePanelProvider");
  return ctx;
}

/**
 * Holds the workspace panel's *internal* view state (the active tab) so it is
 * shared across the panel and the chat evidence chips (chip → tab) — without
 * entangling it with conversation/stream state, which stays in their existing
 * providers. The active tab is persisted; transient per-tab state (e.g. the
 * Documents filter) stays local to its tab so it does not leak across routes.
 */
export function WorkspacePanelProvider({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTabState] = useState<WorkspaceTabId>("documents");

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
    <WorkspacePanelContext.Provider value={{ activeTab, setActiveTab }}>
      {children}
    </WorkspacePanelContext.Provider>
  );
}
