"use client";

import { useState, useCallback, useEffect } from "react";
import type { WorkspaceDocument } from "@/components/WorkspaceSidebar";

interface UseWorkspaceDocumentsResult {
  documents: WorkspaceDocument[];
  documentsLoading: boolean;
  documentsError: string | null;
  refresh: () => Promise<void>;
}

export function useWorkspaceDocuments(): UseWorkspaceDocumentsResult {
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      // limit=1000 is the backend cap; the list is virtualized + client-side
      // searched, so loading the full set keeps search + active-doc restore
      // working. (True server-side pagination is deferred — see workspace-refactor.md.)
      const response = await fetch("/api/proxy/api/workspace/documents?limit=1000", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to load documents: ${response.statusText}`);
      }

      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      // Documents are optional; surface the failure in the panel rather than crash.
      console.warn("Failed to load workspace documents:", err);
      setDocumentsError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { documents, documentsLoading, documentsError, refresh };
}
