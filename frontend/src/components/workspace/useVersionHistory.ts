"use client";

import { useState, useCallback } from "react";
import { toast } from "sonner";
import type { VersionEntry } from "./types";
import { workspaceAuthHeaders } from "./utils";

interface UseVersionHistoryArgs {
  onDocumentsRefresh?: () => void;
  /** Shared processing token owned by the parent tab (preserves cross-action disable). */
  setProcessingAction: (value: string | null) => void;
  /** Called after a successful restore (e.g. to reload the editor if the doc is open). */
  onAfterRestore?: (docId: string) => void;
}

/**
 * Owns the version-history panel state for a single document: the loaded
 * version list, an optional inline preview, and the restore action.
 */
export function useVersionHistory({
  onDocumentsRefresh,
  setProcessingAction,
  onAfterRestore,
}: UseVersionHistoryArgs) {
  const [historyDocId, setHistoryDocId] = useState<string | null>(null);
  const [historyVersions, setHistoryVersions] = useState<VersionEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [previewVersionId, setPreviewVersionId] = useState<string | null>(null);
  const [previewContent, setPreviewContent] = useState<string>("");

  const closeHistory = useCallback(() => {
    setHistoryDocId(null);
    setHistoryVersions([]);
    setPreviewVersionId(null);
    setPreviewContent("");
  }, []);

  const loadHistory = useCallback(async (docId: string) => {
    setHistoryDocId(docId);
    setHistoryLoading(true);
    setHistoryVersions([]);
    setPreviewVersionId(null);
    setPreviewContent("");
    try {
      const res = await fetch(`/api/proxy/api/workspace/documents/${docId}/versions`, {
        headers: workspaceAuthHeaders(),
      });
      if (!res.ok) throw new Error(res.statusText);
      const data: VersionEntry[] = await res.json();
      setHistoryVersions(data);
    } catch {
      toast.error("Failed to load version history");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const previewVersion = useCallback(
    async (docId: string, version: number, versionId: string) => {
      if (previewVersionId === versionId) {
        setPreviewVersionId(null);
        setPreviewContent("");
        return;
      }
      try {
        const res = await fetch(`/api/proxy/api/workspace/documents/${docId}/versions/${version}`, {
          headers: workspaceAuthHeaders(),
        });
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        setPreviewVersionId(versionId);
        setPreviewContent(data.content_text || "");
      } catch {
        toast.error("Failed to load version preview");
      }
    },
    [previewVersionId],
  );

  const restoreVersion = useCallback(
    async (docId: string, version: number) => {
      setProcessingAction(`restore-${docId}-${version}`);
      try {
        const res = await fetch(
          `/api/proxy/api/workspace/documents/${docId}/versions/${version}/restore`,
          { method: "POST", headers: workspaceAuthHeaders() },
        );
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || res.statusText);
        }
        toast.success(`Restored to v${version}`);
        setHistoryDocId(null);
        setHistoryVersions([]);
        onDocumentsRefresh?.();
        onAfterRestore?.(docId);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Restore failed";
        toast.error("Restore failed", { description: message });
      } finally {
        setProcessingAction(null);
      }
    },
    [onDocumentsRefresh, onAfterRestore, setProcessingAction],
  );

  return {
    historyDocId,
    historyVersions,
    historyLoading,
    previewVersionId,
    previewContent,
    loadHistory,
    previewVersion,
    restoreVersion,
    closeHistory,
  };
}
