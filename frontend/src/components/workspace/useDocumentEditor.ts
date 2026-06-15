"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import { uploadConversationAttachment } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import type { ConversationAttachment } from "@/lib/types";
import type { WorkspaceDocument } from "./types";
import { mimeTypeForFilename, workspaceAuthHeaders } from "./utils";

interface UseDocumentEditorArgs {
  documents: WorkspaceDocument[];
  conversationId?: string | null;
  writebackSavedDocId?: string | null;
  onActiveDocumentChange?: (docId: string | null) => void;
  onAttachmentUploaded?: (attachment: ConversationAttachment) => void;
  onDocumentsRefresh?: () => void;
  /** Called after a successful save (e.g. to refresh an open version-history panel). */
  onAfterSave?: (docId: string) => void;
  /** Shared processing token owned by the parent tab (preserves cross-action disable). */
  setProcessingAction: (value: string | null) => void;
}

/**
 * Owns the inline document editor (open doc id, content, dirty tracking) plus
 * the editor-scoped actions: load, save, re-attach to chat, and download.
 *
 * Dirty tracking: `savedContent` is the last server-confirmed text and
 * `editorDocVersion` the version it was loaded at. `isDirty` is `editorContent
 * !== savedContent`. Saves send `expected_version` for optimistic concurrency —
 * a 409 means the document changed elsewhere, so we reload rather than clobber.
 * Reloads the open doc when the LLM saves a new version via writeback (only when
 * clean), and notifies the parent of the active document via `onActiveDocumentChange`.
 */
export function useDocumentEditor({
  documents,
  conversationId,
  writebackSavedDocId,
  onActiveDocumentChange,
  onAttachmentUploaded,
  onDocumentsRefresh,
  onAfterSave,
  setProcessingAction,
}: UseDocumentEditorArgs) {
  const { t } = useI18n();
  const [editorDocId, setEditorDocIdRaw] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState("");
  // Server-confirmed baseline + its version, for dirty detection and optimistic saves.
  const [savedContent, setSavedContent] = useState("");
  const [editorDocVersion, setEditorDocVersion] = useState<number | null>(null);

  const isDirty = editorDocId != null && editorContent !== savedContent;
  // Ref mirror so effects (writeback reload) can read dirtiness without re-subscribing.
  const isDirtyRef = useRef(isDirty);
  isDirtyRef.current = isDirty;
  // Forward ref to flushSave (assigned after it is defined) so `closeEditor` can
  // auto-flush without a circular useCallback dependency.
  const flushSaveRef = useRef<(() => Promise<boolean>) | null>(null);

  const setEditorDocId = useCallback(
    (docId: string | null) => {
      setEditorDocIdRaw(docId);
      onActiveDocumentChange?.(docId);
    },
    [onActiveDocumentChange],
  );

  const getDocumentName = useCallback(
    (docId: string) => documents.find((d) => d.id === docId)?.filename || docId,
    [documents],
  );

  const getDocument = useCallback(
    (docId: string) => documents.find((d) => d.id === docId),
    [documents],
  );

  const openEditor = useCallback(
    async (docId: string) => {
      setProcessingAction(docId);
      try {
        const response = await fetch(`/api/proxy/api/workspace/documents/${docId}`, {
          method: "GET",
          headers: workspaceAuthHeaders({ "Content-Type": "application/json" }),
        });
        if (!response.ok) {
          throw new Error(`Failed to load document: ${response.statusText}`);
        }
        const data = await response.json();
        const content = data.content_text || "";
        setEditorDocId(docId);
        setEditorContent(content);
        setSavedContent(content);
        setEditorDocVersion(typeof data.version === "number" ? data.version : null);
      } catch (err) {
        const message = err instanceof Error ? err.message : t("workspace.loadDocumentFailed");
        toast.error(t("workspace.loadFailed"), { description: message });
      } finally {
        setProcessingAction(null);
      }
    },
    [setProcessingAction, setEditorDocId, t],
  );

  const closeEditor = useCallback(
    (opts?: { force?: boolean }) => {
      // Auto-save unsaved edits before closing (unless forced to discard). On a
      // save failure keep the editor open so the user doesn't lose their text.
      if (!opts?.force && isDirtyRef.current && editorDocId) {
        void flushSaveRef.current?.().then((ok) => {
          if (ok) {
            setEditorDocId(null);
            setEditorContent("");
            setSavedContent("");
            setEditorDocVersion(null);
          }
        });
        return;
      }
      setEditorDocId(null);
      setEditorContent("");
      setSavedContent("");
      setEditorDocVersion(null);
    },
    [setEditorDocId, editorDocId],
  );

  // When the LLM has saved a new version via writeback, reload the editor if that
  // doc is open AND the user has no unsaved edits. If dirty, don't clobber — warn
  // and let the next manual save surface the conflict (409 → reload).
  useEffect(() => {
    if (writebackSavedDocId && editorDocId === writebackSavedDocId) {
      if (!isDirtyRef.current) {
        openEditor(writebackSavedDocId);
      } else {
        toast.warning(t("workspace.writebackWhileDirty"));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [writebackSavedDocId]);

  /**
   * Persist the current editor content. Returns true on success. Sends
   * `expected_version` so a concurrent change yields a 409 (reload + inform)
   * instead of silently overwriting.
   */
  const flushSave = useCallback(async (): Promise<boolean> => {
    if (!editorDocId) return true; // no editor open — nothing to flush
    if (!editorContent.trim()) return true; // empty content — skip save so send isn't blocked
    const contentAtSave = editorContent;
    setProcessingAction(editorDocId);
    try {
      const docName = getDocumentName(editorDocId);
      const mime = mimeTypeForFilename(docName);
      const blob = new Blob([contentAtSave], { type: `${mime};charset=utf-8` });
      const file = new File([blob], docName, { type: mime });
      const formData = new FormData();
      formData.append("file", file);
      if (editorDocVersion != null) {
        formData.append("expected_version", String(editorDocVersion));
      }

      const response = await fetch(`/api/proxy/api/workspace/documents/${editorDocId}`, {
        method: "PUT",
        headers: workspaceAuthHeaders(),
        body: formData,
      });
      if (response.status === 409) {
        toast.error(t("workspace.saveConflict"));
        await openEditor(editorDocId);
        return false;
      }
      if (!response.ok) {
        throw new Error(`Failed to update document: ${response.statusText}`);
      }
      const data = await response.json().catch(() => null);
      const updated = data?.document;
      // Baseline moves to the text we just saved; version follows the server.
      setSavedContent(contentAtSave);
      if (updated && typeof updated.version === "number") {
        setEditorDocVersion(updated.version);
      }
      toast.success(t("workspace.saveChanges"), { description: `${docName} saved successfully` });
      onDocumentsRefresh?.();
      onAfterSave?.(editorDocId);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : t("workspace.saveChangesFailed");
      toast.error(t("workspace.saveFailed"), { description: message });
      return false;
    } finally {
      setProcessingAction(null);
    }
  }, [editorDocId, editorContent, editorDocVersion, getDocumentName, onDocumentsRefresh, onAfterSave, openEditor, setProcessingAction, t]);

  flushSaveRef.current = flushSave;

  const saveEditor = useCallback(() => {
    void flushSave();
  }, [flushSave]);

  const reattachEditor = useCallback(async () => {
    if (!editorDocId || !editorContent.trim()) return;
    const docName = getDocumentName(editorDocId);
    const file = new File([editorContent], docName, { type: mimeTypeForFilename(docName) });

    if (!conversationId) {
      toast.error(t("workspace.reattachFailed"), { description: t("workspace.noActiveConversation") });
      return;
    }

    try {
      const uploaded = await uploadConversationAttachment(conversationId, file);
      if (!uploaded) {
        toast.error(t("workspace.reattachFailed"), { description: t("workspace.attachToChat") });
        return;
      }
      onAttachmentUploaded?.(uploaded);
      toast.success(t("workspace.attachedToChat"), { description: uploaded.original_name });
    } catch (err) {
      toast.error(t("workspace.reattachFailed"), {
        description: err instanceof Error ? err.message : t("workspace.uploadFailed"),
      });
    }
  }, [conversationId, editorDocId, editorContent, onAttachmentUploaded, getDocumentName, t]);

  const downloadEditor = useCallback(() => {
    if (!editorDocId || !editorContent) return;
    const docName = getDocumentName(editorDocId);
    const blob = new Blob([editorContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = docName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success(t("workspace.downloaded"), { description: docName });
  }, [editorDocId, editorContent, getDocumentName, t]);

  return {
    editorDocId,
    editorContent,
    setEditorContent,
    isDirty,
    getDocumentName,
    getDocument,
    openEditor,
    closeEditor,
    saveEditor,
    flushSave,
    reattachEditor,
    downloadEditor,
  };
}
