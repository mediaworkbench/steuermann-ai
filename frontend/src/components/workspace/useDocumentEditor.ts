"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { toast } from "sonner";
import { uploadConversationAttachment } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { CURRENT_USER_ID } from "@/lib/runtime";
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
  /** Shared processing token owned by the parent tab (preserves cross-action disable). */
  setProcessingAction: (value: string | null) => void;
}

/**
 * Owns the inline document editor (open doc id, content, resizable height) plus
 * the editor-scoped actions: load, save, re-attach to chat, and download.
 * Reloads the open doc when the LLM saves a new version via writeback, and
 * notifies the parent of the active document via `onActiveDocumentChange`.
 */
export function useDocumentEditor({
  documents,
  conversationId,
  writebackSavedDocId,
  onActiveDocumentChange,
  onAttachmentUploaded,
  onDocumentsRefresh,
  setProcessingAction,
}: UseDocumentEditorArgs) {
  const { t } = useI18n();
  const [editorDocId, setEditorDocIdRaw] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState("");
  const [editorHeight, setEditorHeight] = useState(220);
  const isDraggingRef = useRef(false);

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
        setEditorDocId(docId);
        setEditorContent(data.content_text || "");
      } catch (err) {
        const message = err instanceof Error ? err.message : t("workspace.loadDocumentFailed");
        toast.error(t("workspace.loadFailed"), { description: message });
      } finally {
        setProcessingAction(null);
      }
    },
    [setProcessingAction, setEditorDocId, t],
  );

  const closeEditor = useCallback(() => {
    setEditorDocId(null);
    setEditorContent("");
  }, [setEditorDocId]);

  // When the LLM has saved a new version via writeback, reload the editor if that doc is open.
  useEffect(() => {
    if (writebackSavedDocId && editorDocId === writebackSavedDocId) {
      openEditor(writebackSavedDocId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [writebackSavedDocId]);

  const onResizeStart = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      isDraggingRef.current = true;
      const startY = e.clientY;
      const startHeight = editorHeight;

      const onMouseMove = (moveEvent: MouseEvent) => {
        if (!isDraggingRef.current) return;
        const delta = moveEvent.clientY - startY;
        setEditorHeight(Math.min(420, Math.max(120, startHeight + delta)));
      };

      const onMouseUp = () => {
        isDraggingRef.current = false;
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      };

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    },
    [editorHeight],
  );

  const saveEditor = useCallback(async () => {
    if (!editorDocId || !editorContent.trim()) return;
    setProcessingAction(editorDocId);
    try {
      const docName = getDocumentName(editorDocId);
      const blob = new Blob([editorContent], { type: "text/plain;charset=utf-8" });
      const file = new File([blob], docName, { type: "text/plain" });
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`/api/proxy/api/workspace/documents/${editorDocId}`, {
        method: "PUT",
        headers: workspaceAuthHeaders(),
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Failed to update document: ${response.statusText}`);
      }
      toast.success(t("workspace.saveChanges"), { description: `${docName} saved successfully` });
      onDocumentsRefresh?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : t("workspace.saveChangesFailed");
      toast.error(t("workspace.saveFailed"), { description: message });
    } finally {
      setProcessingAction(null);
    }
  }, [editorDocId, editorContent, getDocumentName, onDocumentsRefresh, setProcessingAction, t]);

  const reattachEditor = useCallback(async () => {
    if (!editorDocId || !editorContent.trim()) return;
    const docName = getDocumentName(editorDocId);
    const file = new File([editorContent], docName, { type: mimeTypeForFilename(docName) });

    if (!conversationId) {
      toast.error(t("workspace.reattachFailed"), { description: t("workspace.noActiveConversation") });
      return;
    }

    try {
      const uploaded = await uploadConversationAttachment(conversationId, file, CURRENT_USER_ID);
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
    editorHeight,
    getDocumentName,
    openEditor,
    closeEditor,
    onResizeStart,
    saveEditor,
    reattachEditor,
    downloadEditor,
  };
}
