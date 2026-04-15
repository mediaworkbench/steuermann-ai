"use client";

import { useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import { Icon } from "./Icon";
import { uploadConversationAttachment } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type { ConversationAttachment } from "@/lib/types";

export interface WorkspaceDocument {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  version: number;
  created_at?: string;
  updated_at?: string;
}

export interface WorkspaceSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  conversationId?: string | null;
  documents: WorkspaceDocument[];
  isLoading?: boolean;
  onDocumentsRefresh?: () => void;
  onInsertCommand?: (command: string) => void;
  onAttachmentUploaded?: (attachment: ConversationAttachment) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
}

export function WorkspaceSidebar({
  isOpen,
  onToggle,
  conversationId,
  documents,
  isLoading = false,
  onDocumentsRefresh,
  onInsertCommand,
  onAttachmentUploaded,
}: WorkspaceSidebarProps) {
  const { t } = useI18n();
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [processingAction, setProcessingAction] = useState<string | null>(null);
  const [editorDocId, setEditorDocId] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState("");
  const [editorHeight, setEditorHeight] = useState(220);
  const [uploadingFile, setUploadingFile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isDraggingRef = useRef(false);

  const getDocumentName = useCallback(
    (docId: string) => {
      return documents.find((d) => d.id === docId)?.filename || docId;
    },
    [documents],
  );

  const loadDocumentIntoEditor = useCallback(async (docId: string) => {
    setProcessingAction(docId);
    try {
      const response = await fetch(`/api/proxy/api/workspace/documents/${docId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "",
        },
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
  }, [t]);

  const handleEditorResizeStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    isDraggingRef.current = true;
    const startY = e.clientY;
    const startHeight = editorHeight;

    const onMouseMove = (moveEvent: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const delta = moveEvent.clientY - startY;
      const next = Math.min(420, Math.max(120, startHeight + delta));
      setEditorHeight(next);
    };

    const onMouseUp = () => {
      isDraggingRef.current = false;
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  }, [editorHeight]);

  const handleUpdateDocument = useCallback(async () => {
    if (!editorDocId || !editorContent.trim()) return;
    setProcessingAction(editorDocId);
    try {
      const docName = getDocumentName(editorDocId);
      const blob = new Blob([editorContent], {
        type: "text/plain;charset=utf-8",
      });
      const file = new File([blob], docName, { type: "text/plain" });

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`/api/proxy/api/workspace/documents/${editorDocId}`, {
        method: "PUT",
        headers: {
          "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "",
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to update document: ${response.statusText}`);
      }

      toast.success(t("workspace.saveChanges"), {
        description: `${docName} saved successfully`,
      });
      onDocumentsRefresh?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : t("workspace.saveChangesFailed");
      toast.error(t("workspace.saveFailed"), { description: message });
    } finally {
      setProcessingAction(null);
    }
  }, [editorDocId, editorContent, getDocumentName, onDocumentsRefresh, t]);

  const handleDeleteDocument = useCallback(async (docId: string) => {
    setProcessingAction(docId);
    try {
      const docName = getDocumentName(docId);
      const response = await fetch(`/api/proxy/api/workspace/documents/${docId}`, {
        method: "DELETE",
        headers: {
          "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to delete document: ${response.statusText}`);
      }

      if (editorDocId === docId) {
        setEditorDocId(null);
        setEditorContent("");
      }
      setExpandedDoc(null);

      toast.success(t("workspace.delete"), { description: docName });
      onDocumentsRefresh?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : t("workspace.deleteDocumentFailed");
      toast.error(t("workspace.deleteFailed"), { description: message });
    } finally {
      setProcessingAction(null);
    }
  }, [editorDocId, getDocumentName, onDocumentsRefresh, t]);

  const handleDownloadDocument = useCallback((docId: string) => {
    const docName = getDocumentName(docId);
    const url = `/api/proxy/api/workspace/documents/${docId}/download`;
    const a = document.createElement("a");
    a.href = url;
    a.download = docName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    toast.success(t("workspace.downloaded"), { description: docName });
  }, [getDocumentName, t]);

  const handleReattachEditorContent = useCallback(async () => {
    if (!editorDocId || !editorContent.trim()) return;
    const docName = getDocumentName(editorDocId);
    const file = new File([editorContent], docName, {
      type: docName.endsWith(".md") ? "text/markdown" : "text/plain",
    });
    
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
        description: err instanceof Error ? err.message : t("workspace.uploadFailed")
      });
    }
  }, [conversationId, editorDocId, editorContent, onAttachmentUploaded, getDocumentName, t]);

  const handleDownloadEditorContent = useCallback(() => {
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

  const handleUploadFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setUploadingFile(true);
      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/api/proxy/api/workspace/documents", {
          method: "POST",
          headers: {
            "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "",
          },
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }

        toast.success(t("workspace.documentUploaded"), { description: file.name });
        onDocumentsRefresh?.();
      } catch (err) {
        const message = err instanceof Error ? err.message : t("workspace.uploadFailed");
        toast.error(t("workspace.uploadFailed"), { description: message });
      } finally {
        setUploadingFile(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [onDocumentsRefresh, t]
  );

  const handleInsertLiveRefCommand = useCallback(
    (doc: WorkspaceDocument) => {
      onInsertCommand?.(
        t("workspace.commandReferenceTemplate", { filename: doc.filename, id: doc.id })
      );
      toast.success(t("workspace.commandAdded"), { description: doc.filename });
    },
    [onInsertCommand, t]
  );

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

        {/* Content area */}
        <div className="flex-1 overflow-y-auto flex flex-col">
          {/* Upload section */}
          <div className="px-3 py-3 border-b border-gray-100">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.markdown"
              onChange={handleUploadFile}
              disabled={uploadingFile || isLoading}
              className="hidden"
              aria-label="Upload document"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadingFile || isLoading}
              className="w-full px-3 py-2 rounded-lg border border-pacific-blue/40 bg-pacific-blue/5 text-pacific-blue hover:bg-pacific-blue/10 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Icon name="upload_file" size={16} />
              {uploadingFile ? t("workspace.uploading") : t("workspace.uploadDocument")}
            </button>
          </div>

          {/* Documents section */}
          {documents.length > 0 && (
            <div className="px-3 py-3 border-b border-gray-100">
              <p className="text-xs font-semibold text-evergreen/70 uppercase tracking-wide mb-2 px-1">
                {t("workspace.myDocuments", { count: documents.length })}
              </p>
              <div className="space-y-1.5">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="rounded-lg border border-gray-200 bg-gray-50 hover:bg-gray-100 transition-colors"
                  >
                    <button
                      onClick={() =>
                        setExpandedDoc(expandedDoc === doc.id ? null : doc.id)
                      }
                      className="w-full px-3 py-2 flex items-center justify-between text-left"
                    >
                      <div className="flex items-center gap-2 min-w-0 flex-1">
                        <Icon
                          name={
                            doc.filename.endsWith(".md")
                              ? "description"
                              : "text_snippet"
                          }
                          size={16}
                          className="text-evergreen/60 shrink-0"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-medium text-evergreen truncate">
                            {doc.filename}
                          </p>
                          <p className="text-xs text-evergreen/50">
                            {formatFileSize(doc.size_bytes)}
                            {doc.updated_at && (
                              <>
                                {" "}
                                • v{doc.version}
                              </>
                            )}
                          </p>
                        </div>
                      </div>
                      <Icon
                        name={expandedDoc === doc.id ? "expand_less" : "expand_more"}
                        size={16}
                        className="text-evergreen/40 shrink-0 ml-1"
                      />
                    </button>

                    {/* Expanded actions */}
                    {expandedDoc === doc.id && (
                      <div className="px-3 py-2 border-t border-gray-200 bg-white flex gap-1.5 flex-wrap">
                        <button
                          onClick={() => loadDocumentIntoEditor(doc.id)}
                          disabled={processingAction === doc.id || isLoading}
                          className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                     bg-pacific-blue/5 text-pacific-blue border border-pacific-blue/20
                                     hover:bg-pacific-blue/10 disabled:opacity-40 disabled:cursor-not-allowed
                                     transition-colors"
                          title={t("workspace.edit")}
                        >
                          <Icon name="edit" size={14} className="mr-1 inline" />
                          {t("workspace.edit")}
                        </button>
                        <button
                          onClick={() => handleInsertLiveRefCommand(doc)}
                          disabled={processingAction === doc.id || isLoading}
                          className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                     bg-evergreen/5 text-evergreen border border-evergreen/20
                                     hover:bg-evergreen/10 disabled:opacity-40 disabled:cursor-not-allowed
                                     transition-colors"
                          title={t("workspace.reference")}
                        >
                          <Icon name="chat" size={14} className="mr-1 inline" />
                          {t("workspace.reference")}
                        </button>
                        <button
                          onClick={() => handleDownloadDocument(doc.id)}
                          disabled={processingAction === doc.id || isLoading}
                          className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                     bg-burnt-tangerine/5 text-burnt-tangerine border border-burnt-tangerine/20
                                     hover:bg-burnt-tangerine/10 disabled:opacity-40 disabled:cursor-not-allowed
                                     transition-colors"
                          title={t("workspace.download")}
                        >
                          <Icon name="download" size={14} className="mr-1 inline" />
                          {t("workspace.download")}
                        </button>
                        <button
                          onClick={() => handleDeleteDocument(doc.id)}
                          disabled={processingAction === doc.id || isLoading}
                          className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                     bg-red-50 text-red-600 border border-red-200
                                     hover:bg-red-100 disabled:opacity-40 disabled:cursor-not-allowed
                                     transition-colors"
                          title={t("workspace.delete")}
                        >
                          <Icon name="delete" size={14} className="mr-1 inline" />
                          {t("workspace.delete")}
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Document editor */}
          {editorDocId && (
            <div className="px-3 py-3 border-t border-gray-100 bg-white">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-semibold text-evergreen/70 uppercase tracking-wide">
                  {t("workspace.editor")}
                </p>
                <button
                  type="button"
                  onClick={() => {
                    setEditorDocId(null);
                    setEditorContent("");
                  }}
                  className="text-evergreen/45 hover:text-evergreen"
                  aria-label={t("workspace.closeEditor")}
                >
                  <Icon name="close" size={14} />
                </button>
              </div>
              <p className="text-xs text-evergreen/50 mb-2 truncate">
                {getDocumentName(editorDocId)}
              </p>
              <div
                role="separator"
                aria-label={t("workspace.resizeEditor")}
                onMouseDown={handleEditorResizeStart}
                className="mb-2 h-2 cursor-row-resize flex items-center justify-center"
                title={t("workspace.resizeEditor")}
              >
                <div className="h-1 w-14 rounded-full bg-gray-300 hover:bg-gray-400" />
              </div>
              <textarea
                value={editorContent}
                onChange={(e) => setEditorContent(e.target.value)}
                style={{ height: `${editorHeight}px` }}
                className="w-full resize-none rounded-md border border-gray-300 px-2 py-2 text-xs text-evergreen focus:border-pacific-blue focus:ring-0"
                placeholder={t("workspace.editDocumentPlaceholder")}
              />
              <button
                type="button"
                onClick={handleUpdateDocument}
                disabled={!editorContent.trim() || isLoading || processingAction === editorDocId}
                className="mt-2 w-full rounded-md px-3 py-2 text-xs font-medium bg-burnt-tangerine text-white hover:bg-burnt-tangerine/90 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {processingAction === editorDocId ? t("common.saving") : t("workspace.saveChanges")}
              </button>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={handleReattachEditorContent}
                  disabled={!editorContent.trim() || isLoading}
                  className="flex-1 rounded-md px-3 py-2 text-xs font-medium bg-pacific-blue text-white hover:bg-pacific-blue/90 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {t("workspace.attachToChat")}
                </button>
                <button
                  type="button"
                  onClick={handleDownloadEditorContent}
                  disabled={!editorContent.trim()}
                  className="flex-1 rounded-md px-3 py-2 text-xs font-medium bg-evergreen/10 text-evergreen hover:bg-evergreen/15 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {t("workspace.download")}
                </button>
              </div>
            </div>
          )}

          {/* Empty state */}
          {documents.length === 0 && (
            <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
              <Icon
                name="folder_open"
                size={32}
                className="text-evergreen/20 mb-2"
              />
              <p className="text-xs font-medium text-evergreen/50 mb-1">
                {t("workspace.noDocuments")}
              </p>
              <p className="text-xs text-evergreen/40">
                {t("workspace.uploadToGetStarted")}
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
