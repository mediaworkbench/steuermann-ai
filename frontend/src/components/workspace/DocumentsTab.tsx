"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { createPortal } from "react-dom";
import { toast } from "sonner";
import { Icon } from "../Icon";
import {
  attachWorkspaceDocumentToConversation,
  clearAllWorkspaceDocuments,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import type { ConversationAttachment } from "@/lib/types";
import type { WorkspaceDocument } from "./types";
import { formatFileSize, workspaceAuthHeaders } from "./utils";
import { useDocumentEditor } from "./useDocumentEditor";
import { useVersionHistory } from "./useVersionHistory";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { useWorkspacePanel } from "@/context/WorkspacePanelContext";

interface DocumentsTabProps {
  conversationId?: string | null;
  documents: WorkspaceDocument[];
  isLoading?: boolean;
  onDocumentsRefresh?: () => void;
  onEnsureConversation?: () => Promise<string | null>;
  onAttachmentUploaded?: (attachment: ConversationAttachment) => void;
  writebackSavedDocId?: string | null;
  onActiveDocumentChange?: (docId: string | null) => void;
  documentsLoading?: boolean;
  documentsError?: string | null;
  onRetryDocuments?: () => void;
}

export function DocumentsTab({
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
}: DocumentsTabProps) {
  const { t } = useI18n();
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [processingAction, setProcessingAction] = useState<string | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [nukePending, setNukePending] = useState(false);
  const [renamingDocId, setRenamingDocId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [lightboxDoc, setLightboxDoc] = useState<WorkspaceDocument | null>(null);
  const [mounted, setMounted] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Filter text lives in the panel context so it survives tab switches.
  const { documentQuery: query, setDocumentQuery: setQuery } = useWorkspacePanel();

  const filteredDocuments = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return documents;
    return documents.filter((d) => d.filename.toLowerCase().includes(q));
  }, [documents, query]);

  const {
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
  } = useDocumentEditor({
    documents,
    conversationId,
    writebackSavedDocId,
    onActiveDocumentChange,
    onAttachmentUploaded,
    onDocumentsRefresh,
    setProcessingAction,
  });

  const {
    historyDocId,
    historyVersions,
    historyLoading,
    previewVersionId,
    previewContent,
    loadHistory,
    previewVersion,
    restoreVersion,
    closeHistory,
  } = useVersionHistory({
    onDocumentsRefresh,
    setProcessingAction,
    onAfterRestore: (docId) => {
      if (editorDocId === docId) openEditor(docId);
    },
  });

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!lightboxDoc) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightboxDoc(null);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [lightboxDoc]);

  const handleDeleteDocument = useCallback(
    async (docId: string) => {
      setProcessingAction(docId);
      try {
        const docName = getDocumentName(docId);
        const response = await fetch(`/api/proxy/api/workspace/documents/${docId}`, {
          method: "DELETE",
          headers: workspaceAuthHeaders(),
        });
        if (!response.ok) {
          throw new Error(`Failed to delete document: ${response.statusText}`);
        }
        if (editorDocId === docId) closeEditor();
        setExpandedDoc(null);
        toast.success(t("workspace.delete"), { description: docName });
        onDocumentsRefresh?.();
      } catch (err) {
        const message = err instanceof Error ? err.message : t("workspace.deleteDocumentFailed");
        toast.error(t("workspace.deleteFailed"), { description: message });
      } finally {
        setProcessingAction(null);
      }
    },
    [editorDocId, closeEditor, getDocumentName, onDocumentsRefresh, t],
  );

  const handleRenameDocument = useCallback(
    async (docId: string, newFilename: string) => {
      const trimmed = newFilename.trim();
      if (!trimmed) return;
      setProcessingAction(docId);
      try {
        const response = await fetch(`/api/proxy/api/workspace/documents/${docId}`, {
          method: "PATCH",
          headers: workspaceAuthHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({ filename: trimmed }),
        });
        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || `Rename failed: ${response.statusText}`);
        }
        setRenamingDocId(null);
        setRenameValue("");
        toast.success("Renamed", { description: trimmed });
        onDocumentsRefresh?.();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Rename failed";
        toast.error("Rename failed", { description: message });
      } finally {
        setProcessingAction(null);
      }
    },
    [onDocumentsRefresh],
  );

  const handleDownloadDocument = useCallback(
    (docId: string) => {
      const docName = getDocumentName(docId);
      const url = `/api/proxy/api/workspace/documents/${docId}/download`;
      const a = document.createElement("a");
      a.href = url;
      a.download = docName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      toast.success(t("workspace.downloaded"), { description: docName });
    },
    [getDocumentName, t],
  );

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
          headers: workspaceAuthHeaders(),
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
    [onDocumentsRefresh, t],
  );

  const handleAttachFromWorkspace = useCallback(
    async (doc: WorkspaceDocument) => {
      const convId = conversationId ?? (onEnsureConversation ? await onEnsureConversation() : null);
      if (!convId) {
        toast.error(t("workspace.attachFailed"));
        return;
      }
      setProcessingAction(doc.id);
      try {
        const attachment = await attachWorkspaceDocumentToConversation(convId, doc.id);
        if (attachment) {
          onAttachmentUploaded?.(attachment);
          toast.success(t("workspace.attachSuccess"), { description: doc.filename });
        } else {
          toast.error(t("workspace.attachFailed"));
        }
      } finally {
        setProcessingAction(null);
      }
    },
    [conversationId, onEnsureConversation, onAttachmentUploaded, t],
  );

  const handleClearAllDocuments = useCallback(async () => {
    setNukePending(false);
    setUploadingFile(true);
    try {
      const count = await clearAllWorkspaceDocuments();
      toast.success(t("workspace.nukeSuccess", { count }));
      onDocumentsRefresh?.();
    } catch {
      toast.error(t("workspace.nukeFailed"));
    } finally {
      setUploadingFile(false);
    }
  }, [onDocumentsRefresh, t]);

  return (
    <>
      {/* Upload section */}
      <div className="px-3 py-3 border-b border-gray-100">
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.markdown,.json,.yaml,.yml,.csv,.html,.xml,.jpg,.jpeg,.png,.gif,.webp"
          onChange={handleUploadFile}
          disabled={uploadingFile || isLoading}
          className="hidden"
          aria-label="Upload document or image"
        />
        <div className="flex gap-1.5">
          <button
            onClick={() => {
              setNukePending(false);
              fileInputRef.current?.click();
            }}
            disabled={uploadingFile || isLoading}
            className="flex-[2] px-3 py-2 rounded-lg border border-pacific-blue/40 bg-pacific-blue/5 text-pacific-blue hover:bg-pacific-blue/10 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium transition-colors flex items-center justify-center gap-2"
          >
            <Icon name="upload_file" size={16} />
            {uploadingFile ? t("workspace.uploading") : t("workspace.uploadDocument")}
          </button>
          {documents.length > 0 &&
            (nukePending ? (
              <div className="flex-1 flex gap-1">
                <button
                  onClick={() => setNukePending(false)}
                  className="flex-1 px-2 py-2 rounded-lg border border-gray-300 bg-gray-50 text-gray-600 hover:bg-gray-100 text-xs font-medium transition-colors"
                >
                  <Icon name="close" size={14} className="mx-auto" />
                </button>
                <button
                  onClick={handleClearAllDocuments}
                  disabled={uploadingFile}
                  className="flex-1 px-2 py-2 rounded-lg border border-red-300 bg-red-50 text-red-600 hover:bg-red-100 disabled:opacity-40 text-xs font-medium transition-colors"
                  title={t("workspace.nukeConfirm")}
                >
                  <Icon name="delete_forever" size={14} className="mx-auto" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setNukePending(true)}
                disabled={uploadingFile || isLoading}
                className="flex-1 px-2 py-2 rounded-lg border border-red-200 bg-red-50 text-red-500 hover:bg-red-100 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium transition-colors flex items-center justify-center gap-1"
                title={t("workspace.nukeAll")}
              >
                <Icon name="delete_sweep" size={16} />
              </button>
            ))}
        </div>
      </div>

      {/* Search / filter */}
      {documents.length > 0 && (
        <div className="px-3 pt-3">
          <div className="relative">
            <Icon
              name="search"
              size={15}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-evergreen/40 pointer-events-none"
            />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("workspace.searchDocuments")}
              aria-label={t("workspace.searchDocuments")}
              className="w-full pl-8 pr-7 py-1.5 text-xs rounded-lg border border-gray-200 bg-gray-50 text-evergreen placeholder:text-evergreen/40 focus:bg-white focus:border-pacific-blue focus:outline-none transition-colors"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-evergreen/40 hover:text-evergreen"
                aria-label={t("workspace.clearSearch")}
              >
                <Icon name="close" size={14} />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Documents section */}
      {documents.length > 0 && (
        <div className="px-3 py-3 border-b border-gray-100">
          <p className="text-xs font-semibold text-evergreen/70 uppercase tracking-wide mb-2 px-1">
            {t("workspace.myDocuments", { count: documents.length })}
          </p>
          {filteredDocuments.length === 0 ? (
            <WorkspaceTabState
              icon="search_off"
              title={t("workspace.noResults")}
              hint={t("workspace.noResultsHint")}
            />
          ) : (
          <div className="space-y-1.5">
            {filteredDocuments.map((doc) => (
              <div
                key={doc.id}
                className="rounded-lg border border-gray-200 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <button
                  onClick={() => setExpandedDoc(expandedDoc === doc.id ? null : doc.id)}
                  className="w-full px-3 py-2 flex items-center justify-between text-left"
                >
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {doc.mime_type?.startsWith("image/") ? (
                      <div
                        className="relative w-16 h-11 shrink-0 rounded overflow-hidden bg-gray-200 cursor-pointer"
                        onClick={(e) => {
                          e.stopPropagation();
                          setLightboxDoc(doc);
                        }}
                        title={t("workspace.thumbnailClickHint")}
                      >
                        <img
                          src={`/api/proxy/api/workspace/documents/${doc.id}/thumbnail`}
                          alt={doc.filename}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                        <span
                          className="absolute bottom-0 left-0 right-0 text-center bg-black/50 text-white truncate px-0.5"
                          style={{ fontSize: "8px", lineHeight: "13px" }}
                        >
                          {formatFileSize(doc.size_bytes)}
                        </span>
                      </div>
                    ) : (
                      <Icon
                        name={doc.filename.endsWith(".md") ? "description" : "text_snippet"}
                        size={16}
                        className="text-evergreen/60 shrink-0"
                      />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-medium text-evergreen truncate">{doc.filename}</p>
                      <p className="text-xs text-evergreen/50">
                        {formatFileSize(doc.size_bytes)}
                        {doc.updated_at && !doc.mime_type?.startsWith("image/") && (
                          <> • v{doc.version}</>
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
                    {renamingDocId === doc.id ? (
                      <div className="w-full flex gap-1.5">
                        <input
                          autoFocus
                          type="text"
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleRenameDocument(doc.id, renameValue);
                            if (e.key === "Escape") {
                              setRenamingDocId(null);
                              setRenameValue("");
                            }
                          }}
                          className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-pacific-blue"
                          placeholder={doc.filename}
                        />
                        <button
                          onClick={() => handleRenameDocument(doc.id, renameValue)}
                          disabled={!renameValue.trim() || processingAction === doc.id}
                          className="px-2 py-1 rounded text-xs font-medium bg-pacific-blue text-white
                                     hover:bg-pacific-blue/80 disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          <Icon name="check" size={14} />
                        </button>
                        <button
                          onClick={() => {
                            setRenamingDocId(null);
                            setRenameValue("");
                          }}
                          className="px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-600
                                     hover:bg-gray-200"
                        >
                          <Icon name="close" size={14} />
                        </button>
                      </div>
                    ) : null}
                    {!doc.mime_type?.startsWith("image/") && (
                      <button
                        onClick={() => openEditor(doc.id)}
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
                    )}
                    <button
                      onClick={() => handleAttachFromWorkspace(doc)}
                      disabled={processingAction === doc.id || isLoading}
                      className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                 bg-pacific-blue/5 text-pacific-blue border border-pacific-blue/20
                                 hover:bg-pacific-blue/10 disabled:opacity-40 disabled:cursor-not-allowed
                                 transition-colors"
                      title={t("workspace.attach")}
                    >
                      <Icon name="attach_file" size={14} className="mr-1 inline" />
                      {t("workspace.attach")}
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
                    {!doc.mime_type?.startsWith("image/") && (
                      <button
                        onClick={() => loadHistory(doc.id)}
                        disabled={processingAction === doc.id || isLoading}
                        className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                   bg-gray-50 text-gray-600 border border-gray-200
                                   hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed
                                   transition-colors"
                        title="Version history"
                      >
                        <Icon name="history" size={14} className="mr-1 inline" />
                        History
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setRenamingDocId(doc.id);
                        setRenameValue(doc.filename);
                      }}
                      disabled={processingAction === doc.id || isLoading}
                      className="flex-1 min-w-fit px-2.5 py-1.5 rounded text-xs font-medium
                                 bg-gray-50 text-gray-600 border border-gray-200
                                 hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed
                                 transition-colors"
                      title="Rename"
                    >
                      <Icon name="drive_file_rename_outline" size={14} className="mr-1 inline" />
                      Rename
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
          )}
        </div>
      )}

      {/* Version history panel */}
      {historyDocId && (
        <div className="px-3 py-3 border-t border-gray-100 bg-white">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-evergreen/70 uppercase tracking-wide">
              Version History — {getDocumentName(historyDocId)}
            </p>
            <button onClick={closeHistory} className="text-evergreen/45 hover:text-evergreen">
              <Icon name="close" size={14} />
            </button>
          </div>
          {historyLoading ? (
            <p className="text-xs text-evergreen/50 py-2">Loading…</p>
          ) : historyVersions.length === 0 ? (
            <p className="text-xs text-evergreen/50 py-2">No saved versions yet.</p>
          ) : (
            <div className="space-y-1.5">
              {historyVersions.map((v) => (
                <div key={v.id} className="rounded border border-gray-200 bg-gray-50">
                  <div className="flex items-center justify-between px-2 py-1.5">
                    <div>
                      <span className="text-xs font-medium text-evergreen">v{v.version}</span>
                      <span className="text-xs text-evergreen/50 ml-2">
                        {formatFileSize(v.size_bytes)}
                        {v.created_at && ` · ${new Date(v.created_at).toLocaleDateString()}`}
                      </span>
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={() => previewVersion(historyDocId, v.version, v.id)}
                        className="px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-600 hover:bg-gray-200"
                      >
                        {previewVersionId === v.id ? "Hide" : "Preview"}
                      </button>
                      <button
                        onClick={() => restoreVersion(historyDocId, v.version)}
                        disabled={processingAction === `restore-${historyDocId}-${v.version}`}
                        className="px-1.5 py-0.5 rounded text-xs bg-pacific-blue/10 text-pacific-blue hover:bg-pacific-blue/20 disabled:opacity-40"
                      >
                        Restore
                      </button>
                    </div>
                  </div>
                  {previewVersionId === v.id && previewContent && (
                    <div className="px-2 pb-2">
                      <pre className="text-xs bg-white border border-gray-200 rounded p-2 max-h-40 overflow-y-auto whitespace-pre-wrap font-mono">
                        {previewContent.slice(0, 2000)}
                        {previewContent.length > 2000 ? "\n…" : ""}
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
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
              onClick={closeEditor}
              className="text-evergreen/45 hover:text-evergreen"
              aria-label={t("workspace.closeEditor")}
            >
              <Icon name="close" size={14} />
            </button>
          </div>
          <p className="text-xs text-evergreen/50 mb-2 truncate">{getDocumentName(editorDocId)}</p>
          <div
            role="separator"
            aria-label={t("workspace.resizeEditor")}
            onMouseDown={onResizeStart}
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
            onClick={saveEditor}
            disabled={!editorContent.trim() || isLoading || processingAction === editorDocId}
            className="mt-2 w-full rounded-md px-3 py-2 text-xs font-medium bg-burnt-tangerine text-white hover:bg-burnt-tangerine/90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {processingAction === editorDocId ? t("common.saving") : t("workspace.saveChanges")}
          </button>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={reattachEditor}
              disabled={!editorContent.trim() || isLoading}
              className="flex-1 rounded-md px-3 py-2 text-xs font-medium bg-pacific-blue text-white hover:bg-pacific-blue/90 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {t("workspace.attachToChat")}
            </button>
            <button
              type="button"
              onClick={downloadEditor}
              disabled={!editorContent.trim()}
              className="flex-1 rounded-md px-3 py-2 text-xs font-medium bg-evergreen/10 text-evergreen hover:bg-evergreen/15 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {t("workspace.download")}
            </button>
          </div>
        </div>
      )}

      {/* Empty / loading / error state (only when there is nothing to list) */}
      {documents.length === 0 &&
        (documentsError ? (
          <WorkspaceTabState
            tone="error"
            icon="error"
            title={t("workspace.documentsLoadError")}
            hint={documentsError}
            action={
              onRetryDocuments ? { label: t("workspace.retry"), onClick: onRetryDocuments } : undefined
            }
          />
        ) : documentsLoading ? (
          <WorkspaceTabState
            tone="loading"
            icon="folder_open"
            title={t("workspace.loadingDocuments")}
          />
        ) : (
          <WorkspaceTabState
            icon="folder_open"
            title={t("workspace.noDocuments")}
            hint={t("workspace.uploadToGetStarted")}
          />
        ))}

      {/* Image lightbox — portaled to body so it is not confined by the panel's
          translate transform (which would clip a position: fixed overlay). */}
      {mounted &&
        lightboxDoc &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
            onClick={() => setLightboxDoc(null)}
            role="dialog"
            aria-modal
            aria-label={lightboxDoc.filename}
          >
            <div className="relative max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
              <img
                src={`/api/proxy/api/workspace/documents/${lightboxDoc.id}/download`}
                alt={lightboxDoc.filename}
                className="max-w-full max-h-[85vh] rounded-lg shadow-2xl object-contain"
              />
              <button
                onClick={() => setLightboxDoc(null)}
                className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-white shadow flex items-center justify-center"
                aria-label="Close"
              >
                <Icon name="close" size={16} className="text-gray-700" />
              </button>
              <p className="mt-2 text-center text-white/70 text-xs truncate">{lightboxDoc.filename}</p>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
