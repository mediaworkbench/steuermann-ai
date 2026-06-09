"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { createPortal } from "react-dom";
import Image from "next/image";
import { toast } from "sonner";
import { Check, ChevronDown, ChevronUp, Eraser, FileText, ScrollText, Search, Trash, Upload, X } from "lucide-react";
import {
  attachWorkspaceDocumentToConversation,
  clearAllWorkspaceDocuments,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { ConversationAttachment } from "@/lib/types";
import type { WorkspaceDocument } from "./types";
import { formatFileSize, workspaceAuthHeaders } from "./utils";
import { useActiveDocument } from "@/context/ActiveDocumentContext";
import { DocumentEditorView } from "./DocumentEditorView";
import { WorkspaceDocActionButton } from "./WorkspaceDocActionButton";
import { WorkspaceSectionLabel } from "./WorkspaceSectionLabel";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { VirtualizedList } from "./VirtualizedList";

/** Above this many (filtered) rows the list switches to windowed rendering. */
const VIRTUALIZE_THRESHOLD = 50;

interface DocumentsTabProps {
  conversationId?: string | null;
  documents: WorkspaceDocument[];
  isLoading?: boolean;
  onDocumentsRefresh?: () => void;
  onEnsureConversation?: () => Promise<string | null>;
  onAttachmentUploaded?: (attachment: ConversationAttachment) => void;
  documentsLoading?: boolean;
  documentsError?: string | null;
  onRetryDocuments?: () => void;
  /** When true the split-view pane owns the editor; the inline editor collapses. */
  splitViewActive?: boolean;
}

export function DocumentsTab({
  conversationId,
  documents,
  isLoading = false,
  onDocumentsRefresh,
  onEnsureConversation,
  onAttachmentUploaded,
  documentsLoading = false,
  documentsError = null,
  onRetryDocuments,
  splitViewActive = false,
}: DocumentsTabProps) {
  const { t } = useI18n();
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [nukePending, setNukePending] = useState(false);
  const [renamingDocId, setRenamingDocId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [lightboxDoc, setLightboxDoc] = useState<WorkspaceDocument | null>(null);
  const [mounted, setMounted] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Filter text is local. The Documents tab stays mounted across tab switches
  // (display:contents in WorkspacePanel), so it persists without leaking across
  // routes the way panel-context state would.
  const [query, setQuery] = useState("");

  const filteredDocuments = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return documents;
    return documents.filter((d) => d.filename.toLowerCase().includes(q));
  }, [documents, query]);

  // Editor + version-history + the shared processing token now live in the
  // ActiveDocumentProvider (one source of truth, shared with the split-view pane).
  const {
    editorDocId,
    closeEditor,
    getDocumentName,
    openEditor,
    loadHistory,
    processingAction,
    setProcessingAction,
  } = useActiveDocument();

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
        toast.success(t("workspace.renamed"), { description: trimmed });
        onDocumentsRefresh?.();
      } catch (err) {
        const message = err instanceof Error ? err.message : t("workspace.renameFailed");
        toast.error(t("workspace.renameFailed"), { description: message });
      } finally {
        setProcessingAction(null);
      }
    },
    [onDocumentsRefresh, t],
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

  const renderDocRow = (doc: WorkspaceDocument) => (
    <div
      key={doc.id}
      className="rounded-lg border border-border bg-surface-muted transition-colors hover:bg-surface-elevated"
    >
      <Button
        type="button"
        onClick={() => setExpandedDoc(expandedDoc === doc.id ? null : doc.id)}
        variant="ghost"
        size="sm"
        className="w-full justify-between px-3 py-2 text-left"
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {doc.mime_type?.startsWith("image/") ? (
            <div
              className="relative h-11 w-16 shrink-0 cursor-pointer overflow-hidden rounded bg-surface-elevated"
              onClick={(e) => {
                e.stopPropagation();
                setLightboxDoc(doc);
              }}
              title={t("workspace.thumbnailClickHint")}
            >
              <Image
                src={`/api/proxy/api/workspace/documents/${doc.id}/thumbnail`}
                alt={doc.filename}
                fill
                sizes="64px"
                unoptimized
                className="object-cover"
              />
              <span
                className="absolute bottom-0 left-0 right-0 text-center bg-foreground/55 text-background truncate px-0.5"
                style={{ fontSize: "8px", lineHeight: "13px" }}
              >
                {formatFileSize(doc.size_bytes)}
              </span>
            </div>
          ) : (
            doc.filename.endsWith(".md")
              ? <FileText size={16} className="shrink-0 text-muted-foreground" />
              : <ScrollText size={16} className="shrink-0 text-muted-foreground" />
          )}
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium text-foreground">{doc.filename}</p>
            <p className="text-xs text-muted-foreground">
              {formatFileSize(doc.size_bytes)}
              {doc.updated_at && !doc.mime_type?.startsWith("image/") && (
                <> • v{doc.version}</>
              )}
            </p>
          </div>
        </div>
        {expandedDoc === doc.id
          ? <ChevronUp size={16} className="ml-1 shrink-0 text-muted-foreground" />
          : <ChevronDown size={16} className="ml-1 shrink-0 text-muted-foreground" />
        }
      </Button>

      {/* Expanded actions */}
      {expandedDoc === doc.id && (
        <div className="flex flex-wrap gap-1.5 border-t border-border bg-surface px-3 py-2">
          {renamingDocId === doc.id ? (
            <div className="w-full flex gap-1.5">
              <Input
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
                className="h-8 flex-1 rounded border border-border bg-surface px-2 py-1 text-xs text-foreground shadow-none"
                placeholder={doc.filename}
              />
              <Button
                type="button"
                onClick={() => handleRenameDocument(doc.id, renameValue)}
                disabled={!renameValue.trim() || processingAction === doc.id}
                variant="primary"
                size="sm"
                className="px-2 py-1 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Check size={14} />
              </Button>
              <Button
                type="button"
                onClick={() => {
                  setRenamingDocId(null);
                  setRenameValue("");
                }}
                variant="secondary"
                size="sm"
                className="px-2 py-1 text-xs"
              >
                <X size={14} />
              </Button>
            </div>
          ) : null}
          {!doc.mime_type?.startsWith("image/") && (
            <WorkspaceDocActionButton
              onClick={() => openEditor(doc.id)}
              disabled={processingAction === doc.id || isLoading}
              variant="ghost"
              icon="edit"
              label={t("workspace.edit")}
              title={t("workspace.edit")}
            />
          )}
          <WorkspaceDocActionButton
            onClick={() => handleAttachFromWorkspace(doc)}
            disabled={processingAction === doc.id || isLoading}
            variant="ghost"
            icon="attach_file"
            label={t("workspace.attach")}
            title={t("workspace.attach")}
          />
          <WorkspaceDocActionButton
            onClick={() => handleDownloadDocument(doc.id)}
            disabled={processingAction === doc.id || isLoading}
            variant="ghost"
            icon="download"
            label={t("workspace.download")}
            title={t("workspace.download")}
          />
          {!doc.mime_type?.startsWith("image/") && (
            <WorkspaceDocActionButton
              onClick={() => loadHistory(doc.id)}
              disabled={processingAction === doc.id || isLoading}
              variant="secondary"
              icon="history"
              label={t("workspace.history")}
              title={t("workspace.versionHistoryTooltip")}
            />
          )}
          <WorkspaceDocActionButton
            onClick={() => {
              setRenamingDocId(doc.id);
              setRenameValue(doc.filename);
            }}
            disabled={processingAction === doc.id || isLoading}
            variant="secondary"
            icon="drive_file_rename_outline"
            label={t("workspace.rename")}
            title={t("workspace.rename")}
          />
          <WorkspaceDocActionButton
            onClick={() => handleDeleteDocument(doc.id)}
            disabled={processingAction === doc.id || isLoading}
            variant="destructive"
            icon="delete"
            label={t("workspace.delete")}
            title={t("workspace.delete")}
          />
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Upload section */}
      <div className="border-b border-border px-3 py-3">
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.markdown,.json,.yaml,.yml,.csv,.html,.xml,.jpg,.jpeg,.png,.gif,.webp"
          onChange={handleUploadFile}
          disabled={uploadingFile || isLoading}
          className="hidden"
          aria-label={t("workspace.uploadAriaLabel")}
        />
        <div className="flex gap-1.5">
          <Button
            type="button"
            onClick={() => {
              setNukePending(false);
              fileInputRef.current?.click();
            }}
            disabled={uploadingFile || isLoading}
            variant="secondary"
            size="sm"
            className="flex-2 text-xs font-medium"
          >
            <Upload size={16} />
            {uploadingFile ? t("workspace.uploading") : t("workspace.uploadDocument")}
          </Button>
          {documents.length > 0 &&
            (nukePending ? (
              <div className="flex-1 flex gap-1">
                <Button
                  type="button"
                  onClick={() => setNukePending(false)}
                  variant="secondary"
                  size="sm"
                  className="flex-1 px-2 py-2 text-xs"
                >
                  <X size={14} className="mx-auto" />
                </Button>
                <Button
                  type="button"
                  onClick={handleClearAllDocuments}
                  disabled={uploadingFile}
                  variant="destructive"
                  size="sm"
                  className="flex-1 px-2 py-2 text-xs font-medium"
                  title={t("workspace.nukeConfirm")}
                >
                  <Trash size={14} className="mx-auto" />
                </Button>
              </div>
            ) : (
              <Button
                type="button"
                onClick={() => setNukePending(true)}
                disabled={uploadingFile || isLoading}
                variant="destructive"
                size="sm"
                className="flex-1 px-2 py-2 text-xs font-medium"
                title={t("workspace.nukeAll")}
              >
                <Eraser size={16} />
              </Button>
            ))}
        </div>
      </div>

      {/* Search / filter */}
      {documents.length > 0 && (
        <div className="px-3 pt-3">
          <div className="relative">
            <Search
              size={15}
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground/50"
            />
            <Input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("workspace.searchDocuments")}
              aria-label={t("workspace.searchDocuments")}
              className="h-9 rounded-lg border-border bg-surface-muted py-1.5 pl-8 pr-7 text-xs text-foreground placeholder:text-muted-foreground/60 focus:bg-surface"
            />
            {query && (
              <Button
                type="button"
                onClick={() => setQuery("")}
                variant="ghost"
                size="sm"
                className="absolute right-2 top-1/2 -translate-y-1/2 p-0 text-muted-foreground hover:text-foreground"
                aria-label={t("workspace.clearSearch")}
              >
                <X size={14} />
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Documents section */}
      {documents.length > 0 && (
        <div className="border-b border-border px-3 py-3">
          <WorkspaceSectionLabel className="mb-2 text-xs tracking-wide">
            {t("workspace.myDocuments", { count: documents.length })}
          </WorkspaceSectionLabel>
          {filteredDocuments.length === 0 ? (
            <WorkspaceTabState
              icon="search_off"
              title={t("workspace.noResults")}
              hint={t("workspace.noResultsHint")}
            />
          ) : filteredDocuments.length > VIRTUALIZE_THRESHOLD ? (
            <VirtualizedList
              items={filteredDocuments}
              getKey={(doc) => doc.id}
              renderItem={renderDocRow}
              testId="virtualized-doc-list"
            />
          ) : (
          <div className="space-y-1.5">
            {filteredDocuments.map((doc) => renderDocRow(doc))}
          </div>
          )}
        </div>
      )}

      {/* Editor + version history. When split-view is on, the pane owns the
          editor instead (one shared instance, so they never both render). */}
      {!splitViewActive && <DocumentEditorView isLoading={isLoading} />}

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
              <Image
                src={`/api/proxy/api/workspace/documents/${lightboxDoc.id}/download`}
                alt={lightboxDoc.filename}
                width={1600}
                height={1200}
                unoptimized
                className="max-w-full max-h-[85vh] rounded-lg shadow-2xl object-contain"
              />
              <Button
                type="button"
                onClick={() => setLightboxDoc(null)}
                variant="secondary"
                size="sm"
                className="absolute -right-3 -top-3 h-8 w-8 rounded-full bg-surface p-0 shadow"
                aria-label={t("workspace.closeLightbox")}
              >
                <X size={16} className="text-foreground" />
              </Button>
              <p className="mt-2 text-center text-muted-foreground text-xs truncate">{lightboxDoc.filename}</p>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
