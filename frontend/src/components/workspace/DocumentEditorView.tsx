"use client";

import { useI18n } from "@/hooks/useI18n";
import { useActiveDocument } from "@/context/ActiveDocumentContext";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { formatFileSize } from "./utils";
import { WorkspaceMutedCard } from "./WorkspaceMutedCard";
import { WorkspacePanelHeaderRow } from "./WorkspacePanelHeaderRow";
import { WorkspacePanelSection } from "./WorkspacePanelSection";

interface DocumentEditorViewProps {
  /** Disables save/attach while the parent is busy (mirrors the old inline gate). */
  isLoading?: boolean;
}

/**
 * The document editor + version-history UI, driven entirely by
 * `useActiveDocument()`. Rendered inside `ActiveDocumentPane` — the single
 * editing surface. The pane header owns the doc name and close control.
 */
export function DocumentEditorView({ isLoading = false }: DocumentEditorViewProps) {
  const { t } = useI18n();
  const {
    editorDocId,
    editorContent,
    setEditorContent,
    isDirty,
    getDocumentName,
    getDocument,
    saveEditor,
    reattachEditor,
    downloadEditor,
    processingAction,
    historyDocId,
    historyVersions,
    historyLoading,
    previewVersionId,
    previewContent,
    previewVersion,
    restoreVersion,
    closeHistory,
  } = useActiveDocument();

  const versionSourceLabel = (source?: string | null): string | null => {
    if (source === "user") return t("workspace.versionSourceUser");
    if (source === "assistant") return t("workspace.versionSourceAssistant");
    if (source === "restore") return t("workspace.versionSourceRestored");
    return null;
  };

  // The current/latest version lives only on the document record — it is never a
  // historical snapshot (snapshots capture the *prior* content on each change).
  // Surface it as a non-restorable "Current" row so the version shown in the
  // Documents list (e.g. just after a restore) is always reflected in history.
  const currentDoc = historyDocId ? getDocument(historyDocId) : undefined;
  const pastVersions = currentDoc
    ? historyVersions.filter((v) => v.version !== currentDoc.version)
    : historyVersions;

  const versionHistory = historyDocId ? (
    <WorkspacePanelSection className="border-t-0">
      <WorkspacePanelHeaderRow
        title={t("workspace.versionHistoryTitle", { name: getDocumentName(historyDocId) })}
        onClose={closeHistory}
        closeLabel={t("common.close")}
      />
      {historyLoading ? (
        <p className="py-2 text-xs text-muted-foreground">{t("workspace.loadingVersions")}</p>
      ) : !currentDoc && pastVersions.length === 0 ? (
        <p className="py-2 text-xs text-muted-foreground">{t("workspace.noSavedVersions")}</p>
      ) : (
        <div className="space-y-1.5">
          {currentDoc && (
            <WorkspaceMutedCard className="rounded border border-primary/40">
              <div className="flex items-center justify-between px-2 py-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium text-foreground">v{currentDoc.version}</span>
                  <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                    {t("workspace.versionCurrent")}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(currentDoc.size_bytes)}
                    {currentDoc.updated_at &&
                      ` · ${new Date(currentDoc.updated_at).toLocaleDateString()}`}
                  </span>
                </div>
              </div>
            </WorkspaceMutedCard>
          )}
          {pastVersions.map((v) => (
            <WorkspaceMutedCard key={v.id} className="rounded border">
              <div className="flex items-center justify-between px-2 py-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium text-foreground">v{v.version}</span>
                  {versionSourceLabel(v.source) && (
                    <span className="rounded-full bg-surface-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                      {versionSourceLabel(v.source)}
                    </span>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(v.size_bytes)}
                    {v.created_at && ` · ${new Date(v.created_at).toLocaleDateString()}`}
                  </span>
                </div>
                <div className="flex gap-1">
                  <Button
                    type="button"
                    onClick={() => previewVersion(historyDocId, v.version, v.id)}
                    variant="secondary"
                    size="sm"
                    className="px-1.5 py-0.5 text-xs"
                  >
                    {previewVersionId === v.id ? t("workspace.hidePreview") : t("workspace.preview")}
                  </Button>
                  <Button
                    type="button"
                    onClick={() => restoreVersion(historyDocId, v.version)}
                    disabled={processingAction === `restore-${historyDocId}-${v.version}`}
                    variant="ghost"
                    size="sm"
                    className="px-1.5 py-0.5 text-xs text-primary hover:bg-primary/10 disabled:opacity-40"
                  >
                    {t("workspace.restore")}
                  </Button>
                </div>
              </div>
              {previewVersionId === v.id && previewContent && (
                <div className="px-2 pb-2">
                  <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded border border-border bg-surface p-2 font-mono text-xs">
                    {previewContent.slice(0, 2000)}
                    {previewContent.length > 2000 ? `\n… ${t("workspace.previewTruncated")}` : ""}
                  </pre>
                </div>
              )}
            </WorkspaceMutedCard>
          ))}
        </div>
      )}
    </WorkspacePanelSection>
  ) : null;

  const editor = editorDocId ? (
    <WorkspacePanelSection className="flex min-h-0 flex-1 flex-col border-t-0">
      <Textarea
        value={editorContent}
        onChange={(e) => setEditorContent(e.target.value)}
        className="w-full min-h-0 flex-1 resize-none rounded-md border border-border px-2 py-2 text-xs text-foreground shadow-none focus:border-primary focus:ring-0"
        placeholder={t("workspace.editDocumentPlaceholder")}
      />
      <Button
        type="button"
        onClick={saveEditor}
        disabled={!editorContent.trim() || !isDirty || isLoading || processingAction === editorDocId}
        variant="primary"
        size="sm"
        className="mt-2 w-full rounded-md px-3 py-2 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
      >
        {processingAction === editorDocId
          ? t("common.saving")
          : isDirty
            ? `${t("workspace.saveChanges")} • ${t("workspace.unsavedChanges")}`
            : t("workspace.saveChanges")}
      </Button>
      <div className="mt-2 flex gap-2">
        <Button
          type="button"
          onClick={reattachEditor}
          disabled={!editorContent.trim() || isLoading}
          variant="primary"
          size="sm"
          className="flex-1 rounded-md px-3 py-2 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
        >
          {t("workspace.attachToChat")}
        </Button>
        <Button
          type="button"
          onClick={downloadEditor}
          disabled={!editorContent.trim()}
          variant="secondary"
          size="sm"
          className="flex-1 rounded-md px-3 py-2 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
        >
          {t("workspace.download")}
        </Button>
      </div>
    </WorkspacePanelSection>
  ) : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {versionHistory}
      {editor}
    </div>
  );
}
