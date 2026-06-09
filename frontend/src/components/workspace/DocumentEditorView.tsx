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
  /**
   * `inline` (Documents tab): fixed-height textarea + vertical resize handle.
   * `pane` (split-view): fills the pane height; the pane owns the horizontal resize.
   */
  variant?: "inline" | "pane";
}

/**
 * The document editor + version-history UI, driven entirely by
 * `useActiveDocument()`. Rendered inline in the Documents tab (when split-view is
 * off) and inside `ActiveDocumentPane` (when split-view is on) — both share the
 * single editor instance, so there are never two competing editors.
 */
export function DocumentEditorView({ isLoading = false, variant = "inline" }: DocumentEditorViewProps) {
  const { t } = useI18n();
  const {
    editorDocId,
    editorContent,
    setEditorContent,
    editorHeight,
    getDocumentName,
    closeEditor,
    onResizeStart,
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

  const pane = variant === "pane";

  const versionHistory = historyDocId ? (
    <WorkspacePanelSection className={pane ? "border-t-0" : undefined}>
      <WorkspacePanelHeaderRow
        title={t("workspace.versionHistoryTitle", { name: getDocumentName(historyDocId) })}
        onClose={closeHistory}
        closeLabel={t("common.close")}
      />
      {historyLoading ? (
        <p className="py-2 text-xs text-muted-foreground">{t("workspace.loadingVersions")}</p>
      ) : historyVersions.length === 0 ? (
        <p className="py-2 text-xs text-muted-foreground">{t("workspace.noSavedVersions")}</p>
      ) : (
        <div className="space-y-1.5">
          {historyVersions.map((v) => (
            <WorkspaceMutedCard key={v.id} className="rounded border">
              <div className="flex items-center justify-between px-2 py-1.5">
                <div>
                  <span className="text-xs font-medium text-foreground">v{v.version}</span>
                  <span className="ml-2 text-xs text-muted-foreground">
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
                    {previewContent.length > 2000 ? "\n…" : ""}
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
    <WorkspacePanelSection className={pane ? "flex min-h-0 flex-1 flex-col border-t-0" : undefined}>
      <WorkspacePanelHeaderRow
        title={t("workspace.editor")}
        onClose={closeEditor}
        closeLabel={t("workspace.closeEditor")}
      />
      <p className="mb-2 truncate text-xs text-muted-foreground">{getDocumentName(editorDocId)}</p>
      {!pane && (
        <div
          role="separator"
          aria-label={t("workspace.resizeEditor")}
          onMouseDown={onResizeStart}
          className="mb-2 h-2 cursor-row-resize flex items-center justify-center"
          title={t("workspace.resizeEditor")}
        >
          <div className="h-1 w-14 rounded-full bg-border hover:bg-border-strong" />
        </div>
      )}
      <Textarea
        value={editorContent}
        onChange={(e) => setEditorContent(e.target.value)}
        style={pane ? undefined : { height: `${editorHeight}px` }}
        className={`w-full resize-none rounded-md border border-border px-2 py-2 text-xs text-foreground shadow-none focus:border-primary focus:ring-0${
          pane ? " min-h-0 flex-1" : ""
        }`}
        placeholder={t("workspace.editDocumentPlaceholder")}
      />
      <Button
        type="button"
        onClick={saveEditor}
        disabled={!editorContent.trim() || isLoading || processingAction === editorDocId}
        variant="primary"
        size="sm"
        className="mt-2 w-full rounded-md px-3 py-2 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40"
      >
        {processingAction === editorDocId ? t("common.saving") : t("workspace.saveChanges")}
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

  if (pane) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        {versionHistory}
        {editor}
      </div>
    );
  }

  return (
    <>
      {versionHistory}
      {editor}
    </>
  );
}
