"use client";

import { useState, useCallback, useRef } from "react";
import { X } from "lucide-react";
import { useI18n } from "@/hooks/useI18n";
import { useActiveDocument } from "@/context/ActiveDocumentContext";
import { Button } from "@/components/ui/button";
import { DocumentEditorView } from "./DocumentEditorView";

interface ActiveDocumentPaneProps {
  /** Mirrors the chat-level loading gate for save/attach buttons. */
  isLoading?: boolean;
}

const MIN_WIDTH = 320;
const MAX_WIDTH = 720;
const DEFAULT_WIDTH = 460;

/**
 * First-class editing pane for the active workspace document, shown between the
 * chat column and the workspace panel when a document is open for editing. Owns
 * a horizontal resize on its left edge. On mobile it is a full-screen overlay.
 * Closing via the X calls closeEditor(), which clears editorDocId and causes the
 * parent to unmount this pane.
 */
export function ActiveDocumentPane({ isLoading = false }: ActiveDocumentPaneProps) {
  const { t } = useI18n();
  const { editorDocId, historyDocId, getDocumentName, closeEditor, closeHistory, isDirty } =
    useActiveDocument();
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const draggingRef = useRef(false);

  // Horizontal mirror of the old useDocumentEditor resize: dragging the left edge
  // leftwards (negative clientX delta) widens the pane.
  const onResizeStart = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      draggingRef.current = true;
      const startX = e.clientX;
      const startWidth = width;

      const onMouseMove = (moveEvent: MouseEvent) => {
        if (!draggingRef.current) return;
        const delta = startX - moveEvent.clientX;
        setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth + delta)));
      };
      const onMouseUp = () => {
        draggingRef.current = false;
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
      };

      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    },
    [width],
  );

  return (
    <div
      className="fixed inset-0 z-30 flex min-h-0 w-full flex-col border-l border-border bg-surface md:relative md:inset-auto md:z-auto md:w-(--pane-w) md:shrink-0"
      style={{ "--pane-w": `${width}px` } as React.CSSProperties}
      role="region"
      aria-label={t("workspace.splitViewTitle")}
    >
      {/* Left-edge resize handle (desktop only) */}
      <div
        aria-hidden="true"
        onMouseDown={onResizeStart}
        title={t("workspace.resizeSplitView")}
        className="absolute left-0 top-0 bottom-0 z-10 hidden w-1.5 -translate-x-1/2 cursor-col-resize md:block"
      >
        <div className="mx-auto h-full w-0.5 bg-border hover:bg-border-strong" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2 shrink-0">
        <p className="flex min-w-0 items-center gap-1.5 truncate text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <span className="truncate">
            {editorDocId
              ? getDocumentName(editorDocId)
              : historyDocId
                ? getDocumentName(historyDocId)
                : t("workspace.splitViewTitle")}
          </span>
          {isDirty && (
            <span
              className="shrink-0 text-warning"
              title={t("workspace.unsavedChanges")}
              aria-label={t("workspace.unsavedChanges")}
            >
              ●
            </span>
          )}
        </p>
        <Button
          type="button"
          onClick={() => {
            closeEditor();
            closeHistory();
          }}
          variant="ghost"
          size="sm"
          className="p-0 text-muted-foreground hover:text-foreground"
          aria-label={t("workspace.closeSplitView")}
          title={t("workspace.closeSplitView")}
        >
          <X size={16} />
        </Button>
      </div>

      {/* Body — guard against the brief null window during close batching */}
      {(editorDocId || historyDocId) && <DocumentEditorView isLoading={isLoading} />}
    </div>
  );
}

/**
 * Mount gate for the pane. Lives inside `ActiveDocumentProvider` so it can read
 * the editor/history state directly (ChatInterface sits above the provider and
 * cannot). Renders the pane whenever a document is open for editing OR its
 * version history is open; renders nothing otherwise. `ActiveDocumentPane`
 * itself keeps its mount contract (always renders its region when mounted).
 */
export function ActiveDocumentPaneSlot({ isLoading = false }: ActiveDocumentPaneProps) {
  const { editorDocId, historyDocId } = useActiveDocument();
  if (!editorDocId && !historyDocId) return null;
  return <ActiveDocumentPane isLoading={isLoading} />;
}
