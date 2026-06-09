"use client";

import { useState, useCallback, useRef } from "react";
import { X } from "lucide-react";
import { useI18n } from "@/hooks/useI18n";
import { useActiveDocument } from "@/context/ActiveDocumentContext";
import { Button } from "@/components/ui/button";
import { DocumentEditorView } from "./DocumentEditorView";

interface ActiveDocumentPaneProps {
  /** Closes the pane (turns split-view off). */
  onClose: () => void;
  /** Mirrors the chat-level loading gate for save/attach buttons. */
  isLoading?: boolean;
}

const MIN_WIDTH = 320;
const MAX_WIDTH = 720;
const DEFAULT_WIDTH = 460;

/**
 * First-class editing pane for the active workspace document, shown between the
 * chat column and the workspace panel when split-view is on. It renders the
 * shared `DocumentEditorView` (so it edits the same single editor instance as the
 * Documents tab) and owns a horizontal resize on its left edge. On mobile it is a
 * full-screen overlay rather than a side-by-side column.
 */
export function ActiveDocumentPane({ onClose, isLoading = false }: ActiveDocumentPaneProps) {
  const { t } = useI18n();
  const { editorDocId, getDocumentName } = useActiveDocument();
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const draggingRef = useRef(false);

  // Horizontal mirror of useDocumentEditor.onResizeStart: dragging the left edge
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
        role="separator"
        aria-orientation="vertical"
        aria-label={t("workspace.resizeSplitView")}
        onMouseDown={onResizeStart}
        title={t("workspace.resizeSplitView")}
        className="absolute left-0 top-0 bottom-0 z-10 hidden w-1.5 -translate-x-1/2 cursor-col-resize md:block"
      >
        <div className="mx-auto h-full w-0.5 bg-border hover:bg-border-strong" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-2 shrink-0">
        <p className="truncate text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {editorDocId ? getDocumentName(editorDocId) : t("workspace.splitViewTitle")}
        </p>
        <Button
          type="button"
          onClick={onClose}
          variant="ghost"
          size="sm"
          className="p-0 text-muted-foreground hover:text-foreground"
          aria-label={t("workspace.closeSplitView")}
          title={t("workspace.closeSplitView")}
        >
          <X size={16} />
        </Button>
      </div>

      {/* Body */}
      {editorDocId ? (
        <DocumentEditorView variant="pane" isLoading={isLoading} />
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
          <p className="text-sm font-medium text-foreground">{t("workspace.splitViewEmpty")}</p>
          <p className="mt-1 text-xs text-muted-foreground">{t("workspace.splitViewEmptyHint")}</p>
        </div>
      )}
    </div>
  );
}
