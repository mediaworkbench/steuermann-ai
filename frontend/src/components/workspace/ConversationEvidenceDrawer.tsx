"use client";

import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { fetchConversation } from "@/lib/api";
import { toUiMessage } from "@/lib/messageMapping";
import { useI18n } from "@/hooks/useI18n";
import { Button } from "@/components/ui/button";
import type { Message } from "@/lib/types";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { WorkspaceEvidenceTabs } from "./WorkspaceEvidenceTabs";

type LoadState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; latest: Message | null };

/**
 * Read-only right-side drawer that surfaces the Knowledge / Memory / Outputs /
 * Inspector evidence of a past conversation's **latest assistant answer**.
 * Loaded from the persisted conversation (no backend change): `toUiMessage`
 * restores `metrics` + `nodeTrace`, which feed the shared `WorkspaceEvidenceTabs`.
 *
 * The parent mounts this only while a row is selected (so each open re-fetches
 * and replays the entrance), and unmounts it on close.
 */
export function ConversationEvidenceDrawer({
  conversationId,
  title,
  onClose,
}: {
  conversationId: string;
  title: string;
  onClose: () => void;
}) {
  const { t, formatTime } = useI18n();
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [reloadNonce, setReloadNonce] = useState(0);
  const [entered, setEntered] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Fetch the conversation, map to UI messages, and keep the latest assistant
  // one. `formatTime` is stable across renders (memoized on locale), so it is a
  // safe dependency and never causes a refetch loop.
  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    (async () => {
      const detail = await fetchConversation(conversationId);
      if (cancelled) return;
      if (!detail) {
        setState({ status: "error" });
        return;
      }
      const latest =
        [...detail.messages]
          .reverse()
          .map((m) => toUiMessage(m, formatTime))
          .find((m) => m.role === "assistant") ?? null;
      setState({ status: "ready", latest });
    })();
    return () => {
      cancelled = true;
    };
  }, [conversationId, reloadNonce, formatTime]);

  // Slide-in on mount; Escape dismisses; focus moves to the close control.
  useEffect(() => {
    const id = requestAnimationFrame(() => setEntered(true));
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    dialogRef.current?.focus();
    return () => {
      cancelAnimationFrame(id);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <>
      {/* Backdrop — mouse convenience only; aria-hidden so the dialog's own
          close button is the single AT-facing dismiss control. */}
      <button
        type="button"
        aria-hidden="true"
        tabIndex={-1}
        onClick={onClose}
        className={`fixed inset-0 z-40 bg-black/20 transition-opacity duration-200 ${
          entered ? "opacity-100" : "opacity-0"
        }`}
      />

      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={t("workspace.answerEvidence")}
        tabIndex={-1}
        className={`fixed right-0 top-0 z-50 flex h-screen w-full max-w-md flex-col border-l border-border
          bg-surface shadow-xl outline-none transition-transform duration-200
          ${entered ? "translate-x-0" : "translate-x-full"}`}
      >
        {/* Header */}
        <div className="shrink-0 border-b border-border px-4 py-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              {t("workspace.answerEvidence")}
            </p>
            <h2 className="truncate text-sm font-semibold text-foreground" title={title}>
              {title}
            </h2>
          </div>
          <Button
            type="button"
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="p-1.5 text-muted-foreground shrink-0"
            aria-label={t("chats.closeEvidence")}
          >
            <X size={18} />
          </Button>
        </div>

        {/* Body */}
        {state.status === "loading" && (
          <WorkspaceTabState
            icon="account_tree"
            title={t("chats.evidenceLoading")}
            tone="loading"
          />
        )}

        {state.status === "error" && (
          <WorkspaceTabState
            icon="account_tree"
            title={t("chats.evidenceError")}
            tone="error"
            action={{ label: t("workspace.retry"), onClick: () => setReloadNonce((n) => n + 1) }}
          />
        )}

        {state.status === "ready" && state.latest === null && (
          <WorkspaceTabState
            icon="account_tree"
            title={t("chats.evidenceEmpty")}
          />
        )}

        {state.status === "ready" && state.latest !== null && (
          <div className="flex min-h-0 flex-1 flex-col">
            <p className="shrink-0 px-4 py-2 text-[11px] text-muted-foreground border-b border-border/60">
              {t("chats.evidenceLatestHint")}
            </p>
            <WorkspaceEvidenceTabs
              metrics={state.latest.metrics}
              nodeTrace={state.latest.nodeTrace}
              className="flex-1"
            />
          </div>
        )}
      </div>
    </>
  );
}
