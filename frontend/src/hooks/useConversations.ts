"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { toast } from "sonner";
import type { Conversation } from "@/lib/types";
import {
  createConversation,
  fetchConversations,
  updateConversation,
  deleteConversation,
  exportConversation,
} from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";

const USER_ID = CURRENT_USER_ID;

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const initialLoadDone = useRef(false);

  // ── Load conversations ─────────────────────────────────────────────
  // Fetch up to the backend max (200); the sidebar shows pinned + 5 recent,
  // the /chats page paginates over the full loaded set.

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchConversations(USER_ID, 200, 0);
      if (data) {
        setConversations(data.conversations);
        setTotal(data.total);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!initialLoadDone.current) {
      initialLoadDone.current = true;
      refresh();
    }
  }, [refresh]);

  // ── Create ─────────────────────────────────────────────────────────

  const create = useCallback(
    async (title?: string, language?: string) => {
      const conv = await createConversation(USER_ID, title || "New conversation", language || "en");
      if (conv) {
        setConversations((prev) => [conv, ...prev]);
        setTotal((t) => t + 1);
        setActiveId(conv.id);
      }
      return conv;
    },
    [],
  );

  // ── Update (rename, pin, language) ─────────────────────────────────

  const update = useCallback(
    async (
      id: string,
      updates: { title?: string; pinned?: boolean; language?: string },
    ) => {
      const conv = await updateConversation(id, updates);
      if (conv) {
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? conv : c)),
        );
      }
      return conv;
    },
    [],
  );

  // ── Delete ─────────────────────────────────────────────────────────

  const remove = useCallback(
    async (id: string) => {
      const ok = await deleteConversation(id);
      if (ok) {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        setTotal((t) => Math.max(0, t - 1));
        if (activeId === id) setActiveId(null);
        toast.success("Conversation deleted");
      } else {
        toast.error("Failed to delete conversation");
      }
      return ok;
    },
    [activeId],
  );

  // ── Rename (convenience wrapper) ──────────────────────────────────

  const rename = useCallback(
    async (id: string, title: string) => {
      return update(id, { title });
    },
    [update],
  );

  // ── Bulk delete ───────────────────────────────────────────────────

  const bulkDelete = useCallback(
    async (ids: string[]) => {
      await Promise.all(ids.map((id) => deleteConversation(id)));
      setConversations((prev) => prev.filter((c) => !ids.includes(c.id)));
      setTotal((t) => Math.max(0, t - ids.length));
      if (activeId && ids.includes(activeId)) setActiveId(null);
      toast.success(`Deleted ${ids.length} conversation${ids.length > 1 ? "s" : ""}`);
    },
    [activeId],
  );

  // ── Bulk pin / unpin ──────────────────────────────────────────────

  const bulkPin = useCallback(
    async (ids: string[], pinned: boolean) => {
      await Promise.all(ids.map((id) => updateConversation(id, { pinned })));
      setConversations((prev) =>
        prev.map((c) => (ids.includes(c.id) ? { ...c, pinned } : c)),
      );
      toast.success(
        `${pinned ? "Pinned" : "Unpinned"} ${ids.length} conversation${ids.length > 1 ? "s" : ""}`,
      );
    },
    [],
  );

  // ── Export ────────────────────────────────────────────────────────

  const doExport = useCallback(
    async (id: string, format: "json" | "markdown") => {
      const result = await exportConversation(id, format);
      if (!result) return;
      // Download as file
      const blob = new Blob(
        [typeof result === "string" ? result : JSON.stringify(result, null, 2)],
        { type: format === "markdown" ? "text/markdown" : "application/json" },
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation-${id}.${format === "markdown" ? "md" : "json"}`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Conversation exported");
    },
    [],
  );

  // ── Derived ────────────────────────────────────────────────────────

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;

  return {
    conversations,
    total,
    loading,
    activeId,
    activeConversation,
    setActiveId,
    create,
    update,
    remove,
    rename,
    bulkDelete,
    bulkPin,
    doExport,
    refresh,
  };
}
