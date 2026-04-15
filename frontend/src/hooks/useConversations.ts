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
  const [showArchived, setShowArchived] = useState(false);
  const initialLoadDone = useRef(false);

  // ── Load conversations ─────────────────────────────────────────────

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchConversations(USER_ID, showArchived, 100, 0);
      if (data) {
        setConversations(data.conversations);
        setTotal(data.total);
      }
    } finally {
      setLoading(false);
    }
  }, [showArchived]);

  useEffect(() => {
    if (!initialLoadDone.current) {
      initialLoadDone.current = true;
      refresh();
    }
  }, [refresh]);

  // Re-fetch when showArchived changes (after initial load)
  useEffect(() => {
    if (initialLoadDone.current) refresh();
  }, [showArchived, refresh]);

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

  // ── Update (rename, pin, archive, language) ────────────────────────

  const update = useCallback(
    async (
      id: string,
      updates: { title?: string; archived?: boolean; pinned?: boolean; language?: string },
    ) => {
      const conv = await updateConversation(id, updates);
      if (conv) {
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? conv : c)),
        );
        // If archiving the active conversation, deselect it
        if (updates.archived && id === activeId) {
          setActiveId(null);
        }
      }
      return conv;
    },
    [activeId],
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

  // ── Archive / Unarchive ───────────────────────────────────────────

  const archive = useCallback(
    async (id: string, archived: boolean) => {
      await update(id, { archived });
      toast.success(archived ? "Conversation archived" : "Conversation unarchived");
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

  // ── Bulk archive ──────────────────────────────────────────────────

  const bulkArchive = useCallback(
    async (ids: string[]) => {
      await Promise.all(ids.map((id) => updateConversation(id, { archived: true })));
      setConversations((prev) =>
        prev.map((c) => (ids.includes(c.id) ? { ...c, archived: true } : c)),
      );
      if (activeId && ids.includes(activeId)) setActiveId(null);
      toast.success(`Archived ${ids.length} conversation${ids.length > 1 ? "s" : ""}`);
    },
    [activeId],
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

  // ── Toggle archived view ──────────────────────────────────────────

  const toggleArchived = useCallback(() => {
    setShowArchived((v) => !v);
  }, []);

  // ── Derived ────────────────────────────────────────────────────────

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;

  return {
    conversations,
    total,
    loading,
    activeId,
    activeConversation,
    showArchived,
    setActiveId,
    create,
    update,
    remove,
    rename,
    archive,
    bulkDelete,
    bulkArchive,
    doExport,
    toggleArchived,
    refresh,
  };
}
