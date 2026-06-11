"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { toast } from "sonner";
import type { Conversation } from "@/lib/types";
import {
  createConversation,
  fetchConversations,
  fetchConversation,
  updateConversation,
  deleteConversation,
} from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";

const USER_ID = CURRENT_USER_ID;

// The shared context loads only a modest slice — enough for the sidebar's pinned + 5
// recent. The /chats page does its own server-side pagination/search via
// useConversationBrowser, so it does not depend on this list's size.
const SIDEBAR_LIMIT = 50;

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [activeId, setActiveIdState] = useState<string | null>(null);
  // Fallback for activeConversation when the active chat isn't in the loaded slice
  // (e.g. opened from a later /chats page, deep link, or reload). Seeded from the
  // clicked row when available, otherwise fetched by id.
  const [activeConvCache, setActiveConvCache] = useState<Conversation | null>(null);
  // Monotonic counter bumped after every successful mutation. Other views (the
  // /chats browser) watch it to refetch so all surfaces stay consistent.
  const [revision, setRevision] = useState(0);
  const initialLoadDone = useRef(false);

  const bump = useCallback(() => setRevision((r) => r + 1), []);

  // ── Load conversations (sidebar slice) ─────────────────────────────

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchConversations(USER_ID, SIDEBAR_LIMIT, 0);
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

  // ── Active conversation selection ──────────────────────────────────

  // Accepts an optional conversation to seed the cache (avoids a fetch when the
  // caller — sidebar/chats row — already has the full object).
  const setActiveId = useCallback((id: string | null, conv?: Conversation | null) => {
    setActiveIdState(id);
    if (conv) setActiveConvCache(conv);
    else if (id === null) setActiveConvCache(null);
  }, []);

  // Resolve activeConversation by id: prefer the loaded slice, else the seeded/fetched cache.
  const activeConversation = useMemo<Conversation | null>(() => {
    if (!activeId) return null;
    return (
      conversations.find((c) => c.id === activeId) ??
      (activeConvCache?.id === activeId ? activeConvCache : null)
    );
  }, [conversations, activeId, activeConvCache]);

  // When the active chat is neither in the loaded slice nor cached, fetch it once by id.
  useEffect(() => {
    if (!activeId) return;
    if (conversations.some((c) => c.id === activeId)) return;
    if (activeConvCache?.id === activeId) return;
    let cancelled = false;
    (async () => {
      const detail = await fetchConversation(activeId);
      if (!cancelled && detail?.conversation) setActiveConvCache(detail.conversation);
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId, conversations, activeConvCache]);

  // ── Create ─────────────────────────────────────────────────────────

  const create = useCallback(
    async (title?: string, language?: string) => {
      const conv = await createConversation(USER_ID, title || "New conversation", language || "en");
      if (conv) {
        setConversations((prev) => [conv, ...prev]);
        setTotal((t) => t + 1);
        setActiveIdState(conv.id);
        setActiveConvCache(conv);
        bump();
      }
      return conv;
    },
    [bump],
  );

  // ── Update (rename, pin, language) ─────────────────────────────────

  const update = useCallback(
    async (
      id: string,
      updates: { title?: string; pinned?: boolean; language?: string },
    ) => {
      const conv = await updateConversation(id, updates);
      if (conv) {
        setConversations((prev) => prev.map((c) => (c.id === id ? conv : c)));
        setActiveConvCache((prev) => (prev?.id === id ? conv : prev));
        bump();
      }
      return conv;
    },
    [bump],
  );

  // ── Delete ─────────────────────────────────────────────────────────

  const remove = useCallback(
    async (id: string) => {
      const ok = await deleteConversation(id);
      if (ok) {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        setTotal((t) => Math.max(0, t - 1));
        if (activeId === id) {
          setActiveIdState(null);
          setActiveConvCache(null);
        }
        bump();
        toast.success("Conversation deleted");
      } else {
        toast.error("Failed to delete conversation");
      }
      return ok;
    },
    [activeId, bump],
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
      if (activeId && ids.includes(activeId)) {
        setActiveIdState(null);
        setActiveConvCache(null);
      }
      bump();
      toast.success(`Deleted ${ids.length} conversation${ids.length > 1 ? "s" : ""}`);
    },
    [activeId, bump],
  );

  // ── Bulk pin / unpin ──────────────────────────────────────────────

  const bulkPin = useCallback(
    async (ids: string[], pinned: boolean) => {
      await Promise.all(ids.map((id) => updateConversation(id, { pinned })));
      setConversations((prev) =>
        prev.map((c) => (ids.includes(c.id) ? { ...c, pinned } : c)),
      );
      setActiveConvCache((prev) => (prev && ids.includes(prev.id) ? { ...prev, pinned } : prev));
      bump();
      toast.success(
        `${pinned ? "Pinned" : "Unpinned"} ${ids.length} conversation${ids.length > 1 ? "s" : ""}`,
      );
    },
    [bump],
  );

  return {
    conversations,
    total,
    loading,
    activeId,
    activeConversation,
    revision,
    setActiveId,
    create,
    update,
    remove,
    rename,
    bulkDelete,
    bulkPin,
    refresh,
  };
}
