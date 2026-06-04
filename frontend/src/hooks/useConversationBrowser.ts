"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { fetchConversations } from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type { Conversation } from "@/lib/types";

const PAGE_SIZE = 50;

/**
 * Server-side paginated + searchable conversation list for the /chats page.
 *
 * Independent of the sidebar's `useConversations` slice — it queries the server
 * directly (`GET /api/conversations?q=&limit=&offset=`) so it can browse/search
 * across **all** chats, not just the loaded slice. Pass the context `revision`
 * so any mutation (from this page or the sidebar) triggers a refetch of the
 * current page, keeping both surfaces consistent.
 */
export function useConversationBrowser(revision: number) {
  const [items, setItems] = useState<Conversation[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [query, setQueryState] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const seqRef = useRef(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounce the raw query into the value the fetch keys on.
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setDebouncedQuery(query.trim()), 350);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  const load = useCallback(async (off: number, q: string) => {
    const seq = ++seqRef.current; // invalidate any in-flight request
    setLoading(true);
    try {
      const data = await fetchConversations(CURRENT_USER_ID, PAGE_SIZE, off, q || undefined);
      if (seq !== seqRef.current) return; // superseded by a newer load
      if (data) {
        setItems(data.conversations);
        setTotal(data.total);
        // Clamp: if we've paged past the end (e.g. after deleting the last items
        // on the final page), step back to the last valid page (triggers a reload).
        if (off > 0 && off >= data.total) {
          setOffset(Math.max(0, (Math.ceil(data.total / PAGE_SIZE) - 1) * PAGE_SIZE));
        }
      }
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, []);

  // Refetch whenever the page, the (debounced) query, or the shared revision changes.
  useEffect(() => {
    load(offset, debouncedQuery);
  }, [offset, debouncedQuery, revision, load]);

  const setQuery = useCallback((q: string) => {
    setQueryState(q);
    setOffset(0); // reset paging in the same handler to avoid a double-fetch
  }, []);

  const nextPage = useCallback(() => setOffset((o) => o + PAGE_SIZE), []);
  const prevPage = useCallback(() => setOffset((o) => Math.max(0, o - PAGE_SIZE)), []);
  const refresh = useCallback(() => load(offset, debouncedQuery), [load, offset, debouncedQuery]);

  // Optimistic local edits so single-row pin/rename/delete feel instant; the
  // revision-driven refetch then reconciles against the server.
  const patchItem = useCallback((id: string, patch: Partial<Conversation>) => {
    setItems((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }, []);
  const removeItems = useCallback((ids: string[]) => {
    const drop = new Set(ids);
    setItems((prev) => prev.filter((c) => !drop.has(c.id)));
  }, []);

  return {
    items,
    total,
    offset,
    query,
    loading,
    pageSize: PAGE_SIZE,
    setQuery,
    nextPage,
    prevPage,
    refresh,
    patchItem,
    removeItems,
  };
}
