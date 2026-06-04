"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchRagCollections,
  searchRag,
  type RagCollection,
  type RagSearchMode,
  type RagSearchResponse,
} from "@/lib/api";

const DEFAULT_TOP_K = 10;

/**
 * State for the admin RAG knowledge explorer (`/admin/rag`).
 *
 * Search is **explicit** (call `runSearch()` from a submit/Enter handler) — not
 * keystroke-debounced — because each search embeds the query and hits Qdrant.
 * A sequence guard drops superseded responses so a slow earlier search can't
 * overwrite a newer one. Collections are loaded once on mount so the admin can
 * pick a target and confirm it is populated.
 */
export function useRagSearch() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<RagSearchMode>("raw");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [collection, setCollection] = useState<string>("");

  const [collections, setCollections] = useState<RagCollection[]>([]);
  const [result, setResult] = useState<RagSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const seqRef = useRef(0);

  // Load collections once; seed the dropdown with the configured default.
  useEffect(() => {
    let cancelled = false;
    fetchRagCollections().then((data) => {
      if (cancelled || !data) return;
      setCollections(data.collections);
      setCollection((current) => current || data.default_collection || "");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    const seq = ++seqRef.current; // invalidate any in-flight search
    setLoading(true);
    setError(null);
    try {
      const data = await searchRag({
        q,
        mode,
        topK,
        collection: collection || undefined,
      });
      if (seq !== seqRef.current) return; // superseded by a newer search
      setResult(data);
    } catch (err) {
      if (seq !== seqRef.current) return;
      setError(err instanceof Error ? err.message : "Knowledge base search failed");
      setResult(null);
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, [query, mode, topK, collection]);

  return {
    // inputs
    query,
    setQuery,
    mode,
    setMode,
    topK,
    setTopK,
    collection,
    setCollection,
    collections,
    // output
    result,
    loading,
    error,
    runSearch,
  };
}
