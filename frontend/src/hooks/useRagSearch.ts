"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchRagCollections,
  searchRag,
  type RagCollection,
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
export function useRagSearch(defaultCollectionName?: string) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [collection, setCollection] = useState<string>(defaultCollectionName ?? "");

  const [collections, setCollections] = useState<RagCollection[]>([]);
  const [result, setResult] = useState<RagSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const seqRef = useRef(0);
  const userInteractedRef = useRef(false);

  const handleSetCollection = useCallback((value: string) => {
    userInteractedRef.current = true;
    setCollection(value);
  }, []);

  // When the profile's default collection name becomes available, apply it
  // if no user selection has been made yet.
  useEffect(() => {
    if (defaultCollectionName && !userInteractedRef.current) {
      setCollection(defaultCollectionName);
    }
  }, [defaultCollectionName]);

  // Load collections once; seed the dropdown with the configured default.
  useEffect(() => {
    let cancelled = false;
    fetchRagCollections().then((data) => {
      if (cancelled || !data) return;
      setCollections(data.collections);
      setCollection((current) => current || defaultCollectionName || data.default_collection || "");
    });
    return () => {
      cancelled = true;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    const seq = ++seqRef.current; // invalidate any in-flight search
    setLoading(true);
    setError(null);
    try {
      const data = await searchRag({ q, topK, collection: collection || undefined });
      if (seq !== seqRef.current) return; // superseded by a newer search
      setResult(data);
    } catch (err) {
      if (seq !== seqRef.current) return;
      setError(err instanceof Error ? err.message : "Knowledge base search failed");
      setResult(null);
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, [query, topK, collection]);

  return {
    // inputs
    query,
    setQuery,
    topK,
    setTopK,
    collection,
    setCollection: handleSetCollection,
    collections,
    // output
    result,
    loading,
    error,
    runSearch,
  };
}
