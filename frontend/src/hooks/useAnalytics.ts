"use client";

import { useCallback, useEffect, useState } from "react";
import {
  fetchUsageTrends,
  fetchTokenConsumption,
  fetchLatencyAnalysis,
} from "@/lib/api";

interface UseAnalyticsOptions {
  days?: number;
  autoRefresh?: boolean;
  refetchInterval?: number;
}

interface UseAnalyticsReturn {
  usageTrends: { date: string; requests: number; users: number }[] | null;
  tokenConsumption: { date: string; total_tokens: number; avg_tokens: number; requests: number }[] | null;
  latencyAnalysis: { date: string; avg_latency_ms: number; min_latency_ms: number; max_latency_ms: number; requests: number }[] | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useAnalytics(options?: UseAnalyticsOptions): UseAnalyticsReturn {
  const {
    days = 30,
    autoRefresh = false,
    refetchInterval = 60000,
  } = options || {};
  const [usageTrends, setUsageTrends] = useState<{ date: string; requests: number; users: number }[] | null>(null);
  const [tokenConsumption, setTokenConsumption] = useState<{ date: string; total_tokens: number; avg_tokens: number; requests: number }[] | null>(null);
  const [latencyAnalysis, setLatencyAnalysis] = useState<{ date: string; avg_latency_ms: number; min_latency_ms: number; max_latency_ms: number; requests: number }[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [trends, tokens, latency] = await Promise.all([
        fetchUsageTrends(days),
        fetchTokenConsumption(days),
        fetchLatencyAnalysis(days),
      ]);

      if (trends) setUsageTrends(trends.trends || []);
      if (tokens) setTokenConsumption(tokens.consumption || []);
      if (latency) setLatencyAnalysis(latency.latency_data || []);

      if (!trends || !tokens || !latency) {
        setError("Failed to load some analytics data");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    refetch();

    if (autoRefresh) {
      const interval = setInterval(refetch, refetchInterval);
      return () => clearInterval(interval);
    }
  }, [refetch, autoRefresh, refetchInterval]);

  return {
    usageTrends,
    tokenConsumption,
    latencyAnalysis,
    loading,
    error,
    refetch,
  };
}
