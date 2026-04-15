"use client";

import { useCallback, useEffect, useState } from "react";
import { MetricsData, fetchMetricsData } from "@/lib/api";

interface UseMetricsReturn {
  metrics: MetricsData | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useMetrics(autoRefresh: boolean = true, refreshInterval: number = 10000): UseMetricsReturn {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMetricsData();
      if (data) {
        setMetrics(data);
      } else {
        setError("Failed to load metrics");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      refetch();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, refetch]);

  return { metrics, loading, error, refetch };
}
