"use client";

import { memo, useMemo } from "react";
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useI18n } from "@/hooks/useI18n";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";
import { ChartTooltipCard } from "@/components/product/ChartTooltipCard";

interface LatencyAnalysisChartProps {
  data: Array<{
    date: string;
    avg_latency_ms: number;
    min_latency_ms: number;
    max_latency_ms: number;
    requests: number;
  }>;
  loading?: boolean;
  showMin?: boolean;
  showAvg?: boolean;
  showMax?: boolean;
}

function LatencyAnalysisChart({
  data,
  loading = false,
  showMin = true,
  showAvg = true,
  showMax = true,
}: LatencyAnalysisChartProps) {
  const { t, locale } = useI18n();
  // Always call hooks at the top level
  const chartData = useMemo(
    () =>
      data?.map((d) => ({
        date: new Date(d.date).toLocaleDateString(locale, { month: "short", day: "numeric" }),
        fullDate: d.date,
        avg: Math.round(d.avg_latency_ms * 10) / 10,
        min: Math.round(d.min_latency_ms * 10) / 10,
        max: Math.round(d.max_latency_ms * 10) / 10,
      })) ?? [],
    [data, locale]
  );

  if (loading) {
    return (
      <AnalyticsChartCard title={t("metrics.latencyAnalysis")}>
        <div className="py-12 text-center text-muted-foreground">{t("charts.loading")}</div>
      </AnalyticsChartCard>
    );
  }

  if (!data || data.length === 0) {
    return (
      <AnalyticsChartCard title={t("metrics.latencyAnalysis")}>
        <div className="py-12 text-center text-muted-foreground">{t("metrics.noLatencyData")}</div>
      </AnalyticsChartCard>
    );
  }

  if (!showMin && !showAvg && !showMax) {
    return (
      <AnalyticsChartCard title={t("metrics.latencyAnalysis")}>
        <div className="py-12 text-center text-muted-foreground">{t("charts.selectAtLeastOneSeries")}</div>
      </AnalyticsChartCard>
    );
  }

  return (
    <AnalyticsChartCard title={t("charts.requestLatencyAnalysisMs")}>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="date" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <YAxis tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {showMax && <Bar dataKey="max" stackId="latency" fill="var(--chart-red)" name={t("charts.maxLatency")} />}
          {showAvg && <Line type="monotone" dataKey="avg" stroke="var(--chart-blue)" strokeWidth={2} name={t("charts.avgLatency")} />}
          {showMin && <Bar dataKey="min" stackId="latency" fill="var(--chart-green)" name={t("charts.minLatency")} />}
        </ComposedChart>
      </ResponsiveContainer>
    </AnalyticsChartCard>
  );
}

export default memo(LatencyAnalysisChart);

function CustomTooltip({ active, payload }: any) {
  const { formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <ChartTooltipCard title={data.fullDate}>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {entry.name}: {formatNumber(entry.value)}ms
          </p>
        ))}
      </ChartTooltipCard>
    );
  }
  return null;
}
