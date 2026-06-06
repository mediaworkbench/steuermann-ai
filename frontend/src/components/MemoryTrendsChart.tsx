"use client";

import { memo, useMemo } from "react";
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { useI18n } from "@/hooks/useI18n";

interface MemoryTrendsChartProps {
  data: Array<{
    date: string;
    loads: number;
    updates: number;
    errors: number;
    error_rate: number;
    avg_quality_score: number;
  }>;
  loading?: boolean;
}

function MemoryTrendsChart({ data, loading = false }: MemoryTrendsChartProps) {
  const { locale, t } = useI18n();

  const chartData = useMemo(
    () =>
      data?.map((d) => ({
        date: new Date(d.date).toLocaleDateString(locale, { month: "short", day: "numeric" }),
        fullDate: d.date,
        loads: d.loads,
        updates: d.updates,
        errors: d.errors,
        error_rate: d.error_rate,
        avg_quality_score_pct: Math.round(d.avg_quality_score * 1000) / 10,
      })) ?? [],
    [data, locale]
  );

  if (loading) {
    return <div className="text-center py-12 text-muted-foreground">{t("charts.loadingMemoryTrends")}</div>;
  }

  if (!data || data.length === 0) {
    return <div className="text-center py-12 text-muted-foreground">{t("charts.noMemoryTrendData")}</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
        <XAxis dataKey="date" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
        <YAxis yAxisId="left" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
        <YAxis yAxisId="right" orientation="right" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Bar yAxisId="left" dataKey="loads" fill="var(--chart-teal)" name={t("charts.loads")} />
        <Bar yAxisId="left" dataKey="updates" fill="var(--chart-teal-soft)" name={t("charts.updates")} />
        <Bar yAxisId="left" dataKey="errors" fill="var(--chart-red)" name={t("charts.errors")} />
        <Line yAxisId="right" type="monotone" dataKey="error_rate" stroke="var(--chart-orange)" strokeWidth={2} name={t("charts.errorRatePercent")} />
        <Line yAxisId="right" type="monotone" dataKey="avg_quality_score_pct" stroke="var(--chart-blue-strong)" strokeWidth={2} name={t("charts.qualityPercent")} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export default memo(MemoryTrendsChart);

function CustomTooltip({ active, payload }: any) {
  const { formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const row = payload[0].payload;
    return (
      <div className="bg-surface p-3 border border-border rounded shadow-lg">
        <p className="text-sm font-semibold text-foreground">{row.fullDate}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {entry.name}: {formatNumber(entry.value)}
            {entry.dataKey === "error_rate" || entry.dataKey === "avg_quality_score_pct" ? "%" : ""}
          </p>
        ))}
      </div>
    );
  }
  return null;
}
