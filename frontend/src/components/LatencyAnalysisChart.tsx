"use client";

import { memo, useMemo } from "react";
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useI18n } from "@/hooks/useI18n";

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
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("metrics.latencyAnalysis")}</h3>
        <div className="text-center py-12 text-gray-500">{t("charts.loading")}</div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("metrics.latencyAnalysis")}</h3>
        <div className="text-center py-12 text-gray-500">{t("metrics.noLatencyData")}</div>
      </div>
    );
  }

  if (!showMin && !showAvg && !showMax) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("metrics.latencyAnalysis")}</h3>
        <div className="text-center py-12 text-gray-500">{t("charts.selectAtLeastOneSeries")}</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("charts.requestLatencyAnalysisMs")}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {showMax && <Bar dataKey="max" stackId="latency" fill="#ef4444" name={t("charts.maxLatency")} />}
          {showAvg && <Line type="monotone" dataKey="avg" stroke="#3b82f6" strokeWidth={2} name={t("charts.avgLatency")} />}
          {showMin && <Bar dataKey="min" stackId="latency" fill="#10b981" name={t("charts.minLatency")} />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export default memo(LatencyAnalysisChart);

function CustomTooltip({ active, payload }: any) {
  const { formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
        <p className="text-sm font-semibold text-gray-900">{data.fullDate}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {entry.name}: {formatNumber(entry.value)}ms
          </p>
        ))}
      </div>
    );
  }
  return null;
}
