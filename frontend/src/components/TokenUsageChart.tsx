"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useI18n } from "@/hooks/useI18n";

interface TokenUsageChartProps {
  data: Record<string, number>;
}

export function TokenUsageChart({ data }: TokenUsageChartProps) {
  const { t } = useI18n();
  if (Object.keys(data).length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("charts.tokenUsage")}</h3>
        <div className="text-center py-8 text-gray-500">{t("charts.noTokenUsageData")}</div>
      </div>
    );
  }

  const chartData = Object.entries(data).map(([label, value]) => ({
    name: label.length > 20 ? label.substring(0, 20) + "..." : label,
    tokens: typeof value === "number" ? Math.round(value) : 0,
    fullName: label,
  }));

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("charts.tokenUsageByModel")}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
          <YAxis />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="tokens" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CustomTooltip({ active, payload }: any) {
  const { t, formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
        <p className="text-sm font-semibold text-gray-900">{data.fullName}</p>
        <p className="text-sm text-blue-600">{t("charts.tokens")}: {formatNumber(payload[0].value)}</p>
      </div>
    );
  }
  return null;
}
