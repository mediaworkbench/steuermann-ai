"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useI18n } from "@/hooks/useI18n";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";
import { ChartTooltipCard } from "@/components/product/ChartTooltipCard";

interface TokenUsageChartProps {
  data: Record<string, number>;
}

export function TokenUsageChart({ data }: TokenUsageChartProps) {
  const { t } = useI18n();
  if (Object.keys(data).length === 0) {
    return (
      <AnalyticsChartCard title={t("charts.tokenUsage")}>
        <div className="py-8 text-center text-muted-foreground">{t("charts.noTokenUsageData")}</div>
      </AnalyticsChartCard>
    );
  }

  const chartData = Object.entries(data).map(([label, value]) => ({
    name: label.length > 20 ? label.substring(0, 20) + "..." : label,
    tokens: typeof value === "number" ? Math.round(value) : 0,
    fullName: label,
  }));

  return (
    <AnalyticsChartCard title={t("charts.tokenUsageByModel")}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <YAxis tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="tokens" fill="var(--chart-blue)" />
        </BarChart>
      </ResponsiveContainer>
    </AnalyticsChartCard>
  );
}

function CustomTooltip({ active, payload }: any) {
  const { t, formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <ChartTooltipCard title={data.fullName}>
        <p className="text-sm text-primary">{t("charts.tokens")}: {formatNumber(payload[0].value)}</p>
      </ChartTooltipCard>
    );
  }
  return null;
}
