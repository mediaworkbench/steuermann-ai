"use client";

import { PieChart, Pie, Cell, Tooltip } from "recharts";
import { useI18n } from "@/hooks/useI18n";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";
import { AnalyticsChartState } from "@/components/product/AnalyticsChartState";
import { AnalyticsChartViewport } from "@/components/product/AnalyticsChartViewport";

interface RequestsChartProps {
  data: Record<string, number>;
}

const COLORS = [
  "var(--chart-green)",
  "var(--chart-amber)",
  "var(--chart-red)",
  "var(--chart-indigo)",
  "var(--chart-violet)",
];

export function RequestsChart({ data }: RequestsChartProps) {
  const { t, formatNumber } = useI18n();
  if (Object.keys(data).length === 0) {
    return (
      <AnalyticsChartCard title={t("charts.requestsByStatus")}>
        <AnalyticsChartState compact message={t("charts.noRequestData")} />
      </AnalyticsChartCard>
    );
  }

  const chartData = Object.entries(data).map(([name, value]) => ({
    name,
    value: typeof value === "number" ? Math.round(value) : 0,
  }));

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  return (
    <AnalyticsChartCard title={t("charts.requestsByStatus")}>
      <div className="mb-4 text-sm text-muted-foreground">{t("charts.total")}: {formatNumber(total)}</div>
      <AnalyticsChartViewport>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name}: ${value}`}
            outerRadius={100}
            fill="var(--chart-indigo)"
            dataKey="value"
          >
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => formatNumber(value)} />
        </PieChart>
      </AnalyticsChartViewport>
    </AnalyticsChartCard>
  );
}
