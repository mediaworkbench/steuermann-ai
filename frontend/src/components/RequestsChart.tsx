"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { useI18n } from "@/hooks/useI18n";

interface RequestsChartProps {
  data: Record<string, number>;
}

const COLORS = ["#10b981", "#f59e0b", "#ef4444", "#6366f1", "#8b5cf6"];

export function RequestsChart({ data }: RequestsChartProps) {
  const { t, formatNumber } = useI18n();
  if (Object.keys(data).length === 0) {
    return (
      <div className="rounded-lg border border-border bg-surface p-6 shadow-sm">
        <h3 className="mb-4 text-lg font-semibold text-foreground">{t("charts.requestsByStatus")}</h3>
        <div className="py-8 text-center text-muted-foreground">{t("charts.noRequestData")}</div>
      </div>
    );
  }

  const chartData = Object.entries(data).map(([name, value]) => ({
    name,
    value: typeof value === "number" ? Math.round(value) : 0,
  }));

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="rounded-lg border border-border bg-surface p-6 shadow-sm">
      <h3 className="mb-4 text-lg font-semibold text-foreground">{t("charts.requestsByStatus")}</h3>
      <div className="mb-4 text-sm text-muted-foreground">{t("charts.total")}: {formatNumber(total)}</div>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name}: ${value}`}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => formatNumber(value)} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
