"use client";

import { memo, useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useI18n } from "@/hooks/useI18n";

interface TokenConsumptionChartProps {
  data: Array<{
    date: string;
    total_tokens: number;
    avg_tokens: number;
    requests: number;
  }>;
  loading?: boolean;
  showTotalTokens?: boolean;
  showAvgTokens?: boolean;
}

function TokenConsumptionChart({
  data,
  loading = false,
  showTotalTokens = true,
  showAvgTokens = true,
}: TokenConsumptionChartProps) {
  const { t, locale } = useI18n();
  // Always call hooks at the top level
  const chartData = useMemo(
    () =>
      data?.map((d) => ({
        date: new Date(d.date).toLocaleDateString(locale, { month: "short", day: "numeric" }),
        fullDate: d.date,
        total_tokens: d.total_tokens,
        avg_tokens: Math.round(d.avg_tokens),
      })) ?? [],
    [data, locale]
  );

  if (loading) {
    return (
      <div className="bg-surface border border-border rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">{t("metrics.tokenConsumption")}</h3>
        <div className="text-center py-12 text-muted-foreground">{t("charts.loading")}</div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="bg-surface border border-border rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">{t("metrics.tokenConsumption")}</h3>
        <div className="text-center py-12 text-muted-foreground">{t("metrics.noTokenData")}</div>
      </div>
    );
  }

  if (!showTotalTokens && !showAvgTokens) {
    return (
      <div className="bg-surface border border-border rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">{t("metrics.tokenConsumption")}</h3>
        <div className="text-center py-12 text-muted-foreground">{t("charts.selectAtLeastOneSeries")}</div>
      </div>
    );
  }

  return (
    <div className="bg-surface border border-border rounded-lg shadow-sm p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">{t("charts.dailyTokenConsumption")}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="date" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <YAxis yAxisId="left" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {showTotalTokens && (
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="total_tokens"
              fill="var(--chart-amber-soft)"
              stroke="var(--chart-amber)"
              name={t("metrics.totalTokens")}
            />
          )}
          {showAvgTokens && (
            <Area
              yAxisId="right"
              type="monotone"
              dataKey="avg_tokens"
              fill="var(--chart-blue-soft)"
              stroke="var(--chart-blue)"
              name={t("metrics.average")}
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export default memo(TokenConsumptionChart);

function CustomTooltip({ active, payload }: any) {
  const { formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-surface p-3 border border-border rounded shadow-lg">
        <p className="text-sm font-semibold text-foreground">{data.fullDate}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {entry.name}: {formatNumber(entry.value)}
          </p>
        ))}
      </div>
    );
  }
  return null;
}
