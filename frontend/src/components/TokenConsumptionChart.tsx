"use client";

import { memo, useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { useI18n } from "@/hooks/useI18n";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";
import { AnalyticsChartState } from "@/components/product/AnalyticsChartState";
import { AnalyticsChartViewport } from "@/components/product/AnalyticsChartViewport";
import { ChartTooltipCard } from "@/components/product/ChartTooltipCard";

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
      <AnalyticsChartCard title={t("metrics.tokenConsumption")}>
        <AnalyticsChartState message={t("charts.loading")} />
      </AnalyticsChartCard>
    );
  }

  if (!data || data.length === 0) {
    return (
      <AnalyticsChartCard title={t("metrics.tokenConsumption")}>
        <AnalyticsChartState message={t("metrics.noTokenData")} />
      </AnalyticsChartCard>
    );
  }

  if (!showTotalTokens && !showAvgTokens) {
    return (
      <AnalyticsChartCard title={t("metrics.tokenConsumption")}>
        <AnalyticsChartState message={t("charts.selectAtLeastOneSeries")} />
      </AnalyticsChartCard>
    );
  }

  return (
    <AnalyticsChartCard title={t("charts.dailyTokenConsumption")}>
      <AnalyticsChartViewport>
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
      </AnalyticsChartViewport>
    </AnalyticsChartCard>
  );
}

export default memo(TokenConsumptionChart);

function CustomTooltip({ active, payload }: any) {
  const { formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <ChartTooltipCard title={data.fullDate}>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {entry.name}: {formatNumber(entry.value)}
          </p>
        ))}
      </ChartTooltipCard>
    );
  }
  return null;
}
