"use client";

import { memo, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useI18n } from "@/hooks/useI18n";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";
import { ChartTooltipCard } from "@/components/product/ChartTooltipCard";

interface UsageTrendChartProps {
  data: Array<{
    date: string;
    requests: number;
    users: number;
  }>;
  loading?: boolean;
  showRequests?: boolean;
  showUsers?: boolean;
}

function UsageTrendChart({
  data,
  loading = false,
  showRequests = true,
  showUsers = true,
}: UsageTrendChartProps) {
  const { t, locale } = useI18n();
  // Always call hooks at the top level
  const chartData = useMemo(
    () =>
      data?.map((d) => ({
        date: new Date(d.date).toLocaleDateString(locale, { month: "short", day: "numeric" }),
        fullDate: d.date,
        requests: d.requests,
        users: d.users,
      })) ?? [],
    [data, locale]
  );

  if (loading) {
    return (
      <AnalyticsChartCard title={t("metrics.usageTrends")}>
        <div className="py-12 text-center text-muted-foreground">{t("charts.loading")}</div>
      </AnalyticsChartCard>
    );
  }

  if (!data || data.length === 0) {
    return (
      <AnalyticsChartCard title={t("metrics.usageTrends")}>
        <div className="py-12 text-center text-muted-foreground">{t("metrics.noUsageData")}</div>
      </AnalyticsChartCard>
    );
  }

  if (!showRequests && !showUsers) {
    return (
      <AnalyticsChartCard title={t("metrics.usageTrends")}>
        <div className="py-12 text-center text-muted-foreground">{t("charts.selectAtLeastOneSeries")}</div>
      </AnalyticsChartCard>
    );
  }

  return (
    <AnalyticsChartCard title={t("charts.dailyUsageTrends")}>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="date" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <YAxis yAxisId="left" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {showRequests && (
            <Line yAxisId="left" type="monotone" dataKey="requests" stroke="var(--chart-blue)" name={t("metrics.requests")} />
          )}
          {showUsers && (
            <Line yAxisId="right" type="monotone" dataKey="users" stroke="var(--chart-green)" name={t("metrics.users")} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </AnalyticsChartCard>
  );
}

export default memo(UsageTrendChart);

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
