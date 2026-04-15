"use client";

import { memo, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useI18n } from "@/hooks/useI18n";

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
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("metrics.usageTrends")}</h3>
        <div className="text-center py-12 text-gray-500">{t("charts.loading")}</div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("metrics.usageTrends")}</h3>
        <div className="text-center py-12 text-gray-500">{t("metrics.noUsageData")}</div>
      </div>
    );
  }

  if (!showRequests && !showUsers) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("metrics.usageTrends")}</h3>
        <div className="text-center py-12 text-gray-500">{t("charts.selectAtLeastOneSeries")}</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{t("charts.dailyUsageTrends")}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          {showRequests && (
            <Line yAxisId="left" type="monotone" dataKey="requests" stroke="#3b82f6" name={t("metrics.requests")} />
          )}
          {showUsers && (
            <Line yAxisId="right" type="monotone" dataKey="users" stroke="#10b981" name={t("metrics.users")} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default memo(UsageTrendChart);

function CustomTooltip({ active, payload }: any) {
  const { formatNumber } = useI18n();
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
        <p className="text-sm font-semibold text-gray-900">{data.fullDate}</p>
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
