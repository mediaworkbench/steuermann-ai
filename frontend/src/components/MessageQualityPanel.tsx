"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MessageQualityResponse } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { Icon } from "./Icon";
import { AnalyticsPanelSurface } from "@/components/product/AnalyticsPanelSurface";
import { AnalyticsStatCard } from "@/components/product/AnalyticsStatCard";

interface Props {
  data: MessageQualityResponse | null;
  formatNumber: (n: number) => string;
}

export function MessageQualityPanel({ data, formatNumber }: Props) {
  const { t } = useI18n();

  if (!data || data.quality_data.length === 0) {
    return (
      <AnalyticsPanelSurface>
        <h3 className="text-sm font-semibold text-foreground mb-1">{t("metrics.messageQuality")}</h3>
        <p className="text-xs text-muted-foreground">{t("metrics.messageQualitySubtitle")}</p>
        <p className="text-xs text-muted-foreground italic mt-3">{t("metrics.noMessageQualityData")}</p>
      </AnalyticsPanelSurface>
    );
  }

  const feedbackRatePct = (data.feedback_rate * 100).toFixed(1);

  const chartData = data.quality_data.map((point) => ({
    date: point.date,
    up: point.up_count,
    down: -point.down_count,
  }));

  return (
    <AnalyticsPanelSurface className="space-y-5">
      <div>
        <h3 className="text-sm font-semibold text-foreground">{t("metrics.messageQuality")}</h3>
        <p className="text-xs text-muted-foreground mt-0.5">{t("metrics.messageQualitySubtitle")}</p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <AnalyticsStatCard
          icon={<Icon name="thumb_up" size={16} />}
          label={t("metrics.thumbsUp")}
          value={formatNumber(data.total_up)}
        />
        <AnalyticsStatCard
          icon={<Icon name="thumb_down" size={16} />}
          label={t("metrics.thumbsDown")}
          value={formatNumber(data.total_down)}
        />
        <AnalyticsStatCard
          icon={<Icon name="trending_up" size={16} />}
          label={t("metrics.netScore")}
          value={formatNumber(data.net_score)}
        />
        <AnalyticsStatCard
          icon={<Icon name="chat_bubble" size={16} />}
          label={t("metrics.feedbackRate")}
          value={feedbackRatePct}
          unit="%"
        />
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <XAxis dataKey="date" tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
            <YAxis tick={{ fill: "var(--chart-axis)", fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="up" name={t("metrics.thumbsUp")} fill="var(--chart-green)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="down" name={t("metrics.thumbsDown")} fill="var(--chart-red)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="text-xs text-muted-foreground">
        <span className="font-semibold text-foreground mr-1">{t("metrics.totalFeedback")}:</span>
        {formatNumber(data.total_feedback)}
      </div>
    </AnalyticsPanelSurface>
  );
}
