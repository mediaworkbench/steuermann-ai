"use client";

import { MessageSquare, ThumbsDown, ThumbsUp, TrendingUp } from "lucide-react";
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

interface Props {
  data: MessageQualityResponse | null;
  formatNumber: (n: number) => string;
}

function MiniCard({
  icon,
  label,
  value,
  unit,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  unit?: string;
}) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-evergreen/10 bg-white px-4 py-3 shadow-sm">
      <div className="mt-0.5 text-evergreen/50">{icon}</div>
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-evergreen/40 mb-0.5">
          {label}
        </p>
        <p className="text-xl font-bold text-evergreen">
          {value}
          {unit && <span className="text-sm font-medium text-evergreen/50 ml-1">{unit}</span>}
        </p>
      </div>
    </div>
  );
}

export function MessageQualityPanel({ data, formatNumber }: Props) {
  const { t } = useI18n();

  if (!data || data.quality_data.length === 0) {
    return (
      <div className="rounded-2xl border border-evergreen/10 bg-white p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-evergreen/70 mb-1">{t("metrics.messageQuality")}</h3>
        <p className="text-xs text-evergreen/40">{t("metrics.messageQualitySubtitle")}</p>
        <p className="text-xs text-evergreen/40 italic mt-3">{t("metrics.noMessageQualityData")}</p>
      </div>
    );
  }

  const feedbackRatePct = (data.feedback_rate * 100).toFixed(1);

  const chartData = data.quality_data.map((point) => ({
    date: point.date,
    up: point.up_count,
    down: -point.down_count,
  }));

  return (
    <div className="rounded-2xl border border-evergreen/10 bg-white p-6 shadow-sm space-y-5">
      <div>
        <h3 className="text-sm font-semibold text-evergreen/70">{t("metrics.messageQuality")}</h3>
        <p className="text-xs text-evergreen/40 mt-0.5">{t("metrics.messageQualitySubtitle")}</p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MiniCard
          icon={<ThumbsUp size={16} />}
          label={t("metrics.thumbsUp")}
          value={formatNumber(data.total_up)}
        />
        <MiniCard
          icon={<ThumbsDown size={16} />}
          label={t("metrics.thumbsDown")}
          value={formatNumber(data.total_down)}
        />
        <MiniCard
          icon={<TrendingUp size={16} />}
          label={t("metrics.netScore")}
          value={formatNumber(data.net_score)}
        />
        <MiniCard
          icon={<MessageSquare size={16} />}
          label={t("metrics.feedbackRate")}
          value={feedbackRatePct}
          unit="%"
        />
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d1d5db" />
            <XAxis dataKey="date" tick={{ fill: "#4b5563", fontSize: 12 }} />
            <YAxis tick={{ fill: "#4b5563", fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="up" name={t("metrics.thumbsUp")} fill="#10b981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="down" name={t("metrics.thumbsDown")} fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="text-xs text-evergreen/60">
        <span className="font-semibold text-evergreen/80 mr-1">{t("metrics.totalFeedback")}:</span>
        {formatNumber(data.total_feedback)}
      </div>
    </div>
  );
}
