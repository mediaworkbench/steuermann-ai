"use client";

import { MessageSquare, Star, ThumbsUp, BarChart2 } from "lucide-react";
import type { MemoryRetrievalQualityData } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

interface Props {
  data: MemoryRetrievalQualityData | null;
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

const BUCKET_ORDER = ["high", "mid", "low", "none"] as const;
const BUCKET_COLORS: Record<string, string> = {
  high: "bg-emerald-500",
  mid: "bg-amber-400",
  low: "bg-red-400",
  none: "bg-slate-300",
};

export function RetrievalFeedbackPanel({ data, formatNumber }: Props) {
  const { t } = useI18n();

  if (!data || data.retrieval_signals_total === 0) {
    return (
      <div className="rounded-2xl border border-evergreen/10 bg-white p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-evergreen/70 mb-1">
          {t("metrics.retrievalFeedback")}
        </h3>
        <p className="text-xs text-evergreen/40 italic">{t("metrics.noRetrievalData")}</p>
      </div>
    );
  }

  const total = data.retrieval_signals_total;
  const priorPct = (data.prior_rating_coverage * 100).toFixed(1);
  const feedbackPct = (data.feedback_coverage * 100).toFixed(1);

  // Build bucket bar segments
  const buckets = data.rating_bucket_distribution ?? {};
  const bucketTotal = Object.values(buckets).reduce((s, v) => s + v, 0) || 1;

  const bucketLabels: Record<string, string> = {
    high: t("metrics.bucketHigh"),
    mid: t("metrics.bucketMid"),
    low: t("metrics.bucketLow"),
    none: t("metrics.bucketNone"),
  };

  return (
    <div className="rounded-2xl border border-evergreen/10 bg-white p-6 shadow-sm space-y-5">
      <div>
        <h3 className="text-sm font-semibold text-evergreen/70">
          {t("metrics.retrievalFeedback")}
        </h3>
        <p className="text-xs text-evergreen/40 mt-0.5">
          {t("metrics.retrievalFeedbackSubtitle")}
        </p>
      </div>

      {/* Mini-cards row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MiniCard
          icon={<MessageSquare size={16} />}
          label={t("metrics.retrievalSignalsTotal")}
          value={formatNumber(total)}
        />
        <MiniCard
          icon={<Star size={16} />}
          label={t("metrics.priorRatingCoverage")}
          value={priorPct}
          unit="%"
        />
        <MiniCard
          icon={<ThumbsUp size={16} />}
          label={t("metrics.ratedAfterRetrieval")}
          value={formatNumber(data.rated_after_retrieval_total)}
        />
        <MiniCard
          icon={<BarChart2 size={16} />}
          label={t("metrics.feedbackCoverage")}
          value={feedbackPct}
          unit="%"
        />
      </div>

      {/* Rating bucket breakdown bar */}
      {bucketTotal > 0 && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-evergreen/40 mb-2">
            {t("metrics.ratingBuckets")}
          </p>
          {/* Stacked bar */}
          <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-100">
            {BUCKET_ORDER.map((key) => {
              const count = buckets[key] ?? 0;
              const pct = (count / bucketTotal) * 100;
              if (pct < 0.5) return null;
              return (
                <div
                  key={key}
                  title={`${bucketLabels[key]}: ${count}`}
                  className={`${BUCKET_COLORS[key]} transition-all`}
                  style={{ width: `${pct}%` }}
                />
              );
            })}
          </div>
          {/* Legend */}
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
            {BUCKET_ORDER.map((key) => {
              const count = buckets[key] ?? 0;
              if (!count) return null;
              const pct = ((count / bucketTotal) * 100).toFixed(1);
              return (
                <span key={key} className="flex items-center gap-1.5 text-xs text-evergreen/60">
                  <span className={`inline-block h-2 w-2 rounded-full ${BUCKET_COLORS[key]}`} />
                  {bucketLabels[key]}: {formatNumber(count)} ({pct}%)
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Retrieved-with vs without rating */}
      <div className="flex gap-4 text-xs text-evergreen/60">
        <span>
          <span className="font-semibold text-evergreen/80">
            {formatNumber(data.retrieved_with_prior_rating)}
          </span>{" "}
          {t("metrics.retrievedWithRating")}
        </span>
        <span>
          <span className="font-semibold text-evergreen/80">
            {formatNumber(data.retrieved_without_prior_rating)}
          </span>{" "}
          {t("metrics.retrievedWithoutRating")}
        </span>
      </div>
    </div>
  );
}
