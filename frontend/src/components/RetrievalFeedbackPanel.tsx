"use client";

import type { MemoryRetrievalQualityData } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { Icon } from "./Icon";
import { AnalyticsPanelSurface } from "@/components/product/AnalyticsPanelSurface";
import { AnalyticsStatCard } from "@/components/product/AnalyticsStatCard";

interface Props {
  data: MemoryRetrievalQualityData | null;
  formatNumber: (n: number) => string;
}

const BUCKET_ORDER = ["high", "mid", "low", "none"] as const;
const BUCKET_COLORS: Record<string, string> = {
  high: "bg-success",
  mid: "bg-warning",
  low: "bg-destructive",
  none: "bg-border-strong",
};

export function RetrievalFeedbackPanel({ data, formatNumber }: Props) {
  const { t } = useI18n();

  if (!data || data.retrieval_signals_total === 0) {
    return (
      <AnalyticsPanelSurface>
        <h3 className="mb-1 text-sm font-semibold text-foreground">
          {t("metrics.retrievalFeedback")}
        </h3>
        <p className="text-xs italic text-muted-foreground">{t("metrics.noRetrievalData")}</p>
      </AnalyticsPanelSurface>
    );
  }

  const total = data.retrieval_signals_total;
  const priorPct = (data.prior_rating_coverage * 100).toFixed(1);
  const feedbackPct = (data.feedback_coverage * 100).toFixed(1);
  const feedbackNum = data.feedback_coverage * 100;

  // Coverage health badge thresholds match monitoring.md runbook
  const coverageHealth =
    feedbackNum >= 20
      ? { label: t("metrics.coverageHealthy"), cls: "bg-success/10 text-success" }
      : feedbackNum >= 5
        ? { label: t("metrics.coverageLow"), cls: "bg-warning/10 text-warning" }
        : { label: t("metrics.coverageVeryLow"), cls: "bg-destructive/10 text-destructive" };

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
    <AnalyticsPanelSurface className="space-y-5">
      <div>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">
            {t("metrics.retrievalFeedback")}
          </h3>
          <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${coverageHealth.cls}`}>
            {coverageHealth.label}
          </span>
        </div>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {t("metrics.retrievalFeedbackSubtitle")}
        </p>
      </div>

      {/* Mini-cards row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <AnalyticsStatCard
          icon={<Icon name="forum" size={16} />}
          label={t("metrics.retrievalSignalsTotal")}
          value={formatNumber(total)}
        />
        <AnalyticsStatCard
          icon={<Icon name="star" size={16} />}
          label={t("metrics.priorRatingCoverage")}
          value={priorPct}
          unit="%"
        />
        <AnalyticsStatCard
          icon={<Icon name="star" size={16} />}
          label={t("metrics.ratedAfterRetrieval")}
          value={formatNumber(data.rated_after_retrieval_total)}
        />
        <AnalyticsStatCard
          icon={<Icon name="insert_chart" size={16} />}
          label={t("metrics.feedbackCoverage")}
          value={feedbackPct}
          unit="%"
        />
      </div>

      {/* Rating bucket breakdown bar */}
      {bucketTotal > 0 && (
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {t("metrics.ratingBuckets")}
          </p>
          {/* Stacked bar */}
          <div className="flex h-3 w-full overflow-hidden rounded-full bg-surface-muted">
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
                <span key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span className={`inline-block h-2 w-2 rounded-full ${BUCKET_COLORS[key]}`} />
                  {bucketLabels[key]}: {formatNumber(count)} ({pct}%)
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Retrieved-with vs without rating */}
      <div className="flex gap-4 text-xs text-muted-foreground">
        <span>
          <span className="font-semibold text-foreground">
            {formatNumber(data.retrieved_with_prior_rating)}
          </span>{" "}
          {t("metrics.retrievedWithRating")}
        </span>
        <span>
          <span className="font-semibold text-foreground">
            {formatNumber(data.retrieved_without_prior_rating)}
          </span>{" "}
          {t("metrics.retrievedWithoutRating")}
        </span>
      </div>
    </AnalyticsPanelSurface>
  );
}
