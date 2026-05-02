"use client";

import { useState, useEffect, useMemo } from "react";
import { RefreshCw, Download } from "lucide-react";
import { useMetrics } from "@/hooks/useMetrics";
import { useAnalytics } from "@/hooks/useAnalytics";
import { useSettings } from "@/hooks/useSettings";
import { useI18n } from "@/hooks/useI18n";
import { MetricCard } from "@/components/MetricCard";
import { TokenUsageChart } from "@/components/TokenUsageChart";
import { RequestsChart } from "@/components/RequestsChart";
import UsageTrendChart from "@/components/UsageTrendChart";
import TokenConsumptionChart from "@/components/TokenConsumptionChart";
import LatencyAnalysisChart from "@/components/LatencyAnalysisChart";
import { MemoryMetricsPanel } from "@/components/MemoryMetricsPanel";
import { CURRENT_USER_ID } from "@/lib/runtime";
import styles from "./Metrics.module.css";

export default function MetricsPage() {
  const { t, formatTime, formatNumber } = useI18n();
  const userId = CURRENT_USER_ID;
  const today = useMemo(() => new Date(), []);
  const [activeTab, setActiveTab] = useState<"realtime" | "trends">("realtime");
  
  // Real-time tab state
  const { metrics, loading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useMetrics(true, 10000);
  const [isRefreshingRealTime, setIsRefreshingRealTime] = useState(false);

  // Trends tab state
  const [days, setDays] = useState(30);
  const [startDate, setStartDate] = useState(() => toInputDate(addDays(today, -29)));
  const [endDate, setEndDate] = useState(() => toInputDate(today));
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [isMounted, setIsMounted] = useState(false);
  const [showUsageRequests, setShowUsageRequests] = useState(true);
  const [showUsageUsers, setShowUsageUsers] = useState(true);
  const [showTokenTotal, setShowTokenTotal] = useState(true);
  const [showTokenAvg, setShowTokenAvg] = useState(true);
  const [showLatencyMin, setShowLatencyMin] = useState(true);
  const [showLatencyAvg, setShowLatencyAvg] = useState(true);
  const [showLatencyMax, setShowLatencyMax] = useState(true);
  const [hasHydratedPreferences, setHasHydratedPreferences] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  const { settings, saveSettings } = useSettings(userId);

  // Compute date range
  const computedDays = useMemo(() => {
    if (!startDate || !endDate) return 0;
    const start = new Date(`${startDate}T00:00:00`);
    const end = new Date(`${endDate}T00:00:00`);
    const diffDays = Math.round((end.getTime() - start.getTime()) / 86400000) + 1;
    return diffDays > 0 ? diffDays : 0;
  }, [startDate, endDate]);

  useEffect(() => {
    if (computedDays > 0 && computedDays !== days) {
      setDays(computedDays);
    }
  }, [computedDays, days]);

  const isRangeInvalid = computedDays === 0;
  const hasUsageSeries = showUsageRequests || showUsageUsers;
  const hasTokenSeries = showTokenTotal || showTokenAvg;
  const hasLatencySeries = showLatencyMin || showLatencyAvg || showLatencyMax;

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const analyticsPreferences = useMemo<AnalyticsPreferences>(
    () => ({
      dateRange: {
        startDate,
        endDate,
      },
      usage: {
        showRequests: showUsageRequests,
        showUsers: showUsageUsers,
      },
      tokens: {
        showTotal: showTokenTotal,
        showAvg: showTokenAvg,
      },
      latency: {
        showMin: showLatencyMin,
        showAvg: showLatencyAvg,
        showMax: showLatencyMax,
      },
    }),
    [
      startDate,
      endDate,
      showUsageRequests,
      showUsageUsers,
      showTokenTotal,
      showTokenAvg,
      showLatencyMin,
      showLatencyAvg,
      showLatencyMax,
    ]
  );

  useEffect(() => {
    if (!settings || hasHydratedPreferences) {
      return;
    }
    const preferences = normalizeAnalyticsPreferences(settings.analytics_preferences);

    if (preferences.dateRange) {
      setStartDate(preferences.dateRange.startDate);
      setEndDate(preferences.dateRange.endDate);
    }

    setShowUsageRequests(preferences.usage.showRequests);
    setShowUsageUsers(preferences.usage.showUsers);
    setShowTokenTotal(preferences.tokens.showTotal);
    setShowTokenAvg(preferences.tokens.showAvg);
    setShowLatencyMin(preferences.latency.showMin);
    setShowLatencyAvg(preferences.latency.showAvg);
    setShowLatencyMax(preferences.latency.showMax);
    setHasHydratedPreferences(true);
  }, [settings, hasHydratedPreferences]);

  useEffect(() => {
    if (!hasHydratedPreferences) {
      return;
    }
    setSaveStatus("saving");
    const timer = setTimeout(() => {
      void saveSettings({ analytics_preferences: analyticsPreferences })
        .then(() => {
          setSaveStatus("saved");
          setTimeout(() => setSaveStatus("idle"), 2000);
        })
        .catch(() => {
          setSaveStatus("error");
          setTimeout(() => setSaveStatus("idle"), 3000);
        });
    }, 500);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analyticsPreferences, hasHydratedPreferences]);

  const {
    usageTrends,
    tokenConsumption,
    latencyAnalysis,
    loading: analyticsLoading,
    error: analyticsError,
    refetch: refetchAnalytics,
  } = useAnalytics({
    days,
    autoRefresh,
    refetchInterval: 60000,
  });

  const handleRefreshRealTime = async () => {
    setIsRefreshingRealTime(true);
    await refetchMetrics();
    setIsRefreshingRealTime(false);
  };

  const handleRefreshTrends = async () => {
    await refetchAnalytics();
    setLastRefresh(new Date());
  };

  const handleExportCSV = () => {
    const csvContent = [
      [t("metrics.export"), new Date().toISOString()],
      [],
      [t("metrics.usageTrends")],
      ["Date", t("metrics.requests"), t("metrics.users")],
      ...(usageTrends?.map((t) => [t.date, t.requests, t.users]) || []),
      [],
      [t("metrics.tokenConsumption")],
      ["Date", t("metrics.totalTokens"), t("metrics.average"), t("metrics.requests")],
      ...(tokenConsumption?.map((t) => [t.date, t.total_tokens, t.avg_tokens.toFixed(2), t.requests]) || []),
      [],
      [t("metrics.latencyAnalysis")],
      ["Date", `${t("metrics.min")} (ms)`, `${t("metrics.average")} (ms)`, `${t("metrics.max")} (ms)`, t("metrics.requests")],
      ...(latencyAnalysis?.map((l) => [
        l.date,
        l.min_latency_ms.toFixed(2),
        l.avg_latency_ms.toFixed(2),
        l.max_latency_ms.toFixed(2),
        l.requests,
      ]) || []),
    ]
      .map((row) => row.join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `metrics-export-${new Date().toISOString().split("T")[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  // Metrics page data
  const attachments = metrics?.attachments ?? {};
  const attachmentRetries = metrics?.attachment_retries ?? {};
  const injectedTotal = attachments.injected_total ?? 0;
  const noneTotal = attachments.none_total ?? 0;
  const retryTotal = attachmentRetries.retry_total ?? 0;
  const retrySuccessTotal = attachmentRetries.retry_success_total ?? 0;
  const retrySuccessRate = retryTotal > 0 ? (retrySuccessTotal / retryTotal) * 100 : 0;
  const profileMismatchTotal = metrics?.profile_guardrails?.profile_id_mismatch_total ?? 0;
  const avgRequestDurationSeconds = Number(metrics?.latency?.avg_request_duration_seconds);
  const avgLatencyMs = Number.isFinite(avgRequestDurationSeconds) && avgRequestDurationSeconds >= 0
    ? avgRequestDurationSeconds * 1000
    : null;

  const validLatencySeries = (latencyAnalysis ?? []).filter((entry) =>
    Number.isFinite(entry.avg_latency_ms)
  );
  const trendsAvgLatencyText = validLatencySeries.length > 0
    ? `${(validLatencySeries.reduce((sum, latency) => sum + latency.avg_latency_ms, 0) / validLatencySeries.length).toFixed(1)}ms`
    : t("metrics.na");

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={styles.title}>System performance and analytics</h1>
      </div>

      <div className={styles.tabBar} role="tablist" aria-label={t("metrics.sectionsLabel")}>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "realtime"}
          className={`${styles.tabButton} ${activeTab === "realtime" ? styles.tabButtonActive : ""}`}
          onClick={() => setActiveTab("realtime")}
        >
          {t("metrics.realtimeTab")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "trends"}
          className={`${styles.tabButton} ${activeTab === "trends" ? styles.tabButtonActive : ""}`}
          onClick={() => setActiveTab("trends")}
        >
          {t("metrics.trendsTab")}
        </button>
      </div>

      {activeTab === "realtime" ? (
      <section className={styles.tabPanel}>
        <h2 className={styles.sectionTitle}>{t("metrics.realtimeTitle")}</h2>
        <>
          <div className={styles.controls}>
            <button
              onClick={handleRefreshRealTime}
              disabled={isRefreshingRealTime}
              className={styles.refreshButton}
            >
              <RefreshCw size={18} className={isRefreshingRealTime ? styles.spin : ""} />
              {t("common.refresh")}
            </button>
          </div>

          {metricsError && (
            <div className={styles.error}>
              <p className={styles.errorTitle}>{t("common.error")}</p>
              <p>{metricsError}</p>
            </div>
          )}

          {metricsLoading && !metrics ? (
            <div className={styles.loading}>
              <RefreshCw size={32} className={styles.spin} />
              <p>{t("metrics.loadingMetrics")}</p>
            </div>
          ) : metrics ? (
            <>
              <div className={styles.statsGrid}>
                <MetricCard
                  label={t("metrics.totalRequests")}
                  value={Object.values(metrics.requests).reduce((a, b) => a + b, 0)}
                  icon="📊"
                />
                <MetricCard
                  label={t("metrics.totalTokens")}
                  value={Math.floor(Object.values(metrics.tokens).reduce((a, b) => a + b, 0))}
                  unit="tokens"
                  icon="🎯"
                />
                <MetricCard
                  label={t("metrics.avgLatency")}
                  value={avgLatencyMs !== null ? avgLatencyMs.toFixed(2) : t("metrics.na")}
                  unit={avgLatencyMs !== null ? "ms" : ""}
                  icon="⚡"
                />
                <MetricCard
                  label={t("metrics.activeSessions")}
                  value={Object.values(metrics.sessions).reduce((a, b) => a + b, 0)}
                  icon="👥"
                />
                <MetricCard
                  label={t("metrics.attachmentsInjected")}
                  value={Math.floor(injectedTotal)}
                  icon="📎"
                />
                <MetricCard
                  label={t("metrics.requestsWithoutAttachments")}
                  value={Math.floor(noneTotal)}
                  icon="🗂️"
                />
                <MetricCard
                  label={t("metrics.attachmentRetryTriggers")}
                  value={Math.floor(retryTotal)}
                  icon="♻️"
                />
                <MetricCard
                  label={t("metrics.attachmentRetrySuccess")}
                  value={retrySuccessRate.toFixed(1)}
                  unit="%"
                  icon="✅"
                />
                <MetricCard
                  label={t("metrics.profileIdMismatches")}
                  value={Math.floor(profileMismatchTotal)}
                  icon="🧭"
                />
              </div>

              <div className={styles.chartsGrid}>
                <TokenUsageChart data={metrics.tokens} />
                <RequestsChart data={metrics.requests} />
              </div>

              {Object.keys(metrics.memory_ops).length > 0 && (
                <MemoryMetricsPanel metrics={metrics} formatNumber={formatNumber} />
              )}

              {Object.keys(metrics.llm_calls).length > 0 && (
                <div className={styles.card}>
                  <h3 className={styles.cardTitle}>{t("metrics.llmCallsByProvider")}</h3>
                  <div className={styles.tableWrapper}>
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>{t("metrics.providerModelStatus")}</th>
                          <th>{t("metrics.count")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(metrics.llm_calls).map(([key, value]) => (
                          <tr key={key}>
                            <td>{key}</td>
                            <td className={styles.numCell}>
                              {typeof value === "number" ? formatNumber(Math.round(value)) : 0}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {Object.keys(attachments).length > 0 && (
                <div className={styles.card}>
                  <h3 className={styles.cardTitle}>{t("metrics.attachmentContextMetrics")}</h3>
                  <div className={styles.opsGrid}>
                    {Object.entries(attachments).map(([key, value]) => (
                      <div key={key} className={styles.opItem}>
                        <p className={styles.opLabel}>{key}</p>
                        <p className={styles.opValue}>
                          {typeof value === "number" ? formatNumber(Math.round(value)) : 0}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {Object.keys(attachmentRetries).length > 0 && (
                <div className={styles.card}>
                  <h3 className={styles.cardTitle}>{t("metrics.attachmentRetryGuardrail")}</h3>
                  <div className={styles.opsGrid}>
                    {Object.entries(attachmentRetries).map(([key, value]) => (
                      <div key={key} className={styles.opItem}>
                        <p className={styles.opLabel}>{key}</p>
                        <p className={styles.opValue}>
                          {typeof value === "number" ? formatNumber(Math.round(value)) : 0}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : null}
        </>
      </section>
      ) : (

      <section className={styles.tabPanel}>
        <h2 className={styles.sectionTitle}>{t("metrics.trendsTitle")}</h2>
        <>
          <div className={styles.trendControls}>
            <div className={styles.controlGroup}>
              <label className={styles.label}>{t("metrics.dateRange")}</label>
              <div className={styles.dateInputs}>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  max={endDate}
                  className={styles.dateInput}
                />
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  min={startDate}
                  max={toInputDate(today)}
                  className={styles.dateInput}
                />
                <span className={styles.dayCount}>
                  {computedDays} {t("metrics.days")}
                  {isRangeInvalid && <span className={styles.invalidRange}>{t("metrics.invalidRange")}</span>}
                </span>
              </div>
              <div className={styles.presets}>
                <button type="button" onClick={() => applyPreset(7, setStartDate, setEndDate)} className={styles.presetBtn}>
                  {t("metrics.last7Days")}
                </button>
                <button type="button" onClick={() => applyPreset(30, setStartDate, setEndDate)} className={styles.presetBtn}>
                  {t("metrics.last30Days")}
                </button>
                <button type="button" onClick={() => applyPreset(90, setStartDate, setEndDate)} className={styles.presetBtn}>
                  {t("metrics.last90Days")}
                </button>
              </div>
            </div>

            <div className={styles.actions}>
              <button onClick={handleRefreshTrends} disabled={analyticsLoading} className={styles.actionBtn}>
                <RefreshCw className={`${styles.icon} ${analyticsLoading ? styles.spin : ""}`} />
                {t("common.refresh")}
              </button>
              <button onClick={handleExportCSV} disabled={analyticsLoading || !usageTrends || isRangeInvalid} className={styles.actionBtn}>
                <Download className={styles.icon} />
                {t("metrics.export")}
              </button>
              <label className={styles.checkboxLabel}>
                <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
                {t("metrics.autoRefresh60s")}
              </label>
            </div>

            {saveStatus !== "idle" && (
              <div className={styles.saveStatus}>
                {saveStatus === "saving" && <><div className={styles.spinner} /><span>{t("metrics.savePreferences")}</span></>}
                {saveStatus === "saved" && <><span className={styles.checkIcon}>✓</span><span>{t("metrics.preferencesSaved")}</span></>}
                {saveStatus === "error" && <><span className={styles.xIcon}>✕</span><span>{t("metrics.preferencesFailed")}</span></>}
              </div>
            )}
          </div>

          <div className={styles.lastRefresh}>
            {t("metrics.lastUpdated")}: {isMounted ? formatTime(lastRefresh) : "--:--:--"}
          </div>

          {analyticsError && (
            <div className={styles.error}>
              <p>{t("metrics.errorLoadingTrends")}: {analyticsError}</p>
            </div>
          )}

          {analyticsLoading && (
            <div className={styles.loading}>
              <div className={styles.loadingSpinner} />
              <p>{t("metrics.loadingTrends")}</p>
            </div>
          )}

          {!analyticsLoading && (
            <div className={styles.chartsGrid}>
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>{t("metrics.usageTrends")}</h3>
                <p className={styles.chartSubtitle}>{t("metrics.dailyRequestsAndUsers")}</p>
                <div className={styles.chartOptions}>
                  <label><input type="checkbox" checked={showUsageRequests} onChange={(e) => setShowUsageRequests(e.target.checked)} /> {t("metrics.requests")}</label>
                  <label><input type="checkbox" checked={showUsageUsers} onChange={(e) => setShowUsageUsers(e.target.checked)} /> {t("metrics.users")}</label>
                </div>
                {usageTrends && usageTrends.length > 0 ? (
                  hasUsageSeries ? <UsageTrendChart data={usageTrends} showRequests={showUsageRequests} showUsers={showUsageUsers} /> : <p>{t("metrics.selectAtLeastOneMetric")}</p>
                ) : <p>{t("metrics.noUsageData")}</p>}
              </div>

              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>{t("metrics.tokenConsumption")}</h3>
                <p className={styles.chartSubtitle}>{t("metrics.totalAndAverageTokens")}</p>
                <div className={styles.chartOptions}>
                  <label><input type="checkbox" checked={showTokenTotal} onChange={(e) => setShowTokenTotal(e.target.checked)} /> {t("metrics.total")}</label>
                  <label><input type="checkbox" checked={showTokenAvg} onChange={(e) => setShowTokenAvg(e.target.checked)} /> {t("metrics.average")}</label>
                </div>
                {tokenConsumption && tokenConsumption.length > 0 ? (
                  hasTokenSeries ? <TokenConsumptionChart data={tokenConsumption} showTotalTokens={showTokenTotal} showAvgTokens={showTokenAvg} /> : <p>{t("metrics.selectAtLeastOneMetric")}</p>
                ) : <p>{t("metrics.noTokenData")}</p>}
              </div>

              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>{t("metrics.latencyAnalysis")}</h3>
                <p className={styles.chartSubtitle}>{t("metrics.minAverageMaxDuration")}</p>
                <div className={styles.chartOptions}>
                  <label><input type="checkbox" checked={showLatencyMin} onChange={(e) => setShowLatencyMin(e.target.checked)} /> {t("metrics.min")}</label>
                  <label><input type="checkbox" checked={showLatencyAvg} onChange={(e) => setShowLatencyAvg(e.target.checked)} /> {t("metrics.average")}</label>
                  <label><input type="checkbox" checked={showLatencyMax} onChange={(e) => setShowLatencyMax(e.target.checked)} /> {t("metrics.max")}</label>
                </div>
                {latencyAnalysis && latencyAnalysis.length > 0 ? (
                  hasLatencySeries ? <LatencyAnalysisChart data={latencyAnalysis} showMin={showLatencyMin} showAvg={showLatencyAvg} showMax={showLatencyMax} /> : <p>{t("metrics.selectAtLeastOneMetric")}</p>
                ) : <p>{t("metrics.noLatencyData")}</p>}
              </div>

              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>{t("metrics.summary")}</h3>
                <p className={styles.chartSubtitle}>{t("metrics.keyMetricsForPeriod")}</p>
                {usageTrends && usageTrends.length > 0 ? (
                  <div className={styles.summaryGrid}>
                    <div className={styles.summaryRow}><span>{t("metrics.totalRequests")}</span><span>{formatNumber(usageTrends.reduce((sum, trend) => sum + trend.requests, 0))}</span></div>
                    <div className={styles.summaryRow}><span>{t("metrics.uniqueUsers")}</span><span>{formatNumber(usageTrends.reduce((sum, trend) => sum + trend.users, 0))}</span></div>
                    <div className={styles.summaryRow}><span>{t("metrics.totalTokens")}</span><span>{tokenConsumption ? (tokenConsumption.reduce((sum, token) => sum + token.total_tokens, 0) / 1000).toFixed(1) + "K" : t("metrics.na")}</span></div>
                    <div className={styles.summaryRow}><span>{t("metrics.avgLatency")}</span><span>{trendsAvgLatencyText}</span></div>
                  </div>
                ) : <p>{t("metrics.noSummaryData")}</p>}
              </div>
            </div>
          )}
        </>
      </section>
      )}
    </main>
  );
}

function toInputDate(date: Date): string {
  const offsetMs = date.getTimezoneOffset() * 60000;
  return new Date(date.getTime() - offsetMs).toISOString().slice(0, 10);
}

function addDays(date: Date, daysToAdd: number): Date {
  const next = new Date(date);
  next.setDate(next.getDate() + daysToAdd);
  return next;
}

function applyPreset(
  daysToShow: number,
  setStartDate: (value: string) => void,
  setEndDate: (value: string) => void
): void {
  const end = new Date();
  const start = addDays(end, -(daysToShow - 1));
  setStartDate(toInputDate(start));
  setEndDate(toInputDate(end));
}

type AnalyticsPreferences = {
  dateRange?: { startDate: string; endDate: string };
  usage: { showRequests: boolean; showUsers: boolean };
  tokens: { showTotal: boolean; showAvg: boolean };
  latency: { showMin: boolean; showAvg: boolean; showMax: boolean };
};

function normalizeAnalyticsPreferences(preferences: Record<string, unknown> | undefined): AnalyticsPreferences {
  const dateRange = (preferences?.dateRange as Record<string, unknown> | undefined) ?? {};
  const usage = (preferences?.usage as Record<string, unknown> | undefined) ?? {};
  const tokens = (preferences?.tokens as Record<string, unknown> | undefined) ?? {};
  const latency = (preferences?.latency as Record<string, unknown> | undefined) ?? {};

  return {
    dateRange:
      typeof (dateRange as { startDate?: string })?.startDate === "string" &&
      typeof (dateRange as { endDate?: string })?.endDate === "string"
        ? {
            startDate: (dateRange as { startDate: string }).startDate,
            endDate: (dateRange as { endDate: string }).endDate,
          }
        : undefined,
    usage: {
      showRequests: typeof (usage as { showRequests?: boolean })?.showRequests === "boolean" ? (usage as { showRequests: boolean }).showRequests : true,
      showUsers: typeof (usage as { showUsers?: boolean })?.showUsers === "boolean" ? (usage as { showUsers: boolean }).showUsers : true,
    },
    tokens: {
      showTotal: typeof (tokens as { showTotal?: boolean })?.showTotal === "boolean" ? (tokens as { showTotal: boolean }).showTotal : true,
      showAvg: typeof (tokens as { showAvg?: boolean })?.showAvg === "boolean" ? (tokens as { showAvg: boolean }).showAvg : true,
    },
    latency: {
      showMin: typeof (latency as { showMin?: boolean })?.showMin === "boolean" ? (latency as { showMin: boolean }).showMin : true,
      showAvg: typeof (latency as { showAvg?: boolean })?.showAvg === "boolean" ? (latency as { showAvg: boolean }).showAvg : true,
      showMax: typeof (latency as { showMax?: boolean })?.showMax === "boolean" ? (latency as { showMax: boolean }).showMax : true,
    },
  };
}
