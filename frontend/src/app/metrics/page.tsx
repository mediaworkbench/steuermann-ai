"use client";

import { useEffect, useMemo, useState } from "react";
import { useAnalytics } from "@/hooks/useAnalytics";
import { useI18n } from "@/hooks/useI18n";
import { useMetrics } from "@/hooks/useMetrics";
import { useProfile } from "@/hooks/useProfile";
import { useSettings } from "@/hooks/useSettings";
import { CURRENT_USER_ID } from "@/lib/runtime";
import { RefreshCw } from "lucide-react";
import { MetricCard } from "@/components/MetricCard";
import { MemoryMetricsPanel } from "@/components/MemoryMetricsPanel";
import { MessageQualityPanel } from "@/components/MessageQualityPanel";
import { RetrievalFeedbackPanel } from "@/components/RetrievalFeedbackPanel";
import { RequestsChart } from "@/components/RequestsChart";
import { TokenUsageChart } from "@/components/TokenUsageChart";
import TokenConsumptionChart from "@/components/TokenConsumptionChart";
import UsageTrendChart from "@/components/UsageTrendChart";
import LatencyAnalysisChart from "@/components/LatencyAnalysisChart";
import MemoryTrendsChart from "@/components/MemoryTrendsChart";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { MetricsSummaryList } from "@/components/product/MetricsSummaryList";
import { MetricsTableCard } from "@/components/product/MetricsTableCard";
import { MetricsKeyValueGridCard } from "@/components/product/MetricsKeyValueGridCard";
import { MetricsStatsGrid } from "@/components/product/MetricsStatsGrid";
import { MetricsTrendChartCard } from "@/components/product/MetricsTrendChartCard";
import { type MetricsSaveStatus } from "@/components/product/MetricsSaveStatusIndicator";
import { MetricsTrendsControls } from "@/components/product/MetricsTrendsControls";
import { MetricsTrendsStatusSection } from "@/components/product/MetricsTrendsStatusSection";

export default function MetricsPage() {
  const { t, formatTime, formatNumber } = useI18n();
  const profile = useProfile();
  const userId = CURRENT_USER_ID;
  const today = useMemo(() => new Date(), []);
  const [activeTab, setActiveTab] = useState<"realtime" | "trends">("realtime");
  const { metrics, loading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useMetrics(true, 10000);
  const [isRefreshingRealTime, setIsRefreshingRealTime] = useState(false);
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
  const [saveStatus, setSaveStatus] = useState<MetricsSaveStatus>("idle");
  const { settings, saveSettings } = useSettings(userId);

  const computedDays = useMemo(() => {
    if (!startDate || !endDate) return 0;
    const start = new Date(`${startDate}T00:00:00`);
    const end = new Date(`${endDate}T00:00:00`);
    const diffDays = Math.round((end.getTime() - start.getTime()) / 86400000) + 1;
    return diffDays > 0 ? diffDays : 0;
  }, [startDate, endDate]);

  useEffect(() => {
    if (computedDays > 0 && computedDays !== days) setDays(computedDays);
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
      dateRange: { startDate, endDate },
      usage: { showRequests: showUsageRequests, showUsers: showUsageUsers },
      tokens: { showTotal: showTokenTotal, showAvg: showTokenAvg },
      latency: { showMin: showLatencyMin, showAvg: showLatencyAvg, showMax: showLatencyMax },
    }),
    [startDate, endDate, showUsageRequests, showUsageUsers, showTokenTotal, showTokenAvg, showLatencyMin, showLatencyAvg, showLatencyMax]
  );

  useEffect(() => {
    if (!settings || hasHydratedPreferences) return;
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
    if (!hasHydratedPreferences) return;
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
    memoryTrends,
    memoryRetrievalQuality,
    messageQuality,
    loading: analyticsLoading,
    error: analyticsError,
    refetch: refetchAnalytics,
  } = useAnalytics({ days, autoRefresh, refetchInterval: 60000 });

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
      ...(usageTrends?.map((row) => [row.date, row.requests, row.users]) || []),
      [],
      [t("metrics.tokenConsumption")],
      ["Date", t("metrics.totalTokens"), t("metrics.average"), t("metrics.requests")],
      ...(tokenConsumption?.map((row) => [row.date, row.total_tokens, row.avg_tokens.toFixed(2), row.requests]) || []),
      [],
      [t("metrics.latencyAnalysis")],
      ["Date", `${t("metrics.min")} (ms)`, `${t("metrics.average")} (ms)`, `${t("metrics.max")} (ms)`, t("metrics.requests")],
      ...(latencyAnalysis?.map((row) => [row.date, row.min_latency_ms.toFixed(2), row.avg_latency_ms.toFixed(2), row.max_latency_ms.toFixed(2), row.requests]) || []),
    ].map((row) => row.join(",")).join("\n");

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

  const attachments = metrics?.attachments ?? {};
  const attachmentRetries = metrics?.attachment_retries ?? {};
  const injectedTotal = attachments.injected_total ?? 0;
  const noneTotal = attachments.none_total ?? 0;
  const retryTotal = attachmentRetries.retry_total ?? 0;
  const retrySuccessTotal = attachmentRetries.retry_success_total ?? 0;
  const retrySuccessRate = retryTotal > 0 ? (retrySuccessTotal / retryTotal) * 100 : 0;
  const profileMismatchTotal = metrics?.profile_guardrails?.profile_id_mismatch_total ?? 0;
  const avgRequestDurationSeconds = Number(metrics?.latency?.avg_request_duration_seconds);
  const avgLatencyMs = Number.isFinite(avgRequestDurationSeconds) && avgRequestDurationSeconds >= 0 ? avgRequestDurationSeconds * 1000 : null;
  const validLatencySeries = (latencyAnalysis ?? []).filter((entry) => Number.isFinite(entry.avg_latency_ms));
  const trendsAvgLatencyText = validLatencySeries.length > 0
    ? `${(validLatencySeries.reduce((sum, row) => sum + row.avg_latency_ms, 0) / validLatencySeries.length).toFixed(1)}ms`
    : t("metrics.na");

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("metrics.title", { app: profile.appName ?? "AI" })}</h1>
        </div>
      </div>
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "realtime" | "trends")} className="w-fit">
        <TabsList aria-label={t("metrics.sectionsLabel")}>
          <TabsTrigger value="realtime">{t("metrics.realtimeTab")}</TabsTrigger>
          <TabsTrigger value="trends">{t("metrics.trendsTab")}</TabsTrigger>
        </TabsList>
      </Tabs>

      {activeTab === "realtime" ? (
        <section className="rounded-xl border border-border bg-surface px-6 py-6 shadow-sm mt-4">
          <h2 className="mb-4 text-xl font-bold text-foreground">{t("metrics.realtimeTitle")}</h2>
          <div className="mb-6 flex justify-end">
            <Button onClick={handleRefreshRealTime} disabled={isRefreshingRealTime} variant="primary" size="md">
              <RefreshCw size={18} className={isRefreshingRealTime ? "animate-spin" : ""} />
              {t("common.refresh")}
            </Button>
          </div>

          {metricsError ? <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
            <p className="mb-1 font-semibold">{t("common.error")}</p>
            <p>{metricsError}</p>
          </div> : null}

          {metricsLoading && !metrics ? (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <RefreshCw size={32} className="animate-spin" />
              <p className="mt-4 mb-0">{t("metrics.loadingMetrics")}</p>
            </div>
          ) : metrics ? (
            <>
              <MetricsStatsGrid>
                <MetricCard label={t("metrics.totalRequests")} value={Object.values(metrics.requests).reduce((a, b) => a + b, 0)} />
                <MetricCard label={t("metrics.totalTokens")} value={Math.floor(Object.values(metrics.tokens).reduce((a, b) => a + b, 0))} unit="tokens" />
                <MetricCard label={t("metrics.avgLatency")} value={avgLatencyMs !== null ? avgLatencyMs.toFixed(2) : t("metrics.na")} unit={avgLatencyMs !== null ? "ms" : ""} />
                <MetricCard label={t("metrics.activeSessions")} value={Object.values(metrics.sessions).reduce((a, b) => a + b, 0)} />
                <MetricCard label={t("metrics.attachmentsInjected")} value={Math.floor(injectedTotal)} />
                <MetricCard label={t("metrics.requestsWithoutAttachments")} value={Math.floor(noneTotal)} />
                <MetricCard label={t("metrics.attachmentRetryTriggers")} value={Math.floor(retryTotal)} />
                <MetricCard label={t("metrics.attachmentRetrySuccess")} value={retrySuccessRate.toFixed(1)} unit="%" />
                <MetricCard label={t("metrics.profileIdMismatches")} value={Math.floor(profileMismatchTotal)} />
              </MetricsStatsGrid>

              <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
                <TokenUsageChart data={metrics.tokens} />
                <RequestsChart data={metrics.requests} />
              </div>

              {Object.keys(metrics.memory_ops).length > 0 ? <MemoryMetricsPanel metrics={metrics} formatNumber={formatNumber} /> : null}

              {Object.keys(metrics.llm_calls).length > 0 ? (
                <MetricsTableCard
                  title={t("metrics.llmCallsByProvider")}
                  labelHeader={t("metrics.providerModelStatus")}
                  valueHeader={t("metrics.count")}
                  rows={Object.entries(metrics.llm_calls).map(([key, value]) => ({ label: key, value: typeof value === "number" ? formatNumber(Math.round(value)) : 0 }))}
                />
              ) : null}

              {Object.keys(attachments).length > 0 ? (
                <MetricsKeyValueGridCard
                  title={t("metrics.attachmentContextMetrics")}
                  items={Object.entries(attachments).map(([key, value]) => ({ keyLabel: key, value: typeof value === "number" ? formatNumber(Math.round(value)) : 0 }))}
                />
              ) : null}

              {Object.keys(attachmentRetries).length > 0 ? (
                <MetricsKeyValueGridCard
                  title={t("metrics.attachmentRetryGuardrail")}
                  items={Object.entries(attachmentRetries).map(([key, value]) => ({ keyLabel: key, value: typeof value === "number" ? formatNumber(Math.round(value)) : 0 }))}
                />
              ) : null}
            </>
          ) : null}
        </section>
      ) : (
        <section className="rounded-xl border border-border bg-surface px-6 py-6 shadow-sm mt-4">
          <h2 className="mb-4 text-xl font-bold text-foreground">{t("metrics.trendsTitle")}</h2>
          <MetricsTrendsControls
            dateRangeLabel={t("metrics.dateRange")}
            startDate={startDate}
            endDate={endDate}
            onStartDateChange={setStartDate}
            onEndDateChange={setEndDate}
            maxEndDate={toInputDate(today)}
            dayCount={computedDays}
            daysLabel={t("metrics.days")}
            isRangeInvalid={isRangeInvalid}
            invalidRangeLabel={t("metrics.invalidRange")}
            presets={[
              { key: "last-7", label: t("metrics.last7Days"), days: 7 },
              { key: "last-30", label: t("metrics.last30Days"), days: 30 },
              { key: "last-90", label: t("metrics.last90Days"), days: 90 },
            ]}
            onSelectPreset={(daysToShow) => applyPreset(daysToShow, setStartDate, setEndDate)}
            onRefresh={handleRefreshTrends}
            refreshDisabled={analyticsLoading}
            refreshLabel={t("common.refresh")}
            onExport={handleExportCSV}
            exportDisabled={analyticsLoading || !usageTrends || isRangeInvalid}
            exportLabel={t("metrics.export")}
            autoRefresh={autoRefresh}
            onAutoRefreshChange={setAutoRefresh}
            autoRefreshLabel={t("metrics.autoRefresh60s")}
            saveStatus={saveStatus}
            savePreferencesLabel={t("metrics.savePreferences")}
            preferencesSavedLabel={t("metrics.preferencesSaved")}
            preferencesFailedLabel={t("metrics.preferencesFailed")}
          />

          <MetricsTrendsStatusSection
            lastUpdatedLabel={t("metrics.lastUpdated")}
            lastUpdatedValue={isMounted ? formatTime(lastRefresh) : "--:--:--"}
            errorTitle={t("metrics.errorLoadingTrends")}
            errorMessage={analyticsError}
            isLoading={analyticsLoading}
            loadingLabel={t("metrics.loadingTrends")}
          >
            <div className="mb-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
              <MetricsTrendChartCard
                title={t("metrics.usageTrends")}
                subtitle={t("metrics.dailyRequestsAndUsers")}
                options={[
                  { key: "usage-requests", label: t("metrics.requests"), checked: showUsageRequests, onToggle: setShowUsageRequests },
                  { key: "usage-users", label: t("metrics.users"), checked: showUsageUsers, onToggle: setShowUsageUsers },
                ]}
                hasData={Boolean(usageTrends && usageTrends.length > 0)}
                hasSelection={hasUsageSeries}
                emptyDataMessage={t("metrics.noUsageData")}
                emptySelectionMessage={t("metrics.selectAtLeastOneMetric")}
              >
                <UsageTrendChart data={usageTrends ?? []} showRequests={showUsageRequests} showUsers={showUsageUsers} />
              </MetricsTrendChartCard>

              <MetricsTrendChartCard
                title={t("metrics.tokenConsumption")}
                subtitle={t("metrics.totalAndAverageTokens")}
                options={[
                  { key: "token-total", label: t("metrics.total"), checked: showTokenTotal, onToggle: setShowTokenTotal },
                  { key: "token-avg", label: t("metrics.average"), checked: showTokenAvg, onToggle: setShowTokenAvg },
                ]}
                hasData={Boolean(tokenConsumption && tokenConsumption.length > 0)}
                hasSelection={hasTokenSeries}
                emptyDataMessage={t("metrics.noTokenData")}
                emptySelectionMessage={t("metrics.selectAtLeastOneMetric")}
              >
                <TokenConsumptionChart data={tokenConsumption ?? []} showTotalTokens={showTokenTotal} showAvgTokens={showTokenAvg} />
              </MetricsTrendChartCard>

              <MetricsTrendChartCard
                title={t("metrics.latencyAnalysis")}
                subtitle={t("metrics.minAverageMaxDuration")}
                options={[
                  { key: "latency-min", label: t("metrics.min"), checked: showLatencyMin, onToggle: setShowLatencyMin },
                  { key: "latency-avg", label: t("metrics.average"), checked: showLatencyAvg, onToggle: setShowLatencyAvg },
                  { key: "latency-max", label: t("metrics.max"), checked: showLatencyMax, onToggle: setShowLatencyMax },
                ]}
                hasData={Boolean(latencyAnalysis && latencyAnalysis.length > 0)}
                hasSelection={hasLatencySeries}
                emptyDataMessage={t("metrics.noLatencyData")}
                emptySelectionMessage={t("metrics.selectAtLeastOneMetric")}
              >
                <LatencyAnalysisChart data={latencyAnalysis ?? []} showMin={showLatencyMin} showAvg={showLatencyAvg} showMax={showLatencyMax} />
              </MetricsTrendChartCard>

              <MetricsTrendChartCard
                title={t("metrics.memoryTrends")}
                subtitle={t("metrics.dailyMemoryOpsQualityAndErrorRate")}
                hasData={Boolean(memoryTrends && memoryTrends.length > 0)}
                emptyDataMessage={t("metrics.noMemoryTrendData")}
              >
                <MemoryTrendsChart data={memoryTrends ?? []} />
              </MetricsTrendChartCard>

              <RetrievalFeedbackPanel data={memoryRetrievalQuality} formatNumber={formatNumber} />
              <MessageQualityPanel data={messageQuality} formatNumber={formatNumber} />

              <MetricsTrendChartCard
                title={t("metrics.summary")}
                subtitle={t("metrics.keyMetricsForPeriod")}
                hasData={Boolean(usageTrends && usageTrends.length > 0)}
                emptyDataMessage={t("metrics.noSummaryData")}
              >
                <MetricsSummaryList
                  rows={[
                    { label: t("metrics.totalRequests"), value: formatNumber(usageTrends?.reduce((sum, trend) => sum + trend.requests, 0) ?? 0) },
                    { label: t("metrics.uniqueUsers"), value: formatNumber(usageTrends?.reduce((sum, trend) => sum + trend.users, 0) ?? 0) },
                    { label: t("metrics.totalTokens"), value: tokenConsumption ? `${(tokenConsumption.reduce((sum, token) => sum + token.total_tokens, 0) / 1000).toFixed(1)}K` : t("metrics.na") },
                    { label: t("metrics.avgLatency"), value: trendsAvgLatencyText },
                    { label: t("metrics.memoryLoads"), value: memoryTrends ? formatNumber(Math.round(memoryTrends.reduce((sum, day) => sum + day.loads, 0))) : t("metrics.na") },
                    { label: t("metrics.memoryUpdates"), value: memoryTrends ? formatNumber(Math.round(memoryTrends.reduce((sum, day) => sum + day.updates, 0))) : t("metrics.na") },
                    { label: t("metrics.memoryErrorRate"), value: memoryTrends && memoryTrends.length > 0 ? `${(memoryTrends.reduce((sum, day) => sum + day.error_rate, 0) / memoryTrends.length).toFixed(2)}%` : t("metrics.na") },
                  ]}
                />
              </MetricsTrendChartCard>
            </div>
          </MetricsTrendsStatusSection>
        </section>
      )}
      </div>
    </div>
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

function applyPreset(daysToShow: number, setStartDate: (value: string) => void, setEndDate: (value: string) => void): void {
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
        ? { startDate: (dateRange as { startDate: string }).startDate, endDate: (dateRange as { endDate: string }).endDate }
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
