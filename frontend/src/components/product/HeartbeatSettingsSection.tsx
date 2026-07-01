"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  fetchHeartbeatRate,
  fetchHeartbeatRuns,
  fetchHeartbeatTasks,
  updateHeartbeatRate,
  updateHeartbeatCooldown,
  type HeartbeatRateConfig,
  type HeartbeatRun,
  type HeartbeatTaskInfo,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

const MIN_RATE = 1;
const MAX_RATE = 1440;
const MIN_COOLDOWN = 0;
const MAX_COOLDOWN = 604800; // 7 days
const LOG_WINDOW_HOURS = 24; // run log shows the last 24h
const PAGE_SIZE = 25; // ...paginated 25 rows per page

/** Compact human label for a seconds duration (e.g. 86400 → "24h", 300 → "5m"). */
function humanizeSeconds(total: number): string {
  if (total <= 0) return "0s";
  const d = Math.floor(total / 86400);
  const h = Math.floor((total % 86400) / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return [d && `${d}d`, h && `${h}h`, m && `${m}m`, s && `${s}s`].filter(Boolean).join(" ");
}

const SELECT_CLASS =
  "rounded-lg border border-border px-3 py-1.5 text-sm text-foreground bg-surface focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary";

function statusVariant(status: string): "secondary" | "destructive" | "outline" {
  if (status === "error") return "destructive";
  if (status === "skipped") return "outline";
  return "secondary";
}

/**
 * Admin heartbeat panel (rendered on its own `/admin/heartbeat` page): the global
 * beat rate, the configured tasks (scope + last run), and the run log. Global
 * tasks run once per beat; per-user tasks fan out once per active user, so the
 * log shows one row per user for them. The log loads the last {@link LOG_WINDOW_HOURS}h
 * of beats (newest first, server-capped), is filtered client-side by task, user, and
 * status, and is paginated {@link PAGE_SIZE} rows per page. The rate is a
 * deployment-wide setting persisted in Postgres; the LangGraph-embedded scheduler
 * applies a change within ~30s.
 */
export function HeartbeatSettingsSection() {
  const { t, formatRelativeTime } = useI18n();
  const [config, setConfig] = useState<HeartbeatRateConfig | null>(null);
  const [rate, setRate] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const [tasks, setTasks] = useState<HeartbeatTaskInfo[]>([]);
  const [cooldownDraft, setCooldownDraft] = useState<Record<string, string>>({});
  const [savingCooldown, setSavingCooldown] = useState<string | null>(null);
  const [runs, setRuns] = useState<HeartbeatRun[]>([]);
  const [logLoading, setLogLoading] = useState(true);
  const [taskFilter, setTaskFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchHeartbeatRate(), fetchHeartbeatTasks()]).then(([rateData, taskData]) => {
      if (cancelled) return;
      if (!rateData) {
        toast.error(t("adminPage.heartbeatLoadFailed"));
      } else {
        setConfig(rateData);
        setRate(String(rateData.heartbeat_rate_minutes));
      }
      if (taskData) {
        setTasks(taskData);
        setCooldownDraft(Object.fromEntries(taskData.map((tk) => [tk.name, String(tk.cooldown_seconds)])));
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const saveCooldown = useCallback(
    async (taskName: string) => {
      const parsed = Number(cooldownDraft[taskName]);
      if (!Number.isFinite(parsed) || parsed < MIN_COOLDOWN || parsed > MAX_COOLDOWN) {
        toast.error(t("adminPage.heartbeatCooldownInvalid", { max: MAX_COOLDOWN }));
        return;
      }
      setSavingCooldown(taskName);
      const updated = await updateHeartbeatCooldown(taskName, Math.floor(parsed));
      setSavingCooldown(null);
      if (!updated) {
        toast.error(t("adminPage.heartbeatCooldownSaveFailed"));
        return;
      }
      setTasks(updated);
      setCooldownDraft((prev) => ({ ...prev, [taskName]: String(parsed) }));
      toast.success(t("adminPage.heartbeatCooldownSaved"));
    },
    [cooldownDraft, t],
  );

  const loadRuns = useCallback(async () => {
    setLogLoading(true);
    const data = await fetchHeartbeatRuns({ hours: LOG_WINDOW_HOURS });
    setRuns(data ?? []);
    setExpanded(null);
    setPage(0);
    setLogLoading(false);
  }, []);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  // Filter options derived from the loaded log + configured tasks.
  const taskOptions = useMemo(() => {
    const names = new Set<string>(tasks.map((task) => task.name));
    for (const run of runs) names.add(run.task_name);
    return Array.from(names).sort();
  }, [tasks, runs]);

  const userOptions = useMemo(() => {
    const seen = new Set<string>();
    for (const run of runs) {
      if (run.user_id) seen.add(run.user_id);
    }
    return Array.from(seen).sort();
  }, [runs]);

  const statusOptions = useMemo(() => {
    const seen = new Set<string>();
    for (const run of runs) seen.add(run.status);
    return Array.from(seen).sort();
  }, [runs]);

  // All filtering is client-side over the last 24h window of beats.
  const displayedRuns = useMemo(
    () =>
      runs.filter(
        (run) =>
          (!taskFilter || run.task_name === taskFilter) &&
          (!userFilter || run.user_id === userFilter) &&
          (!statusFilter || run.status === statusFilter)
      ),
    [runs, taskFilter, userFilter, statusFilter]
  );

  const pageCount = Math.max(1, Math.ceil(displayedRuns.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pagedRuns = useMemo(
    () => displayedRuns.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE),
    [displayedRuns, safePage]
  );

  // Reset to the first page + collapse any open detail row when the filtered set
  // changes (the expanded index is page-relative and would otherwise mis-point).
  useEffect(() => {
    setExpanded(null);
    setPage(0);
  }, [taskFilter, userFilter, statusFilter]);

  const rateDisabled = saving || (config ? !config.enabled : false);

  const handleSave = async () => {
    const minutes = Number.parseInt(rate, 10);
    if (!Number.isFinite(minutes) || minutes < MIN_RATE || minutes > MAX_RATE) {
      toast.error(t("adminPage.heartbeatSaveFailed"));
      return;
    }
    setSaving(true);
    try {
      const result = await updateHeartbeatRate(minutes);
      if (result) {
        setConfig(result);
        setRate(String(result.heartbeat_rate_minutes));
        toast.success(t("adminPage.heartbeatSaved"));
      } else {
        toast.error(t("adminPage.heartbeatSaveFailed"));
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Beat rate */}
      <Card>
        <CardHeader>
          <CardTitle>{t("adminPage.heartbeatRateTitle")}</CardTitle>
          <CardDescription>{t("adminPage.heartbeatRateHelp")}</CardDescription>
        </CardHeader>
        <div className="px-6 pb-6">
          {loading ? (
            <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
          ) : (
            <>
              {config && !config.enabled && (
                <p className="mb-3 text-sm text-muted-foreground">{t("adminPage.heartbeatDisabledNote")}</p>
              )}
              <div className="max-w-xs">
                <Label htmlFor="heartbeat-rate" className="mb-2 block">
                  {t("adminPage.heartbeatRateLabel")}
                </Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="heartbeat-rate"
                    type="number"
                    min={MIN_RATE}
                    max={MAX_RATE}
                    step={1}
                    value={rate}
                    onChange={(e) => setRate(e.target.value)}
                    disabled={rateDisabled}
                    aria-label={t("adminPage.heartbeatRateLabel")}
                  />
                  <Button type="button" size="sm" onClick={handleSave} disabled={rateDisabled}>
                    {saving ? t("common.saving") : t("adminPage.heartbeatSave")}
                  </Button>
                </div>
                {config && config.source === "default" && (
                  <p className="mt-2 text-sm text-muted-foreground">
                    {t("adminPage.heartbeatDefaultNote", { minutes: config.default_rate_minutes })}
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </Card>

      {/* Configured tasks */}
      <Card>
        <CardHeader>
          <CardTitle>{t("adminPage.heartbeatTasksTitle")}</CardTitle>
          <CardDescription>{t("adminPage.heartbeatTasksHelp")}</CardDescription>
        </CardHeader>
        <div className="px-6 pb-6">
          {loading ? (
            <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
          ) : tasks.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t("adminPage.heartbeatNoTasks")}</p>
          ) : (
            <ul className="space-y-2">
              {tasks.map((task) => (
                <li key={task.name} className="rounded-md border border-border p-3 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-foreground">{task.name}</span>
                    <Badge variant="secondary">
                      {task.scope === "per_user"
                        ? t("adminPage.heartbeatScopePerUser")
                        : t("adminPage.heartbeatScopeGlobal")}
                    </Badge>
                    {!task.enabled && (
                      <Badge variant="outline">{t("adminPage.heartbeatTaskDisabled")}</Badge>
                    )}
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <Label htmlFor={`cooldown-${task.name}`} className="text-xs text-muted-foreground">
                      {t("adminPage.heartbeatCooldownEditLabel")}
                    </Label>
                    <Input
                      id={`cooldown-${task.name}`}
                      type="number"
                      min={MIN_COOLDOWN}
                      max={MAX_COOLDOWN}
                      value={cooldownDraft[task.name] ?? String(task.cooldown_seconds)}
                      onChange={(e) =>
                        setCooldownDraft((prev) => ({ ...prev, [task.name]: e.target.value }))
                      }
                      className="h-8 w-28"
                      aria-label={t("adminPage.heartbeatCooldownEditLabel")}
                    />
                    <span className="text-xs text-muted-foreground">
                      {humanizeSeconds(Number(cooldownDraft[task.name] ?? task.cooldown_seconds) || 0)}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={
                        savingCooldown === task.name ||
                        String(cooldownDraft[task.name] ?? "") === String(task.cooldown_seconds)
                      }
                      onClick={() => saveCooldown(task.name)}
                    >
                      {t("adminPage.heartbeatSave")}
                    </Button>
                    {task.cooldown_source === "override" && (
                      <Badge variant="outline">
                        {t("adminPage.heartbeatCooldownDefault", { seconds: task.cooldown_default })}
                      </Badge>
                    )}
                  </div>
                  <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={task.type}>
                    {task.type}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">
                      {t("adminPage.heartbeatLastRunLabel")}:{" "}
                    </span>
                    {task.last_run && task.last_run.fired_at ? (
                      <>
                        {task.last_run.status} · {formatRelativeTime(task.last_run.fired_at)}
                      </>
                    ) : (
                      t("adminPage.heartbeatNever")
                    )}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Card>

      {/* Run log */}
      <Card>
        <CardHeader>
          <CardTitle>{t("adminPage.heartbeatLogTitle")}</CardTitle>
          <CardDescription>{t("adminPage.heartbeatLogHelp")}</CardDescription>
        </CardHeader>
        <div className="px-6 pb-6 space-y-4">
          <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
            <label className="flex flex-col">
              <span className="mb-1 text-xs font-medium text-muted-foreground">
                {t("adminPage.heartbeatColTask")}
              </span>
              <select
                aria-label={t("adminPage.heartbeatColTask")}
                className={SELECT_CLASS}
                value={taskFilter}
                onChange={(e) => setTaskFilter(e.target.value)}
              >
                <option value="">{t("adminPage.heartbeatFilterAllTasks")}</option>
                {taskOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col">
              <span className="mb-1 text-xs font-medium text-muted-foreground">
                {t("adminPage.heartbeatColUser")}
              </span>
              <select
                aria-label={t("adminPage.heartbeatColUser")}
                className={SELECT_CLASS}
                value={userFilter}
                onChange={(e) => setUserFilter(e.target.value)}
              >
                <option value="">{t("adminPage.heartbeatFilterAllUsers")}</option>
                {userOptions.map((uid) => (
                  <option key={uid} value={uid}>
                    {uid}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col">
              <span className="mb-1 text-xs font-medium text-muted-foreground">
                {t("adminPage.heartbeatColStatus")}
              </span>
              <select
                aria-label={t("adminPage.heartbeatColStatus")}
                className={SELECT_CLASS}
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="">{t("adminPage.heartbeatFilterAllStatuses")}</option>
                {statusOptions.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </label>
            <Button type="button" size="sm" variant="outline" onClick={() => void loadRuns()}>
              {t("adminPage.heartbeatRefresh")}
            </Button>
          </div>

          {logLoading ? (
            <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
          ) : displayedRuns.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t("adminPage.heartbeatNoRuns")}</p>
          ) : (
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2 font-medium">{t("adminPage.heartbeatColTask")}</th>
                    <th className="px-3 py-2 font-medium">{t("adminPage.heartbeatColUser")}</th>
                    <th className="px-3 py-2 font-medium">{t("adminPage.heartbeatColStatus")}</th>
                    <th className="px-3 py-2 font-medium">{t("adminPage.heartbeatColDuration")}</th>
                    <th className="px-3 py-2 font-medium">{t("adminPage.heartbeatColWhen")}</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedRuns.map((run, index) => {
                    const hasDetail =
                      run.status !== "ok" && run.detail && Object.keys(run.detail).length > 0;
                    const isExpanded = expanded === index;
                    return (
                      <tr
                        key={index}
                        className={`border-b border-border last:border-0 align-top ${
                          hasDetail ? "cursor-pointer hover:bg-muted/50" : ""
                        }`}
                        onClick={() => hasDetail && setExpanded(isExpanded ? null : index)}
                      >
                        <td className="px-3 py-2 font-mono text-xs">{run.task_name}</td>
                        <td className="px-3 py-2 font-mono text-xs">
                          {run.user_id ?? t("adminPage.heartbeatGlobalUser")}
                        </td>
                        <td className="px-3 py-2">
                          <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
                          {hasDetail && isExpanded && (
                            <pre className="mt-2 max-w-md overflow-x-auto rounded bg-muted p-2 text-xs">
                              {JSON.stringify(run.detail, null, 2)}
                            </pre>
                          )}
                        </td>
                        <td className="px-3 py-2 text-muted-foreground">{run.duration_ms}ms</td>
                        <td className="px-3 py-2 text-muted-foreground">
                          {run.fired_at ? formatRelativeTime(run.fired_at) : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {!logLoading && displayedRuns.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
              <span>
                {t("adminPage.heartbeatLogPageInfo", {
                  page: safePage + 1,
                  pages: pageCount,
                  total: displayedRuns.length,
                })}
              </span>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={safePage <= 0}
                  onClick={() => {
                    setExpanded(null);
                    setPage((p) => Math.max(0, p - 1));
                  }}
                >
                  {t("adminPage.heartbeatLogPrev")}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={safePage >= pageCount - 1}
                  onClick={() => {
                    setExpanded(null);
                    setPage((p) => Math.min(pageCount - 1, p + 1));
                  }}
                >
                  {t("adminPage.heartbeatLogNext")}
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
