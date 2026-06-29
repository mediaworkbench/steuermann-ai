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
  type HeartbeatRateConfig,
  type HeartbeatRun,
  type HeartbeatTaskInfo,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

const MIN_RATE = 1;
const MAX_RATE = 1440;
const LOG_LIMIT = 50;

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
 * log shows one row per user for them. The log loads the last {@link LOG_LIMIT}
 * beats and is filtered client-side by task, user, and status. The rate is a
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
  const [runs, setRuns] = useState<HeartbeatRun[]>([]);
  const [logLoading, setLogLoading] = useState(true);
  const [taskFilter, setTaskFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);

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
      if (taskData) setTasks(taskData);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const loadRuns = useCallback(async () => {
    setLogLoading(true);
    const data = await fetchHeartbeatRuns({ limit: LOG_LIMIT });
    setRuns(data ?? []);
    setExpanded(null);
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

  // All filtering is client-side over the last {LOG_LIMIT} beats.
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

  // Collapse any open detail row when the filtered set changes (the expanded
  // index would otherwise point at a different row).
  useEffect(() => {
    setExpanded(null);
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
                    <span className="text-muted-foreground">
                      {t("adminPage.heartbeatCooldownLabel", { seconds: task.cooldown_seconds })}
                    </span>
                    {!task.enabled && (
                      <Badge variant="outline">{t("adminPage.heartbeatTaskDisabled")}</Badge>
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
                  {displayedRuns.map((run, index) => {
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
        </div>
      </Card>
    </div>
  );
}
