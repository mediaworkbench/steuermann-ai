"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchHeartbeatRate, updateHeartbeatRate, type HeartbeatRateConfig } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

const MIN_RATE = 1;
const MAX_RATE = 1440;

/**
 * Admin-only control for the global heartbeat beat rate (minutes). The rate is a
 * deployment-wide setting persisted in Postgres; the LangGraph-embedded
 * scheduler applies a change within ~30s. The beat is otherwise enabled/disabled
 * in the profile's core.yaml.
 */
export function HeartbeatSettingsSection() {
  const { t, formatRelativeTime } = useI18n();
  const [config, setConfig] = useState<HeartbeatRateConfig | null>(null);
  const [rate, setRate] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchHeartbeatRate().then((data) => {
      if (cancelled) return;
      if (!data) {
        toast.error(t("adminPage.heartbeatLoadFailed"));
        setLoading(false);
        return;
      }
      setConfig(data);
      setRate(String(data.heartbeat_rate_minutes));
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const disabled = saving || (config ? !config.enabled : false);

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
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle>{t("adminPage.heartbeatSection")}</CardTitle>
        <CardDescription>{t("adminPage.heartbeatDescription")}</CardDescription>
      </CardHeader>
      <div className="space-y-4 px-6 pb-6">
        {loading ? (
          <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
        ) : (
          <>
            {config && !config.enabled && (
              <p className="text-sm text-muted-foreground">{t("adminPage.heartbeatDisabledNote")}</p>
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
                  disabled={disabled}
                  aria-label={t("adminPage.heartbeatRateLabel")}
                />
                <Button type="button" size="sm" onClick={handleSave} disabled={disabled}>
                  {saving ? t("common.saving") : t("adminPage.heartbeatSave")}
                </Button>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">{t("adminPage.heartbeatRateHelp")}</p>
              {config && config.source === "default" && (
                <p className="mt-1 text-sm text-muted-foreground">
                  {t("adminPage.heartbeatDefaultNote", { minutes: config.default_rate_minutes })}
                </p>
              )}
            </div>
            <div className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">{t("adminPage.heartbeatLastRunLabel")}: </span>
              {config && config.last_run ? (
                <span>
                  {config.last_run.status} · {formatRelativeTime(config.last_run.fired_at)}
                </span>
              ) : (
                <span>{t("adminPage.heartbeatNever")}</span>
              )}
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
