"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, Database, Coins, ListChecks, Trash2, Sparkles, RefreshCw } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { fetchDreamingMetrics, type DreamingMetrics } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

interface Metric {
  key: string;
  icon: typeof Activity;
  value: number | string;
}

/**
 * Admin dashboard for the Dreaming Engine — aggregate, anonymized cards only
 * (cycle count, stored vectors, average token cost, pending resolutions, plus
 * deletion/promotion rates). The backend endpoint sources these from
 * COUNT/GROUP-BY rollups, so no user content, ids, or names are ever exposed.
 */
export function DreamingMetricsSection() {
  const { t } = useI18n();
  const [metrics, setMetrics] = useState<DreamingMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setMetrics(await fetchDreamingMetrics());
    setLoading(false);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const cards: Metric[] = metrics
    ? [
        { key: "cyclesRun", icon: Activity, value: metrics.cycles_run },
        { key: "vectorCount", icon: Database, value: metrics.vector_count },
        { key: "avgCost", icon: Coins, value: metrics.avg_tokens_per_cycle },
        { key: "pendingResolutions", icon: ListChecks, value: metrics.pending_resolutions.total },
        { key: "deletions", icon: Trash2, value: metrics.deletion_count },
        { key: "promotions", icon: Sparkles, value: metrics.promotion_count },
      ]
    : [];

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <Button size="sm" variant="outline" onClick={() => void load()} disabled={loading}>
          <RefreshCw className="mr-1 h-4 w-4" aria-hidden="true" />
          {t("dreaming.refresh")}
        </Button>
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">{t("dreaming.refresh")}…</p>
      ) : !metrics ? (
        <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
          {t("dreaming.loadError")}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {cards.map(({ key, icon: Icon, value }) => (
              <Card key={key}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    {t(`dreaming.${key}`)}
                  </CardTitle>
                  <Icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold text-foreground">{value}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardContent className="flex flex-wrap gap-x-8 gap-y-2 pt-6 text-sm text-muted-foreground">
              <span>
                {t("dreaming.openConflicts")}: <span className="font-semibold text-foreground">{metrics.pending_resolutions.open_conflicts}</span>
              </span>
              <span>
                {t("dreaming.proposedRules")}: <span className="font-semibold text-foreground">{metrics.pending_resolutions.proposed_procedural}</span>
              </span>
              <span>
                {t("dreaming.totalTokens")}: <span className="font-semibold text-foreground">{metrics.total_tokens}</span>
              </span>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
