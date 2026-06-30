"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Lock, SlidersHorizontal } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  fetchDreamProcedural,
  decideDreamRule,
  type DreamProceduralRule,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

function statusBadge(status: string): { variant: "default" | "secondary" | "outline"; key: string } {
  if (status === "active") return { variant: "default", key: "statusActive" };
  if (status === "proposed") return { variant: "secondary", key: "statusProposed" };
  if (status === "rejected") return { variant: "outline", key: "statusRejected" };
  return { variant: "outline", key: "statusObserving" };
}

/**
 * Procedural approvals: the engine proposes formatting (Tier 1) and style (Tier 2)
 * preferences learned from behaviour. A proposed rule reaches the prompt only after
 * the user approves it (→ active). Core-logic / safety (Tier 3) rules are never
 * auto-learned, surfaced here only as a locked informational note.
 */
export function ProceduralApprovalsSection() {
  const { t } = useI18n();
  const [rules, setRules] = useState<DreamProceduralRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setRules(await fetchDreamProcedural());
    } catch {
      toast.error(t("dreaming.loadError"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void load();
  }, [load]);

  const decide = useCallback(
    async (ruleKey: string, decision: "approve" | "reject") => {
      setBusy(ruleKey);
      try {
        await decideDreamRule(ruleKey, decision);
        toast.success(t("dreaming.actionSuccess"));
        await load();
      } catch {
        toast.error(t("dreaming.actionFailed"));
      } finally {
        setBusy(null);
      }
    },
    [t, load],
  );

  // Show actionable (proposed) first, then active; observing/rejected are hidden as
  // they aren't user-actionable here.
  const visible = useMemo(
    () => rules.filter((r) => r.status === "proposed" || r.status === "active"),
    [rules],
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <SlidersHorizontal className="h-5 w-5 text-primary" aria-hidden="true" />
          {t("dreaming.proceduralTitle")}
        </CardTitle>
        <CardDescription>{t("dreaming.proceduralDescription")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <p className="text-sm text-muted-foreground">{t("dreaming.refresh")}…</p>
        ) : visible.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("dreaming.proceduralEmpty")}</p>
        ) : (
          <ul className="space-y-3">
            {visible.map((r) => {
              const badge = statusBadge(r.status);
              return (
                <li
                  key={r.rule_key}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface p-4"
                >
                  <div className="min-w-0">
                    <p className="text-sm text-foreground">{r.rule_text}</p>
                    <div className="mt-1 flex items-center gap-2">
                      <Badge variant="outline">{t("dreaming.tierLabel", { tier: r.tier })}</Badge>
                      <Badge variant={badge.variant}>{t(`dreaming.${badge.key}`)}</Badge>
                    </div>
                  </div>
                  {r.status === "proposed" && (
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        disabled={busy === r.rule_key}
                        onClick={() => decide(r.rule_key, "approve")}
                      >
                        {t("dreaming.approve")}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={busy === r.rule_key}
                        onClick={() => decide(r.rule_key, "reject")}
                      >
                        {t("dreaming.reject")}
                      </Button>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <Lock className="h-3.5 w-3.5" aria-hidden="true" />
          {t("dreaming.tier3Locked")}
        </p>
      </CardContent>
    </Card>
  );
}
