"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Undo2, History } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { fetchDreamAudit, undoDreamAction, type DreamAuditItem } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

const ACTION_KEY: Record<string, string> = {
  delete: "actionDelete",
  lower_confidence: "actionLowerConfidence",
  promote: "actionPromote",
  propose: "actionPropose",
};

function remaining(untilIso: string | null): string {
  if (!untilIso) return "";
  const ms = new Date(untilIso).getTime() - Date.now();
  if (ms <= 0) return "";
  const hours = Math.floor(ms / 3_600_000);
  const days = Math.floor(hours / 24);
  if (days >= 1) return `${days}d ${hours % 24}h`;
  if (hours >= 1) return `${hours}h`;
  return `${Math.max(1, Math.floor(ms / 60_000))}m`;
}

/**
 * The 7-day undo feed: engine actions (forget / lower-confidence / consolidate /
 * propose) still inside their reversible window, with a countdown. Undo reverses
 * the change (re-inserts the memory, restores confidence, etc.) and removes it from
 * the feed. Self-scoped to the signed-in user.
 */
export function MemoryUndoSection() {
  const { t } = useI18n();
  const [items, setItems] = useState<DreamAuditItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setItems(await fetchDreamAudit());
    } catch {
      toast.error(t("dreaming.loadError"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void load();
  }, [load]);

  const undo = useCallback(
    async (auditId: string) => {
      setBusy(auditId);
      try {
        await undoDreamAction(auditId);
        setItems((prev) => prev.filter((i) => i.audit_id !== auditId));
        toast.success(t("dreaming.actionSuccess"));
      } catch {
        toast.error(t("dreaming.actionFailed"));
      } finally {
        setBusy(null);
      }
    },
    [t],
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5 text-primary" aria-hidden="true" />
          {t("dreaming.undoTitle")}
        </CardTitle>
        <CardDescription>{t("dreaming.undoDescription")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading ? (
          <p className="text-sm text-muted-foreground">{t("dreaming.refresh")}…</p>
        ) : items.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("dreaming.undoEmpty")}</p>
        ) : (
          <ul className="space-y-2">
            {items.map((item) => {
              const label = ACTION_KEY[item.action ?? ""] ?? "actionDelete";
              const time = remaining(item.reversible_until);
              return (
                <li
                  key={item.audit_id}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface px-4 py-3"
                >
                  <div className="min-w-0">
                    <p className="text-sm text-foreground">{t(`dreaming.${label}`)}</p>
                    {time && (
                      <p className="text-xs text-muted-foreground">
                        {t("dreaming.expiresIn", { time })}
                      </p>
                    )}
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy === item.audit_id}
                    onClick={() => undo(item.audit_id)}
                  >
                    <Undo2 className="mr-1 h-4 w-4" aria-hidden="true" />
                    {t("dreaming.undoButton")}
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
