"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { BrainCog } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  fetchDreamConflicts,
  resolveDreamConflict,
  type DreamConflict,
  type DreamConflictChoice,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

const CHOICES: DreamConflictChoice[] = ["keep_old", "accept_new", "depends"];
const CHOICE_LABEL: Record<DreamConflictChoice, "keepOld" | "acceptNew" | "dependsLabel"> = {
  keep_old: "keepOld",
  accept_new: "acceptNew",
  depends: "dependsLabel",
};

/**
 * User-scoped dissonance queue: the engine lowered a semantic memory's confidence
 * because a newer observation may contradict it. The user picks how to reconcile —
 * keep the remembered fact, accept the new one, or mark "it depends" (reviewed,
 * left as-is). Self-scoped; the backend resolves against the signed-in user only.
 */
export function DissonanceQueueSection() {
  const { t } = useI18n();
  const [conflicts, setConflicts] = useState<DreamConflict[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setConflicts(await fetchDreamConflicts());
    } catch {
      toast.error(t("dreaming.loadError"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void load();
  }, [load]);

  const resolve = useCallback(
    async (conflictId: string, choice: DreamConflictChoice) => {
      setBusy(conflictId);
      try {
        await resolveDreamConflict(conflictId, choice);
        setConflicts((prev) => prev.filter((c) => c.conflict_id !== conflictId));
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
          <BrainCog className="h-5 w-5 text-primary" aria-hidden="true" />
          {t("dreaming.conflictsTitle")}
        </CardTitle>
        <CardDescription>{t("dreaming.conflictsDescription")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <p className="text-sm text-muted-foreground">{t("dreaming.refresh")}…</p>
        ) : conflicts.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("dreaming.conflictsEmpty")}</p>
        ) : (
          <ul className="space-y-4">
            {conflicts.map((c) => (
              <li key={c.conflict_id} className="rounded-lg border border-border bg-surface p-4">
                <dl className="space-y-2 text-sm">
                  <div>
                    <dt className="font-medium text-muted-foreground">{t("dreaming.conflictEstablished")}</dt>
                    <dd className="text-foreground">{c.semantic_text || "—"}</dd>
                  </div>
                  <div>
                    <dt className="font-medium text-muted-foreground">{t("dreaming.conflictNew")}</dt>
                    <dd className="text-foreground">{c.contradiction_text || "—"}</dd>
                  </div>
                </dl>
                <div className="mt-3 flex flex-wrap gap-2">
                  {CHOICES.map((choice) => (
                    <Button
                      key={choice}
                      size="sm"
                      variant={choice === "accept_new" ? "default" : "outline"}
                      disabled={busy === c.conflict_id}
                      onClick={() => resolve(c.conflict_id, choice)}
                    >
                      {t(`dreaming.${CHOICE_LABEL[choice]}`)}
                    </Button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
