"use client";

import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useProviderHealth } from "@/context/ProviderHealthContext";
import { useI18n } from "@/hooks/useI18n";
import { cn } from "@/lib/utils";

/**
 * Global status strip shown at the top of the content area when the external LLM
 * provider is unreachable (`offline`) or only partially reachable (`degraded`).
 * Hidden entirely while `online`. The Retry button re-pings the provider and
 * triggers a background capability reprobe; the banner clears automatically the
 * moment the provider answers.
 */
export function ProviderOfflineBanner() {
  const { status, loading, refresh } = useProviderHealth();
  const { t } = useI18n();

  if (status === "online") return null;

  const isOffline = status === "offline";

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "flex shrink-0 items-center justify-between gap-3 border-b px-4 py-2 text-sm",
        isOffline
          ? "bg-destructive text-destructive-foreground"
          : "bg-warning text-warning-foreground",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <AlertTriangle className="size-4 shrink-0" aria-hidden="true" />
        <span className="truncate font-medium">
          {isOffline ? t("providerHealth.offlineTitle") : t("providerHealth.degradedTitle")}
        </span>
      </div>
      <Button
        size="sm"
        variant="secondary"
        onClick={refresh}
        disabled={loading}
        className="shrink-0"
      >
        <RotateCcw className={cn("size-3.5", loading && "animate-spin")} aria-hidden="true" />
        {loading ? t("providerHealth.checking") : t("providerHealth.retry")}
      </Button>
    </div>
  );
}
