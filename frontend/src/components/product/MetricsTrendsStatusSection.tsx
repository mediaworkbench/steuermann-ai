import * as React from "react";
import { RefreshCw } from "lucide-react";

interface MetricsTrendsStatusSectionProps {
  lastUpdatedLabel: string;
  lastUpdatedValue: string;
  errorTitle: string;
  errorMessage?: React.ReactNode;
  isLoading: boolean;
  loadingLabel: string;
  children: React.ReactNode;
}

export function MetricsTrendsStatusSection({
  lastUpdatedLabel,
  lastUpdatedValue,
  errorTitle,
  errorMessage,
  isLoading,
  loadingLabel,
  children,
}: MetricsTrendsStatusSectionProps) {
  return (
    <>
      <div className="mb-6 text-xs text-muted-foreground">
        {lastUpdatedLabel}: {lastUpdatedValue}
      </div>

      {errorMessage ? <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
        <p className="mb-1 font-semibold">{errorTitle}</p>
        {errorMessage ? <p>{errorMessage}</p> : null}
      </div> : null}

      {isLoading ? <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
        <RefreshCw size={32} className="animate-spin" />
        <p className="mt-4 mb-0">{loadingLabel}</p>
      </div> : children}
    </>
  );
}
