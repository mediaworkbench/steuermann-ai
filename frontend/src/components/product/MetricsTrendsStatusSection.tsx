import * as React from "react";
import { MetricsLoadingState } from "@/components/product/MetricsLoadingState";
import { PageErrorAlert } from "@/components/product/PageErrorAlert";

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

      {errorMessage ? <PageErrorAlert title={errorTitle} message={errorMessage} /> : null}

      {isLoading ? <MetricsLoadingState label={loadingLabel} /> : children}
    </>
  );
}
