import * as React from "react";

interface AnalyticsChartStateProps {
  message: React.ReactNode;
  compact?: boolean;
}

export function AnalyticsChartState({ message, compact = false }: AnalyticsChartStateProps) {
  return <div className={`${compact ? "py-8" : "py-12"} text-center text-muted-foreground`}>{message}</div>;
}
