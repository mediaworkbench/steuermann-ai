import * as React from "react";
import { cn } from "@/lib/utils";

interface AnalyticsChartCardProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function AnalyticsChartCard({ title, children, className }: AnalyticsChartCardProps) {
  return (
    <div className={cn("rounded-lg border border-border bg-surface p-6 shadow-sm", className)}>
      <h3 className="mb-4 text-lg font-semibold text-foreground">{title}</h3>
      {children}
    </div>
  );
}
