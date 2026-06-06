import * as React from "react";

interface AnalyticsStatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  unit?: string;
}

export function AnalyticsStatCard({ icon, label, value, unit }: AnalyticsStatCardProps) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-border bg-surface px-4 py-3 shadow-sm">
      <div className="mt-0.5 text-muted-foreground">{icon}</div>
      <div>
        <p className="mb-0.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">{label}</p>
        <p className="text-xl font-bold text-foreground">
          {value}
          {unit && <span className="ml-1 text-sm font-medium text-muted-foreground">{unit}</span>}
        </p>
      </div>
    </div>
  );
}
