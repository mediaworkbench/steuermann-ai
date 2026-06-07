import * as React from "react";

interface MetricsSummaryListRow {
  label: React.ReactNode;
  value: React.ReactNode;
}

interface MetricsSummaryListProps {
  rows: MetricsSummaryListRow[];
}

export function MetricsSummaryList({ rows }: MetricsSummaryListProps) {
  return (
    <div className="grid gap-2">
      {rows.map((row, index) => (
        <div
          key={`${String(row.label)}-${index}`}
          className="flex items-center justify-between border-b border-border py-2 text-foreground last:border-b-0"
        >
          <span>{row.label}</span>
          <span>{row.value}</span>
        </div>
      ))}
    </div>
  );
}
