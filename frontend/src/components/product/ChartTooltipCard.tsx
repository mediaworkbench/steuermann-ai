import * as React from "react";

interface ChartTooltipCardProps {
  title: React.ReactNode;
  children: React.ReactNode;
}

export function ChartTooltipCard({ title, children }: ChartTooltipCardProps) {
  return (
    <div className="rounded border border-border bg-surface p-3 shadow-lg">
      <p className="text-sm font-semibold text-foreground">{title}</p>
      {children}
    </div>
  );
}
