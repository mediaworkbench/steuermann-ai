import * as React from "react";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";

interface MetricsTableRow {
  label: string;
  value: React.ReactNode;
}

interface MetricsTableCardProps {
  title: string;
  labelHeader: string;
  valueHeader: string;
  rows: MetricsTableRow[];
}

export function MetricsTableCard({
  title,
  labelHeader,
  valueHeader,
  rows,
}: MetricsTableCardProps) {
  return (
    <AnalyticsChartCard title={title}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="border-b border-border px-4 py-2 text-left font-semibold text-muted-foreground">
                {labelHeader}
              </th>
              <th className="border-b border-border px-4 py-2 text-left font-semibold text-muted-foreground">
                {valueHeader}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <td className="border-b border-border px-4 py-2 text-foreground">{row.label}</td>
                <td className="border-b border-border px-4 py-2 text-right font-medium text-foreground">{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AnalyticsChartCard>
  );
}
