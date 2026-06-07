import * as React from "react";
import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";

interface MetricsKeyValueGridItem {
  keyLabel: string;
  value: React.ReactNode;
}

interface MetricsKeyValueGridCardProps {
  title: string;
  items: MetricsKeyValueGridItem[];
}

export function MetricsKeyValueGridCard({ title, items }: MetricsKeyValueGridCardProps) {
  return (
    <AnalyticsChartCard title={title}>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {items.map((item) => (
          <div key={item.keyLabel} className="rounded-md bg-surface-muted p-4">
            <p className="m-0 text-sm text-muted-foreground">{item.keyLabel}</p>
            <p className="m-0 mt-1 text-2xl font-bold text-foreground">{item.value}</p>
          </div>
        ))}
      </div>
    </AnalyticsChartCard>
  );
}
