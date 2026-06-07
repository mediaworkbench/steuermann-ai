import { AnalyticsChartCard } from "@/components/product/AnalyticsChartCard";
import { AnalyticsChartState } from "@/components/product/AnalyticsChartState";
import { MetricsSeriesOptions } from "@/components/product/MetricsSeriesOptions";

interface MetricsTrendChartOption {
  key: string;
  label: string;
  checked: boolean;
  onToggle: (checked: boolean) => void;
}

interface MetricsTrendChartCardProps {
  title: string;
  subtitle: string;
  options?: MetricsTrendChartOption[];
  hasData: boolean;
  hasSelection?: boolean;
  emptyDataMessage: string;
  emptySelectionMessage?: string;
  children: React.ReactNode;
}

export function MetricsTrendChartCard({
  title,
  subtitle,
  options,
  hasData,
  hasSelection = true,
  emptyDataMessage,
  emptySelectionMessage,
  children,
}: MetricsTrendChartCardProps) {
  return (
    <AnalyticsChartCard title={title}>
      <p className="mb-2 text-sm text-muted-foreground">{subtitle}</p>
      {options && options.length > 0 ? <MetricsSeriesOptions options={options} /> : null}
      {!hasData ? (
        <AnalyticsChartState compact message={emptyDataMessage} />
      ) : !hasSelection ? (
        <AnalyticsChartState compact message={emptySelectionMessage ?? emptyDataMessage} />
      ) : (
        children
      )}
    </AnalyticsChartCard>
  );
}
