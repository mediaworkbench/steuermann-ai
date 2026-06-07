interface MetricsStatsGridProps {
  children: React.ReactNode;
}

export function MetricsStatsGrid({ children }: MetricsStatsGridProps) {
  return <div className="mb-8 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">{children}</div>;
}
