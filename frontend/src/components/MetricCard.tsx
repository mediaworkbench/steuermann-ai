"use client";

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: "up" | "down" | "stable";
  icon?: React.ReactNode;
}

export function MetricCard({ label, value, unit = "", trend = "stable", icon }: MetricCardProps) {
  const trendColor = {
    up: "text-destructive",
    down: "text-success",
    stable: "text-muted-foreground",
  }[trend];

  const trendArrow = {
    up: "↑",
    down: "↓",
    stable: "→",
  }[trend];

  return (
    <div className="bg-surface border border-border rounded-lg shadow-sm p-6">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-medium text-muted-foreground">{label}</h3>
        {icon && <div className="text-2xl">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <div className="text-3xl font-bold text-foreground">
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>
        {unit && <div className="text-sm text-muted-foreground">{unit}</div>}
      </div>
      <div className={`text-sm mt-2 ${trendColor}`}>
        {trendArrow} {trend}
      </div>
    </div>
  );
}
