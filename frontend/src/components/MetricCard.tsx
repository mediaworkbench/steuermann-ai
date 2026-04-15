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
    up: "text-red-600",
    down: "text-green-600",
    stable: "text-gray-600",
  }[trend];

  const trendArrow = {
    up: "↑",
    down: "↓",
    stable: "→",
  }[trend];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-medium text-gray-600">{label}</h3>
        {icon && <div className="text-2xl">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <div className="text-3xl font-bold text-gray-900">
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>
        {unit && <div className="text-sm text-gray-600">{unit}</div>}
      </div>
      <div className={`text-sm mt-2 ${trendColor}`}>
        {trendArrow} {trend}
      </div>
    </div>
  );
}
