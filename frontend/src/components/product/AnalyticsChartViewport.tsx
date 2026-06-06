import * as React from "react";
import { ResponsiveContainer } from "recharts";

interface AnalyticsChartViewportProps {
  children: React.ReactElement;
  height?: number;
}

export function AnalyticsChartViewport({ children, height = 300 }: AnalyticsChartViewportProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      {children}
    </ResponsiveContainer>
  );
}
