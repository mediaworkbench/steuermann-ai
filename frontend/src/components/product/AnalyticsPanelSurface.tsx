import * as React from "react";
import { cn } from "@/lib/utils";

interface AnalyticsPanelSurfaceProps {
  children: React.ReactNode;
  className?: string;
}

export function AnalyticsPanelSurface({ children, className }: AnalyticsPanelSurfaceProps) {
  return <div className={cn("rounded-2xl border border-border bg-surface p-6 shadow-sm", className)}>{children}</div>;
}
