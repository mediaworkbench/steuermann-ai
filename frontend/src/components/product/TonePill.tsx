import * as React from "react";
import { cn } from "@/lib/utils";

type TonePillTone = "success" | "warning" | "info" | "muted";

interface TonePillProps {
  children: React.ReactNode;
  tone: TonePillTone;
  className?: string;
}

const toneClasses: Record<TonePillTone, string> = {
  success: "bg-success/10 text-success",
  warning: "bg-warning/10 text-warning",
  info: "bg-info/10 text-info",
  muted: "bg-muted text-foreground",
};

export function TonePill({ children, tone, className }: TonePillProps) {
  return (
    <span className={cn("inline-flex rounded-full px-2 py-0.5 text-xs font-semibold", toneClasses[tone], className)}>
      {children}
    </span>
  );
}
