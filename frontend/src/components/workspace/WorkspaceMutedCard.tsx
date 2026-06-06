import * as React from "react";
import { cn } from "@/lib/utils";

interface WorkspaceMutedCardProps {
  children: React.ReactNode;
  className?: string;
}

export function WorkspaceMutedCard({ children, className }: WorkspaceMutedCardProps) {
  return <div className={cn("rounded-lg border border-border bg-surface-muted", className)}>{children}</div>;
}
