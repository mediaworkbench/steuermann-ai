import * as React from "react";
import { cn } from "@/lib/utils";

interface WorkspacePanelSectionProps {
  children: React.ReactNode;
  className?: string;
}

export function WorkspacePanelSection({ children, className }: WorkspacePanelSectionProps) {
  return <div className={cn("border-t border-border bg-surface px-3 py-3", className)}>{children}</div>;
}
