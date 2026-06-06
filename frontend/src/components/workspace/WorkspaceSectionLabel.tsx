import * as React from "react";
import { cn } from "@/lib/utils";

interface WorkspaceSectionLabelProps {
  children: React.ReactNode;
  className?: string;
}

export function WorkspaceSectionLabel({ children, className }: WorkspaceSectionLabelProps) {
  return (
    <p className={cn("mb-1.5 px-0.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground", className)}>
      {children}
    </p>
  );
}
