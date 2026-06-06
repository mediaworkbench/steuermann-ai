import * as React from "react";
import { cn } from "@/lib/utils";

interface WorkspacePanelShellProps {
  isOpen: boolean;
  children: React.ReactNode;
  className?: string;
}

export function WorkspacePanelShell({ isOpen, children, className }: WorkspacePanelShellProps) {
  return (
    <div
      className={cn(
        "fixed right-0 top-16 z-10 flex h-[calc(100vh-4rem)] w-80 min-h-0 flex-col overflow-hidden border-l border-border bg-surface transition-all duration-200",
        "md:sticky md:top-20 md:z-0 md:h-[calc(100vh-5rem)] md:self-start",
        isOpen ? "translate-x-0 md:w-64 lg:w-72" : "translate-x-full md:w-0 md:translate-x-0 md:border-l-0",
        className
      )}
    >
      {children}
    </div>
  );
}
