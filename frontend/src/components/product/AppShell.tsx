import * as React from "react";
import { cn } from "@/lib/utils";

interface AppShellProps {
  children: React.ReactNode;
  className?: string;
}

export function AppShell({ children, className }: AppShellProps) {
  return (
    <main
      className={cn(
        "relative isolate flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-surface text-foreground",
        className
      )}
    >
      {children}
    </main>
  );
}
