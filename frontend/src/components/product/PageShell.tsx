import * as React from "react";
import { cn } from "@/lib/utils";

interface PageShellProps {
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
}

export function PageShell({ children, className, contentClassName }: PageShellProps) {
  return (
    <main className={cn("flex-1 overflow-y-auto bg-background", className)}>
      <div className={cn("mx-auto w-full px-4 py-6 md:px-8 md:py-8", contentClassName)}>{children}</div>
    </main>
  );
}
