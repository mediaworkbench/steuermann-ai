import * as React from "react";
import { cn } from "@/lib/utils";

type WorkspaceInlineBadgeTone = "default" | "primary" | "destructive";
type WorkspaceInlineBadgeSize = "default" | "compact";

interface WorkspaceInlineBadgeProps {
  children: React.ReactNode;
  className?: string;
  tone?: WorkspaceInlineBadgeTone;
  size?: WorkspaceInlineBadgeSize;
  href?: string;
}

export function WorkspaceInlineBadge({
  children,
  className,
  tone = "default",
  size = "default",
  href,
}: WorkspaceInlineBadgeProps) {
  const baseClass = cn(
    "inline-flex items-center border font-medium",
    size === "compact" ? "gap-1 rounded px-1.5 py-0.5 text-[10px]" : "gap-1.5 rounded-lg px-2.5 py-1.5 text-xs",
    tone === "default" && "bg-surface-muted text-foreground border-border",
    tone === "primary" && "bg-primary/5 text-primary border-primary/20",
    tone === "destructive" && "bg-destructive/10 text-destructive border-destructive/20",
    href && tone === "primary" && "no-underline transition-colors hover:bg-primary/10",
    className
  );

  if (href) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={baseClass}>
        {children}
      </a>
    );
  }

  return <span className={baseClass}>{children}</span>;
}
