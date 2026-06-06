import * as React from "react";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  className?: string;
  actions?: React.ReactNode;
}

export function PageHeader({ title, subtitle, className, actions }: PageHeaderProps) {
  return (
    <div className={cn("flex items-start justify-between flex-wrap gap-3", className)}>
      <div>
        <h1 className="text-3xl font-bold text-foreground">{title}</h1>
        {subtitle ? <p className="mt-1 text-muted-foreground">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}
