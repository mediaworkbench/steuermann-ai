import * as React from "react";
import { cn } from "@/lib/utils";

interface SectionHeaderProps {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  tone?: "default" | "danger";
}

const titleToneClasses: Record<NonNullable<SectionHeaderProps["tone"]>, string> = {
  default: "text-foreground",
  danger: "text-destructive",
};

export function SectionHeader({
  title,
  description,
  actions,
  className,
  tone = "default",
}: SectionHeaderProps) {
  return (
    <div className={cn("mb-4", className)}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className={cn("text-lg font-semibold", titleToneClasses[tone])}>{title}</h3>
          {description ? <p className="mt-1 text-sm text-muted-foreground">{description}</p> : null}
        </div>
        {actions ? <div className="shrink-0">{actions}</div> : null}
      </div>
    </div>
  );
}
