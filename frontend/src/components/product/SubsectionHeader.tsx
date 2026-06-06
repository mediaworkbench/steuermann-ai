import * as React from "react";
import { cn } from "@/lib/utils";

interface SubsectionHeaderProps {
  title: React.ReactNode;
  description?: React.ReactNode;
  className?: string;
}

export function SubsectionHeader({ title, description, className }: SubsectionHeaderProps) {
  return (
    <div className={cn("mb-3", className)}>
      <h4 className="mb-2 text-md font-semibold text-foreground">{title}</h4>
      {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
    </div>
  );
}
