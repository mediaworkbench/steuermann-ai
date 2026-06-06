import * as React from "react";
import { cn } from "@/lib/utils";

interface SectionStateTextProps {
  children: React.ReactNode;
  className?: string;
}

export function SectionStateText({ children, className }: SectionStateTextProps) {
  return <p className={cn("text-sm text-muted-foreground", className)}>{children}</p>;
}
