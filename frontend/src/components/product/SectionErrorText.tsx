import * as React from "react";
import { cn } from "@/lib/utils";

interface SectionErrorTextProps {
  children: React.ReactNode;
  className?: string;
}

export function SectionErrorText({ children, className }: SectionErrorTextProps) {
  return <p className={cn("text-sm text-destructive", className)}>{children}</p>;
}
