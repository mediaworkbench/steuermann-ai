import * as React from "react";
import { cn } from "@/lib/utils";

interface DangerHintTextProps {
  children: React.ReactNode;
  className?: string;
}

export function DangerHintText({ children, className }: DangerHintTextProps) {
  return <p className={cn("mb-3 text-xs text-warning", className)}>{children}</p>;
}
