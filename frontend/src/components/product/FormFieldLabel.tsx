import * as React from "react";
import { cn } from "@/lib/utils";

interface FormFieldLabelProps {
  children: React.ReactNode;
  className?: string;
}

export function FormFieldLabel({ children, className }: FormFieldLabelProps) {
  return <label className={cn("mb-2 block text-sm font-medium text-foreground", className)}>{children}</label>;
}
