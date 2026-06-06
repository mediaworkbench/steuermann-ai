import * as React from "react";
import { cn } from "@/lib/utils";

export type SelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { className, ...props },
  ref
) {
  return (
    <select
      ref={ref}
      className={cn(
        "w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground shadow-sm",
        "outline-none transition-colors",
        "focus:border-primary focus:ring-2 focus:ring-focus-ring/30 focus:ring-offset-0",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    />
  );
});

Select.displayName = "Select";
