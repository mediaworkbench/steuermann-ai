import * as React from "react";
import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Input = React.forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, ...props },
  ref
) {
  return (
    <input
      ref={ref}
      className={cn(
        "w-full rounded-2xl border border-border bg-surface px-4 py-3 text-foreground shadow-sm",
        "outline-none transition-colors",
        "placeholder:text-muted-foreground",
        "focus:border-primary focus:ring-2 focus:ring-focus-ring/30 focus:ring-offset-0",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    />
  );
});

Input.displayName = "Input";