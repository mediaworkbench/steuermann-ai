import * as React from "react";

export type CheckboxProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(function Checkbox(
  { className = "", ...props },
  ref
) {
  const classes = [
    "h-4 w-4 rounded border border-border text-primary focus:ring-2 focus:ring-focus-ring/30 focus:ring-offset-0",
    "disabled:cursor-not-allowed disabled:opacity-60",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return <input ref={ref} type="checkbox" className={classes} {...props} />;
});

Checkbox.displayName = "Checkbox";
