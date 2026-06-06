import * as React from "react";

type ButtonVariant = "primary" | "secondary" | "ghost" | "destructive";
type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  asChild?: boolean;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-primary/40",
  secondary:
    "bg-secondary text-secondary-foreground border border-border hover:bg-surface-elevated focus-visible:ring-border-strong/40",
  ghost:
    "bg-transparent text-foreground hover:bg-surface-muted focus-visible:ring-border-strong/40",
  destructive:
    "bg-destructive text-destructive-foreground hover:bg-destructive/90 focus-visible:ring-destructive/40",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "px-3 py-1.5 text-xs rounded-md gap-1.5",
  md: "px-4 py-2 text-sm rounded-lg gap-2",
  lg: "px-6 py-3 text-sm rounded-xl gap-2",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className = "", variant = "primary", size = "md", asChild = false, type = "button", children, ...props },
  ref
) {
  const classes = [
    "inline-flex items-center justify-center font-semibold transition-colors",
    "disabled:cursor-not-allowed disabled:opacity-60",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
    variantClasses[variant],
    sizeClasses[size],
    className,
  ]
    .filter(Boolean)
    .join(" ");

  if (asChild) {
    const child = React.Children.only(children) as React.ReactElement<{ className?: string }>;
    return React.cloneElement(child, {
      ...props,
      className: [child.props.className, classes].filter(Boolean).join(" "),
    });
  }

  return (
    <button ref={ref} type={type} className={classes} {...props}>
      {children}
    </button>
  );
});

Button.displayName = "Button";