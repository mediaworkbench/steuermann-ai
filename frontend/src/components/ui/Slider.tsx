import * as React from "react";

export type SliderProps = React.InputHTMLAttributes<HTMLInputElement>;

export const Slider = React.forwardRef<HTMLInputElement, SliderProps>(function Slider(
  { className = "", ...props },
  ref
) {
  const classes = [
    "h-2 w-full cursor-pointer appearance-none rounded-full bg-muted",
    "focus:outline-none focus:ring-2 focus:ring-focus-ring/30 focus:ring-offset-0",
    "disabled:cursor-not-allowed disabled:opacity-60",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return <input ref={ref} type="range" className={classes} {...props} />;
});

Slider.displayName = "Slider";
