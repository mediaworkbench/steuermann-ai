import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { cn } from "@/lib/utils";

/**
 * Radix-based Slider.
 * Accepts the same scalar `value` / `onChange` props as a native <input type="range">
 * so the single caller (SettingsPanel RAG top_k) needs no changes.
 * `min` and `max` accept strings as well as numbers for compatibility.
 */
export interface SliderProps {
  min?: number | string;
  max?: number | string;
  step?: number | string;
  value?: number;
  defaultValue?: number;
  /** Native-style handler — receives a synthetic event with `e.target.value` as a string. */
  onChange?: (e: { target: { value: string } }) => void;
  disabled?: boolean;
  className?: string;
}

export const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(function Slider(
  { min = 0, max = 100, step = 1, value, defaultValue, onChange, disabled, className },
  ref
) {
  return (
    <SliderPrimitive.Root
      ref={ref}
      min={Number(min)}
      max={Number(max)}
      step={Number(step)}
      value={value !== undefined ? [value] : undefined}
      defaultValue={defaultValue !== undefined ? [defaultValue] : undefined}
      onValueChange={([v]) => {
        onChange?.({ target: { value: String(v) } });
      }}
      disabled={disabled}
      className={cn(
        "relative flex w-full touch-none select-none items-center",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
    >
      <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-muted">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        className={cn(
          "block h-5 w-5 rounded-full border-2 border-primary bg-surface shadow-md",
          "transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring/30",
          "focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          "disabled:pointer-events-none"
        )}
      />
    </SliderPrimitive.Root>
  );
});

Slider.displayName = "Slider";
