import * as React from "react";
import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";

/**
 * Radix-based Checkbox.
 * Accepts the same `checked` / `onChange` props as a native <input type="checkbox">
 * so existing callers need no changes. The `type` prop is accepted but ignored
 * (Radix renders a <button>, not an <input>).
 */
export interface CheckboxProps {
  checked?: boolean;
  defaultChecked?: boolean;
  /** Native-style change handler — receives a synthetic event with `e.target.checked`. */
  onChange?: React.ChangeEventHandler<HTMLInputElement>;
  disabled?: boolean;
  className?: string;
  id?: string;
  name?: string;
  /** Accepted and silently dropped — Radix renders a <button>, not a native checkbox. */
  type?: string;
}

export const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  CheckboxProps
// eslint-disable-next-line @typescript-eslint/no-unused-vars
>(function Checkbox({ checked, defaultChecked, onChange, disabled, className, type, ...rest }, ref) {
  return (
    <CheckboxPrimitive.Root
      ref={ref}
      checked={checked}
      defaultChecked={defaultChecked}
      onCheckedChange={(state) => {
        onChange?.({
          target: { checked: state === true },
        } as React.ChangeEvent<HTMLInputElement>);
      }}
      disabled={disabled}
      className={cn(
        "peer h-4 w-4 shrink-0 rounded border border-border bg-surface shadow-sm",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring/30 focus-visible:ring-offset-0",
        "disabled:cursor-not-allowed disabled:opacity-60",
        "data-[state=checked]:bg-primary data-[state=checked]:border-primary data-[state=checked]:text-primary-foreground",
        className
      )}
      {...rest}
    >
      <CheckboxPrimitive.Indicator className="flex items-center justify-center text-current">
        <Check size={12} />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );
});

Checkbox.displayName = "Checkbox";
