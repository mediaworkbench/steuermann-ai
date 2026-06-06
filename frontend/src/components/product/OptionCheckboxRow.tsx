import * as React from "react";
import { Checkbox } from "@/components/ui/Checkbox";
import { cn } from "@/lib/utils";

interface OptionCheckboxRowProps {
  checked: boolean;
  onToggle: () => void;
  label: React.ReactNode;
  description?: React.ReactNode;
  alignment?: "start" | "center";
  checkboxClassName?: string;
  descriptionClassName?: string;
  className?: string;
}

export function OptionCheckboxRow({
  checked,
  onToggle,
  label,
  description,
  alignment = "start",
  checkboxClassName,
  descriptionClassName,
  className,
}: OptionCheckboxRowProps) {
  return (
    <label
      className={cn(
        "flex gap-3 cursor-pointer",
        alignment === "center" ? "items-center" : "items-start",
        className
      )}
    >
      <Checkbox
        type="checkbox"
        checked={checked}
        onChange={onToggle}
        className={cn(alignment === "center" ? "" : "mt-0.5", checkboxClassName)}
      />
      <span className="flex flex-col">
        <span className="text-sm font-medium text-foreground">{label}</span>
        {description ? (
          <span className={cn("mt-0.5 text-xs text-foreground/60", descriptionClassName)}>
            {description}
          </span>
        ) : null}
      </span>
    </label>
  );
}
