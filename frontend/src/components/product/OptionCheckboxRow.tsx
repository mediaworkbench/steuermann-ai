import * as React from "react";
import { Checkbox } from "@/components/ui/Checkbox";
import { cn } from "@/lib/utils";

interface OptionCheckboxRowProps {
  checked: boolean;
  onToggle: () => void;
  label: React.ReactNode;
  description: React.ReactNode;
  className?: string;
}

export function OptionCheckboxRow({
  checked,
  onToggle,
  label,
  description,
  className,
}: OptionCheckboxRowProps) {
  return (
    <label className={cn("flex items-start gap-3 cursor-pointer", className)}>
      <Checkbox type="checkbox" checked={checked} onChange={onToggle} className="mt-0.5" />
      <span className="flex flex-col">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="mt-0.5 text-xs text-foreground/60">{description}</span>
      </span>
    </label>
  );
}
