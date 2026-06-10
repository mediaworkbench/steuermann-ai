import { useId } from "react";

import { Checkbox } from "@/components/ui/checkbox";

interface MetricsInlineToggleProps {
  checked: boolean;
  onToggle: (checked: boolean) => void;
  label: string;
  className?: string;
}

export function MetricsInlineToggle({
  checked,
  onToggle,
  label,
  className = "inline-flex items-center gap-2 text-sm text-muted-foreground",
}: MetricsInlineToggleProps) {
  const id = useId();
  return (
    <label htmlFor={id} className={className}>
      <Checkbox id={id} checked={checked} onChange={(event) => onToggle(event.target.checked)} />
      {label}
    </label>
  );
}
