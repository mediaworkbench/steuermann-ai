import { useId } from "react";

import { Checkbox } from "@/components/ui/checkbox";

interface MetricsSeriesOption {
  key: string;
  label: string;
  checked: boolean;
  onToggle: (checked: boolean) => void;
}

interface MetricsSeriesOptionsProps {
  options: MetricsSeriesOption[];
}

export function MetricsSeriesOptions({ options }: MetricsSeriesOptionsProps) {
  const idPrefix = useId();
  return (
    <div className="mb-4 flex flex-wrap gap-4 text-xs text-muted-foreground">
      {options.map((option) => (
        <label key={option.key} htmlFor={`${idPrefix}-${option.key}`} className="inline-flex items-center gap-1.5">
          <Checkbox
            id={`${idPrefix}-${option.key}`}
            checked={option.checked}
            onChange={(event) => option.onToggle(event.target.checked)}
          />
          {option.label}
        </label>
      ))}
    </div>
  );
}
