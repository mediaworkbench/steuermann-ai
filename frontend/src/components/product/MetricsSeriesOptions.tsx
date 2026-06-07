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
  return (
    <div className="mb-4 flex flex-wrap gap-4 text-xs text-muted-foreground">
      {options.map((option) => (
        <label key={option.key} className="inline-flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={option.checked}
            onChange={(event) => option.onToggle(event.target.checked)}
          />
          {option.label}
        </label>
      ))}
    </div>
  );
}
