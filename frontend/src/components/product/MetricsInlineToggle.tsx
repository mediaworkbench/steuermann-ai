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
  return (
    <label className={className}>
      <input type="checkbox" checked={checked} onChange={(event) => onToggle(event.target.checked)} />
      {label}
    </label>
  );
}
