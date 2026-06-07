interface DiagnosticsLegendProps {
  title: string;
  nativeLabel: string;
  structuredLabel: string;
  reactLabel: string;
}

export function DiagnosticsLegend({
  title,
  nativeLabel,
  structuredLabel,
  reactLabel,
}: DiagnosticsLegendProps) {
  return (
    <div className="mb-4 flex flex-wrap gap-2">
      <span className="text-xs font-semibold text-muted-foreground">{title}</span>
      <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-success/10 text-success">{nativeLabel}</span>
      <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-warning/10 text-warning">{structuredLabel}</span>
      <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-info/10 text-info">{reactLabel}</span>
    </div>
  );
}
