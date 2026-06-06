import { TonePill } from "@/components/product/TonePill";

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
      <TonePill tone="success">{nativeLabel}</TonePill>
      <TonePill tone="warning">{structuredLabel}</TonePill>
      <TonePill tone="info">{reactLabel}</TonePill>
    </div>
  );
}
