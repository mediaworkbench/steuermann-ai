interface CapabilityModePillProps {
  mode: string;
}

export function CapabilityModePill({ mode }: CapabilityModePillProps) {
  if (mode === "native") return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-success/10 text-success">{mode}</span>;
  if (mode === "structured") return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-warning/10 text-warning">{mode}</span>;
  if (mode === "react") return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-info/10 text-info">{mode}</span>;
  return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-muted text-foreground">{mode}</span>;
}
