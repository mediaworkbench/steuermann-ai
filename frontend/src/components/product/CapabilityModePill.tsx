import { TonePill } from "@/components/product/TonePill";

interface CapabilityModePillProps {
  mode: string;
}

export function CapabilityModePill({ mode }: CapabilityModePillProps) {
  if (mode === "native") return <TonePill tone="success">{mode}</TonePill>;
  if (mode === "structured") return <TonePill tone="warning">{mode}</TonePill>;
  if (mode === "react") return <TonePill tone="info">{mode}</TonePill>;
  return <TonePill tone="muted">{mode}</TonePill>;
}
