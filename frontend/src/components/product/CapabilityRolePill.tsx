import { TonePill } from "@/components/product/TonePill";

interface CapabilityRolePillProps {
  role?: string;
}

export function CapabilityRolePill({ role }: CapabilityRolePillProps) {
  if (!role) return null;
  if (role === "chat") return <TonePill tone="info">{role}</TonePill>;
  if (role === "vision") return <TonePill tone="warning">{role}</TonePill>;
  if (role === "auxiliary") return <TonePill tone="success">{role}</TonePill>;
  return <TonePill tone="muted">{role}</TonePill>;
}
