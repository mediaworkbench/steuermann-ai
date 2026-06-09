interface CapabilityRolePillProps {
  role?: string;
}

export function CapabilityRolePill({ role }: CapabilityRolePillProps) {
  if (!role) return null;
  if (role === "chat") return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-info/10 text-info">{role}</span>;
  if (role === "vision") return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-warning/10 text-warning">{role}</span>;
  if (role === "auxiliary") return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-success/10 text-success">{role}</span>;
  return <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-muted text-foreground">{role}</span>;
}
