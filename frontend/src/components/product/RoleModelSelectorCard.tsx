import { Select } from "@/components/ui/select";

interface RoleModelSelectorCardProps {
  roleName: string;
  roleLabel: string;
  providerLabel: string;
  systemDefaultLabel: string;
  selectedModel: string;
  modelOptions: string[];
  modelLoadError?: string | null;
  onModelChange: (value: string) => void;
}

export function RoleModelSelectorCard({
  roleName,
  roleLabel,
  providerLabel,
  systemDefaultLabel,
  selectedModel,
  modelOptions,
  modelLoadError,
  onModelChange,
}: RoleModelSelectorCardProps) {
  return (
    <div className="p-4">
      <div className="mb-2">
        <p className="text-sm font-semibold text-foreground">{roleLabel}</p>
        <p className="text-xs text-muted-foreground">{providerLabel}</p>
        <p className="text-xs text-muted-foreground">{systemDefaultLabel}</p>
        {modelLoadError ? <p className="text-xs text-warning">{modelLoadError}</p> : null}
      </div>
      <Select value={selectedModel} onChange={(e) => onModelChange(e.target.value)} aria-label={roleLabel}>
        {modelOptions.map((model) => (
          <option key={`${roleName}:${model}`} value={model}>
            {model}
          </option>
        ))}
      </Select>
    </div>
  );
}
