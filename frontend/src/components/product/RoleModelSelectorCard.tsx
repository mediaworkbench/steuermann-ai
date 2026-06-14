import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";

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
      <Select value={selectedModel} onValueChange={onModelChange}>
        <SelectTrigger aria-label={roleLabel} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {modelOptions.map((model) => (
            <SelectItem key={`${roleName}:${model}`} value={model}>
              {model}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
