import { Button } from "@/components/ui/button";

interface PrimarySaveBarProps {
  label: string;
  loadingLabel: string;
  saving: boolean;
  disabled?: boolean;
  onSave: () => void;
}

export function PrimarySaveBar({
  label,
  loadingLabel,
  saving,
  disabled,
  onSave,
}: PrimarySaveBarProps) {
  return (
    <div className="flex gap-3">
      <Button onClick={onSave} disabled={disabled || saving} className="flex-1">
        {saving ? loadingLabel : label}
      </Button>
    </div>
  );
}
