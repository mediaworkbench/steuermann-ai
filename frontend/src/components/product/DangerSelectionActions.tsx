import { Button } from "@/components/ui/button";

interface DangerSelectionActionsProps {
  hasSelection: boolean;
  hintText: string;
  onAction: () => void;
  loading: boolean;
  loadingLabel: string;
  actionLabel: string;
  disabled?: boolean;
}

export function DangerSelectionActions({
  hasSelection,
  hintText,
  onAction,
  loading,
  loadingLabel,
  actionLabel,
  disabled,
}: DangerSelectionActionsProps) {
  return (
    <>
      {!hasSelection && <p className="mb-3 text-xs text-warning">{hintText}</p>}
      <Button
        type="button"
        onClick={onAction}
        disabled={!hasSelection || disabled || loading}
        variant="destructive"
      >
        {loading ? loadingLabel : actionLabel}
      </Button>
    </>
  );
}
