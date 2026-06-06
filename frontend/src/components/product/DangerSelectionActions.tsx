import { DangerActionButton } from "@/components/product/DangerActionButton";
import { DangerHintText } from "@/components/product/DangerHintText";

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
      {!hasSelection && <DangerHintText>{hintText}</DangerHintText>}
      <DangerActionButton
        onClick={onAction}
        disabled={!hasSelection || disabled}
        loading={loading}
        loadingLabel={loadingLabel}
        label={actionLabel}
      />
    </>
  );
}
