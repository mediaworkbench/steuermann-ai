import { ConfirmDialog } from "@/components/ConfirmDialog";

interface DangerConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel: string;
  checkboxLabel: string;
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
}

export function DangerConfirmDialog({
  isOpen,
  title,
  message,
  confirmLabel,
  checkboxLabel,
  onConfirm,
  onCancel,
}: DangerConfirmDialogProps) {
  return (
    <ConfirmDialog
      isOpen={isOpen}
      title={title}
      message={message}
      confirmLabel={confirmLabel}
      requireChecked
      checkboxLabel={checkboxLabel}
      onConfirm={onConfirm}
      onCancel={onCancel}
      variant="danger"
    />
  );
}
