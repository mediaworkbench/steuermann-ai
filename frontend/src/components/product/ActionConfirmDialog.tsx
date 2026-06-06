import { ConfirmDialog } from "@/components/ConfirmDialog";

interface ActionConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
}

export function ActionConfirmDialog({
  isOpen,
  title,
  message,
  onConfirm,
  onCancel,
}: ActionConfirmDialogProps) {
  return (
    <ConfirmDialog
      isOpen={isOpen}
      title={title}
      message={message}
      variant="default"
      onConfirm={onConfirm}
      onCancel={onCancel}
    />
  );
}
