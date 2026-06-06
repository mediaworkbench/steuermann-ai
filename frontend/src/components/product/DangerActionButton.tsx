import { Button } from "@/components/ui/Button";

interface DangerActionButtonProps {
  label: string;
  loadingLabel: string;
  loading: boolean;
  disabled?: boolean;
  onClick: () => void;
  type?: "button" | "submit" | "reset";
}

export function DangerActionButton({
  label,
  loadingLabel,
  loading,
  disabled,
  onClick,
  type = "button",
}: DangerActionButtonProps) {
  return (
    <Button type={type} onClick={onClick} disabled={disabled || loading} variant="destructive">
      {loading ? loadingLabel : label}
    </Button>
  );
}
