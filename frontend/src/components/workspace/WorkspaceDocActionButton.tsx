import { Button } from "@/components/ui/Button";
import { Icon } from "../Icon";
import { cn } from "@/lib/utils";

interface WorkspaceDocActionButtonProps {
  label: string;
  icon: string;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
  variant?: "primary" | "secondary" | "destructive" | "ghost";
  className?: string;
}

export function WorkspaceDocActionButton({
  label,
  icon,
  onClick,
  disabled,
  title,
  variant = "ghost",
  className,
}: WorkspaceDocActionButtonProps) {
  return (
    <Button
      type="button"
      onClick={onClick}
      disabled={disabled}
      variant={variant}
      size="sm"
      className={cn(
        "flex-1 min-w-fit px-2.5 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        className
      )}
      title={title ?? label}
    >
      <Icon name={icon} size={14} className="mr-1 inline" />
      {label}
    </Button>
  );
}
