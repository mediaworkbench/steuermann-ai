import * as React from "react";
import { Button } from "@/components/ui/Button";
import { Icon } from "../Icon";
import { cn } from "@/lib/utils";

interface WorkspacePanelHeaderRowProps {
  title: React.ReactNode;
  onClose?: () => void;
  closeLabel?: string;
  className?: string;
  titleClassName?: string;
}

export function WorkspacePanelHeaderRow({
  title,
  onClose,
  closeLabel,
  className,
  titleClassName,
}: WorkspacePanelHeaderRowProps) {
  return (
    <div className={cn("mb-2 flex items-center justify-between", className)}>
      <p className={cn("text-xs font-semibold uppercase tracking-wide text-muted-foreground", titleClassName)}>
        {title}
      </p>
      {onClose ? (
        <Button
          type="button"
          onClick={onClose}
          variant="ghost"
          size="sm"
          className="p-0 text-muted-foreground hover:text-foreground"
          aria-label={closeLabel}
        >
          <Icon name="close" size={14} />
        </Button>
      ) : null}
    </div>
  );
}
