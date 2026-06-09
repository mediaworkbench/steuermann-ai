import { Button } from "@/components/ui/button";
import { FolderOpen, X } from "lucide-react";

interface WorkspacePanelTopBarProps {
  title: string;
  onClose: () => void;
  closeLabel: string;
}

export function WorkspacePanelTopBar({ title, onClose, closeLabel }: WorkspacePanelTopBarProps) {
  return (
    <div className="shrink-0 border-b border-border px-3 py-2.5 flex items-center justify-between">
      <div className="min-w-0 flex items-center gap-2">
        <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary">
          <FolderOpen size={16} />
        </span>
        <h3 className="truncate text-sm font-semibold tracking-tight text-foreground">{title}</h3>
      </div>
      <Button
        type="button"
        onClick={onClose}
        variant="ghost"
        size="sm"
        className="p-1.5 text-muted-foreground md:hidden"
        aria-label={closeLabel}
      >
        <X size={18} />
      </Button>
    </div>
  );
}
