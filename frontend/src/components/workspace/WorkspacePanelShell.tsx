import * as React from "react";
import { PanelRightOpen } from "lucide-react";
import { cn } from "@/lib/utils";

interface WorkspacePanelShellProps {
  isOpen: boolean;
  children: React.ReactNode;
  className?: string;
  onToggle?: () => void;
}

export function WorkspacePanelShell({ isOpen, children, className, onToggle }: WorkspacePanelShellProps) {
  return (
    <div
      className={cn(
        "fixed right-0 top-16 z-10 flex h-[calc(100vh-4rem)] w-80 min-h-0 transition-all duration-200",
        "md:sticky md:top-20 md:z-0 md:h-[calc(100vh-5rem)] md:self-start",
        isOpen ? "translate-x-0 md:w-80 lg:w-96" : "translate-x-full md:w-0 md:translate-x-0",
        className
      )}
    >
      {/* Workspace toggle badge */}
      {onToggle && (
        <button
          onClick={onToggle}
          className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-full z-10 flex items-center justify-center
                     w-7 h-14 rounded-l-md border border-border border-r-0 bg-sidebar-muted
                     text-muted-foreground hover:text-foreground hover:bg-sidebar-accent
                     transition-colors cursor-pointer"
          aria-label="Toggle workspace sidebar"
        >
          <PanelRightOpen size={16} />
        </button>
      )}
      <div className="flex flex-col overflow-hidden min-h-0 flex-1 border-l border-border bg-sidebar-muted">
        {children}
      </div>
    </div>
  );
}
