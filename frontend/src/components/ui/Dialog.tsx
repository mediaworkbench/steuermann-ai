import * as React from "react";
import { Icon } from "@/components/Icon";

interface DialogSurfaceProps {
  open: boolean;
  onClose: () => void;
  className?: string;
  children: React.ReactNode;
}

export function DialogSurface({ open, onClose, className = "", children }: DialogSurfaceProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div role="dialog" aria-modal="true" className={`w-full ${className}`.trim()}>
        {children}
      </div>
    </div>
  );
}

interface DialogCardProps {
  children: React.ReactNode;
  className?: string;
}

export function DialogCard({ children, className = "" }: DialogCardProps) {
  return (
    <div className={`rounded-2xl border border-border bg-surface p-6 shadow-2xl ${className}`.trim()}>
      {children}
    </div>
  );
}

interface DialogHeaderProps {
  icon?: string;
  iconClassName?: string;
  title: React.ReactNode;
  onClose: () => void;
  closeLabel: string;
}

export function DialogHeader({ icon, iconClassName = "", title, onClose, closeLabel }: DialogHeaderProps) {
  return (
    <div className="mb-4 flex items-center justify-between gap-4">
      <h3 className="flex items-center gap-2 text-lg font-bold text-foreground">
        {icon ? <Icon name={icon} size={20} className={iconClassName} /> : null}
        {title}
      </h3>
      <button
        type="button"
        onClick={onClose}
        className="rounded-md p-1 text-foreground/50 transition-colors hover:bg-surface-muted hover:text-foreground"
        aria-label={closeLabel}
      >
        <Icon name="close" size={20} />
      </button>
    </div>
  );
}
