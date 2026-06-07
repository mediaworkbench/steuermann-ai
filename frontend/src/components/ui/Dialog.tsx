import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import { iconMap } from "@/lib/iconMap";

/* ──────────────────────────────────────────────────────────────────────────
 * DialogSurface
 * Wraps Radix Dialog.Root + Portal + Overlay + Content.
 * Keeps the same external props so ConfirmDialog and ExportDialog need no
 * changes. Focus trap, scroll lock, and ESC handling come from Radix.
 * ──────────────────────────────────────────────────────────────────────────
 */
interface DialogSurfaceProps {
  open: boolean;
  onClose: () => void;
  className?: string;
  children: React.ReactNode;
}

export function DialogSurface({ open, onClose, className = "", children }: DialogSurfaceProps) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/40" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className={cn(
            "fixed left-1/2 top-1/2 z-50 w-full -translate-x-1/2 -translate-y-1/2 p-4",
            "focus:outline-none",
            className
          )}
        >
          {children}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
 * DialogCard — styled container, unchanged.
 * ──────────────────────────────────────────────────────────────────────────
 */
interface DialogCardProps {
  children: React.ReactNode;
  className?: string;
}

export function DialogCard({ children, className = "" }: DialogCardProps) {
  return (
    <div className={cn("rounded-2xl border border-border bg-surface p-6 shadow-2xl", className)}>
      {children}
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
 * DialogHeader — styled header row, unchanged.
 * Uses DialogPrimitive.Title so the dialog has an accessible name.
 * ──────────────────────────────────────────────────────────────────────────
 */
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
      <DialogPrimitive.Title asChild>
        <h3 className="flex items-center gap-2 text-lg font-bold text-foreground">
          {icon ? (() => { const LucideIcon = iconMap[icon]; return <LucideIcon size={20} className={iconClassName} />; })() : null}
          {title}
        </h3>
      </DialogPrimitive.Title>
      <DialogPrimitive.Close asChild>
        <button
          type="button"
          onClick={onClose}
          className="rounded-md p-1 text-foreground/50 transition-colors hover:bg-surface-muted hover:text-foreground"
          aria-label={closeLabel}
        >
          <X size={20} />
        </button>
      </DialogPrimitive.Close>
    </div>
  );
}
