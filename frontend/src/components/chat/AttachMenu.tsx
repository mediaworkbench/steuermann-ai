"use client";

import { FileText, Image as ImageIcon, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/hooks/useI18n";

interface AttachMenuProps {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  disabled: boolean;
  onFileClick: () => void;
  onImageClick: () => void;
}

export function AttachMenu({
  open,
  onToggle,
  onClose,
  disabled,
  onFileClick,
  onImageClick,
}: AttachMenuProps) {
  const { t } = useI18n();
  return (
    <div className="relative">
      <Button
        type="button"
        disabled={disabled}
        onClick={onToggle}
        variant="ghost"
        size="sm"
        className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-surface-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
        aria-label={t("chat.addAttachment")}
      >
        <Plus size={20} />
      </Button>
      {open && (
        <>
          <div aria-hidden="true" className="fixed inset-0 z-10" onClick={onClose} />
          <div className="absolute bottom-full left-0 z-20 mb-2 min-w-40 rounded-xl border border-border bg-surface py-1 shadow-lg">
            <Button
              type="button"
              onClick={() => { onFileClick(); onClose(); }}
              variant="ghost"
              size="sm"
              className="w-full items-center gap-2.5 px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted"
            >
              <FileText size={16} className="text-muted-foreground" />
              {t("chat.addFile")}
            </Button>
            <Button
              type="button"
              onClick={() => { onImageClick(); onClose(); }}
              variant="ghost"
              size="sm"
              className="w-full items-center gap-2.5 px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted"
            >
              <ImageIcon size={16} className="text-muted-foreground" />
              {t("chat.addImage")}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
