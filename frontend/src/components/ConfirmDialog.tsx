"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Checkbox } from "@/components/ui/Checkbox";
import { DialogCard, DialogHeader, DialogSurface } from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { useI18n } from "@/hooks/useI18n";

interface ConfirmDialogBaseProps {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "default" | "danger";
}

interface TypedConfirmDialogProps extends ConfirmDialogBaseProps {
  requireTyped: string;
  inputLabel?: string;
}

interface CheckedConfirmDialogProps extends ConfirmDialogBaseProps {
  requireChecked: true;
  checkboxLabel: string;
}

export type ConfirmDialogProps =
  | ConfirmDialogBaseProps
  | TypedConfirmDialogProps
  | CheckedConfirmDialogProps;

function isTypedVariant(props: ConfirmDialogProps): props is TypedConfirmDialogProps {
  return "requireTyped" in props;
}

function isCheckedVariant(props: ConfirmDialogProps): props is CheckedConfirmDialogProps {
  return "requireChecked" in props;
}

export function ConfirmDialog(props: ConfirmDialogProps) {
  const { t } = useI18n();
  const [typedValue, setTypedValue] = useState("");
  const [checked, setChecked] = useState(false);

  const typed = isTypedVariant(props);
  const checkboxed = isCheckedVariant(props);
  const isDanger = props.variant === "danger";
  const confirmDisabled =
    (typed && typedValue.trim().toUpperCase() !== props.requireTyped.toUpperCase()) ||
    (checkboxed && !checked);

  const handleCancel = () => {
    setTypedValue("");
    setChecked(false);
    props.onCancel();
  };

  const handleConfirm = () => {
    if (confirmDisabled) return;
    setTypedValue("");
    setChecked(false);
    props.onConfirm();
  };

  const { onCancel } = props;
  useEffect(() => {
    if (!props.isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setTypedValue("");
        setChecked(false);
        onCancel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props.isOpen, onCancel]);

  if (!props.isOpen) return null;

  return (
    <DialogSurface open={props.isOpen} onClose={handleCancel} className="max-w-sm">
      <DialogCard>
        <DialogHeader
          icon={isDanger ? "warning" : "help_outline"}
          iconClassName={isDanger ? "text-destructive" : "text-primary"}
          title={props.title}
          onClose={handleCancel}
          closeLabel={t("common.close")}
        />

        <p className="mb-6 text-sm text-muted-foreground">{props.message}</p>

        {typed && (
          <div className="mb-6">
            {props.inputLabel && (
              <label className="mb-1 block text-xs font-medium text-muted-foreground">
                {props.inputLabel}
              </label>
            )}
            <Input
              type="text"
              autoFocus
              value={typedValue}
              onChange={(e) => setTypedValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleConfirm();
              }}
              placeholder={t("confirmDialog.typeToConfirmPlaceholder")}
              className="rounded-lg font-mono tracking-wider"
            />
          </div>
        )}

        {checkboxed && (
          <label className="flex items-center gap-3 mb-6 cursor-pointer select-none group">
            <Checkbox
              type="checkbox"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
              className="cursor-pointer"
            />
            <span className="text-sm text-muted-foreground transition-colors group-hover:text-foreground">
              {props.checkboxLabel}
            </span>
          </label>
        )}

        <div className="flex items-center justify-end gap-3">
          <Button
            variant="secondary"
            size="md"
            onClick={handleCancel}
          >
            {props.cancelLabel ?? t("common.cancel")}
          </Button>
          <Button
            variant={isDanger ? "destructive" : "primary"}
            size="md"
            onClick={handleConfirm}
            disabled={!!confirmDisabled}
          >
            {props.confirmLabel ?? t("common.confirm")}
          </Button>
        </div>
      </DialogCard>
    </DialogSurface>
  );
}
