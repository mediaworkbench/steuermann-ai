"use client";

import { useRef, useState } from "react";
import { AlertTriangle, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
  const confirmedRef = useRef(false);

  const typed = isTypedVariant(props);
  const checkboxed = isCheckedVariant(props);
  const isDanger = props.variant === "danger";
  const confirmDisabled =
    (typed && typedValue.trim().toUpperCase() !== props.requireTyped.toUpperCase()) ||
    (checkboxed && !checked);

  const reset = () => {
    setTypedValue("");
    setChecked(false);
  };

  const handleConfirm = () => {
    if (confirmDisabled) return;
    confirmedRef.current = true;
    reset();
    props.onConfirm();
  };

  const handleCancel = () => {
    if (confirmedRef.current) {
      confirmedRef.current = false;
      return;
    }
    reset();
    props.onCancel();
  };

  return (
    <AlertDialog
      open={props.isOpen}
      onOpenChange={(open) => {
        if (!open) handleCancel();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia>
            {isDanger ? (
              <AlertTriangle className="size-6 text-destructive" />
            ) : (
              <HelpCircle className="size-6 text-primary" />
            )}
          </AlertDialogMedia>
          <AlertDialogTitle>{props.title}</AlertDialogTitle>
          <AlertDialogDescription>{props.message}</AlertDialogDescription>
        </AlertDialogHeader>

        {typed && (
          <div className="px-2">
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
          <label className="flex items-center gap-3 px-2 cursor-pointer select-none group">
            <Checkbox
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
              className="cursor-pointer"
            />
            <span className="text-sm text-muted-foreground transition-colors group-hover:text-foreground">
              {props.checkboxLabel}
            </span>
          </label>
        )}

        <AlertDialogFooter>
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
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
