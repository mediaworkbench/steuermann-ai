"use client";

import { useEffect, useState } from "react";
import { Icon } from "./Icon";
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
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => {
        if (e.target === e.currentTarget) handleCancel();
      }}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm p-6 mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-evergreen flex items-center gap-2">
            <Icon
              name={isDanger ? "warning" : "help_outline"}
              size={20}
              className={isDanger ? "text-red-500" : "text-pacific-blue"}
            />
            {props.title}
          </h3>
          <button
            onClick={handleCancel}
            className="p-1 rounded hover:bg-gray-100 transition-colors text-evergreen/50 hover:text-evergreen cursor-pointer"
            aria-label={t("common.close")}
          >
            <Icon name="close" size={20} />
          </button>
        </div>

        <p className="text-sm text-evergreen/70 mb-6">{props.message}</p>

        {typed && (
          <div className="mb-6">
            {props.inputLabel && (
              <label className="block text-xs font-medium text-evergreen/60 mb-1">
                {props.inputLabel}
              </label>
            )}
            <input
              type="text"
              autoFocus
              value={typedValue}
              onChange={(e) => setTypedValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleConfirm(); }}
              placeholder={t("confirmDialog.typeToConfirmPlaceholder")}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-pacific-blue text-sm font-mono tracking-wider"
            />
          </div>
        )}

        {checkboxed && (
          <label className="flex items-center gap-3 mb-6 cursor-pointer select-none group">
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500 cursor-pointer"
            />
            <span className="text-sm text-evergreen/70 group-hover:text-evergreen transition-colors">
              {props.checkboxLabel}
            </span>
          </label>
        )}

        <div className="flex items-center justify-end gap-3">
          <button
            onClick={handleCancel}
            className="text-sm text-evergreen/60 hover:text-evergreen px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
          >
            {props.cancelLabel ?? t("common.cancel")}
          </button>
          <button
            onClick={handleConfirm}
            disabled={!!confirmDisabled}
            className={`text-sm text-white px-5 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 cursor-pointer
              ${isDanger ? "bg-red-600 hover:bg-red-700" : "bg-pacific-blue hover:bg-pacific-blue/80"}`}
          >
            {props.confirmLabel ?? t("common.confirm")}
          </button>
        </div>
      </div>
    </div>
  );
}
