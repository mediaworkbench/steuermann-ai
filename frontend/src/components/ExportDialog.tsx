"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Icon } from "./Icon";
import { exportConversation } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

interface ExportDialogProps {
  conversationId: string;
  conversationTitle: string;
  onClose: () => void;
}

export function ExportDialog({
  conversationId,
  conversationTitle,
  onClose,
}: ExportDialogProps) {
  const { t } = useI18n();
  const [format, setFormat] = useState<"json" | "markdown">("markdown");
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setExporting(true);
    setError(null);
    try {
      const data = await exportConversation(conversationId, format);
      if (!data) {
        setError(t("exportDialog.exportFailedNoData"));
        return;
      }

      const content =
        typeof data === "string" ? data : JSON.stringify(data, null, 2);
      const mimeType =
        format === "markdown" ? "text/markdown" : "application/json";
      const ext = format === "markdown" ? "md" : "json";
      const filename = `${conversationTitle.replace(/[^a-zA-Z0-9_-]/g, "_")}.${ext}`;

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success(t("exportDialog.exportedSuccessfully"), { description: filename });
      onClose();
    } catch {
      setError(t("exportDialog.exportFailedTryAgain"));
      toast.error(t("exportDialog.exportFailed"));
    } finally {
      setExporting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-evergreen flex items-center gap-2">
            <Icon name="download" size={20} className="text-pacific-blue" />
            {t("exportDialog.title")}
          </h3>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-100 transition-colors text-evergreen/50 hover:text-evergreen cursor-pointer"
            aria-label={t("exportDialog.closeDialog")}
          >
            <Icon name="close" size={20} />
          </button>
        </div>

        <p className="text-sm text-evergreen/60 mb-4 truncate">
          {conversationTitle}
        </p>

        {/* Format selector */}
        <div className="space-y-2 mb-6">
          <label className="text-xs font-bold text-evergreen/50 uppercase tracking-wider">
            {t("exportDialog.format")}
          </label>
          <div className="flex gap-3">
            <FormatOption
              icon="description"
              label={t("exportDialog.markdown")}
              description={t("exportDialog.markdownDescription")}
              selected={format === "markdown"}
              onClick={() => setFormat("markdown")}
            />
            <FormatOption
              icon="data_object"
              label={t("exportDialog.json")}
              description={t("exportDialog.jsonDescription")}
              selected={format === "json"}
              onClick={() => setFormat("json")}
            />
          </div>
        </div>

        {error && (
          <p className="text-burnt-tangerine text-xs mb-3">{error}</p>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            className="text-sm text-evergreen/60 hover:text-evergreen px-4 py-2 rounded-lg
                       hover:bg-gray-100 transition-colors cursor-pointer"
          >
            {t("common.cancel")}
          </button>
          <button
            onClick={handleExport}
            disabled={exporting}
            className="text-sm text-white bg-pacific-blue hover:bg-pacific-blue/80 px-5 py-2
                       rounded-lg font-medium transition-colors disabled:opacity-50 cursor-pointer
                       flex items-center gap-2"
          >
            {exporting ? (
              <>
                <Icon name="progress_activity" size={16} className="animate-spin" />
                {t("exportDialog.exporting")}
              </>
            ) : (
              <>
                <Icon name="download" size={16} />
                {t("common.export")}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function FormatOption({
  icon,
  label,
  description,
  selected,
  onClick,
}: {
  icon: string;
  label: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center gap-3 p-3 rounded-xl border-2 text-left transition-colors cursor-pointer
        ${
          selected
            ? "border-pacific-blue bg-pacific-blue/5"
            : "border-gray-200 hover:border-pacific-blue/30 bg-white"
        }`}
    >
      <Icon
        name={icon}
        size={24}
        className={selected ? "text-pacific-blue" : "text-evergreen/30"}
      />
      <div>
        <span
          className={`text-sm font-bold block ${
            selected ? "text-pacific-blue" : "text-evergreen/70"
          }`}
        >
          {label}
        </span>
        <span className="text-[11px] text-evergreen/40">{description}</span>
      </div>
    </button>
  );
}
