"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/Button";
import { DialogCard, DialogHeader, DialogSurface } from "@/components/ui/Dialog";
import { Download, LoaderCircle } from "lucide-react";
import { iconMap } from "@/lib/iconMap";
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
    <DialogSurface open onClose={onClose} className="max-w-md">
      <DialogCard>
        <DialogHeader
          icon="download"
          iconClassName="text-primary"
          title={t("exportDialog.title")}
          onClose={onClose}
          closeLabel={t("exportDialog.closeDialog")}
        />

        <p className="mb-4 truncate text-sm text-muted-foreground">
          {conversationTitle}
        </p>

        {/* Format selector */}
        <div className="space-y-2 mb-6">
          <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
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
          <p className="mb-3 text-xs text-destructive">{error}</p>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-3">
          <Button
            variant="secondary"
            size="md"
            onClick={onClose}
          >
            {t("common.cancel")}
          </Button>
          <Button
            variant="primary"
            size="md"
            onClick={handleExport}
            disabled={exporting}
          >
            {exporting ? (
              <>
                <LoaderCircle size={16} className="animate-spin" />
                {t("exportDialog.exporting")}
              </>
            ) : (
              <>
                <Download size={16} />
                {t("common.export")}
              </>
            )}
          </Button>
        </div>
      </DialogCard>
    </DialogSurface>
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
    <Button
      type="button"
      onClick={onClick}
      variant="ghost"
      size="sm"
      className={`flex-1 cursor-pointer items-center gap-3 rounded-xl border-2 p-3 text-left transition-colors
        ${
          selected
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/30 bg-surface"
        }`}
    >
      {(() => { const LucideIcon = iconMap[icon]; return <LucideIcon size={24} className={selected ? "text-primary" : "text-foreground/30"} />; })()}
      <div>
        <span
          className={`text-sm font-bold block ${
            selected ? "text-primary" : "text-foreground/70"
          }`}
        >
          {label}
        </span>
        <span className="text-[11px] text-foreground/40">{description}</span>
      </div>
    </Button>
  );
}
