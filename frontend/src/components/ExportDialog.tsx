"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Code2, Download, FileText, LoaderCircle } from "lucide-react";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
      toast.success(t("exportDialog.exportedSuccessfully"), {
        description: filename,
      });
      onClose();
    } catch {
      setError(t("exportDialog.exportFailedTryAgain"));
      toast.error(t("exportDialog.exportFailed"));
    } finally {
      setExporting(false);
    }
  };

  return (
    <AlertDialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogMedia>
            <Download className="size-6 text-primary" />
          </AlertDialogMedia>
          <AlertDialogTitle>{t("exportDialog.title")}</AlertDialogTitle>
          <AlertDialogDescription className="truncate">
            {conversationTitle}
          </AlertDialogDescription>
        </AlertDialogHeader>

        <div className="space-y-2 px-2">
          <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
            {t("exportDialog.format")}
          </label>
          <div className="flex gap-3">
            <FormatOption
              icon={FileText}
              label={t("exportDialog.markdown")}
              description={t("exportDialog.markdownDescription")}
              selected={format === "markdown"}
              onClick={() => setFormat("markdown")}
            />
            <FormatOption
              icon={Code2}
              label={t("exportDialog.json")}
              description={t("exportDialog.jsonDescription")}
              selected={format === "json"}
              onClick={() => setFormat("json")}
            />
          </div>
        </div>

        {error && (
          <p className="px-2 text-xs text-destructive">{error}</p>
        )}

        <AlertDialogFooter>
          <Button variant="secondary" size="md" onClick={onClose}>
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
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function FormatOption({
  icon: Icon,
  label,
  description,
  selected,
  onClick,
}: {
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex flex-1 cursor-pointer items-center gap-3 rounded-xl border-2 p-3 text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/50 ${
        selected
          ? "border-primary bg-primary/5"
          : "border-border bg-surface hover:border-primary/30"
      }`}
    >
      <Icon
        size={24}
        className={selected ? "shrink-0 text-primary" : "shrink-0 text-foreground/30"}
      />
      <div>
        <span
          className={`block text-sm font-bold ${
            selected ? "text-primary" : "text-foreground/70"
          }`}
        >
          {label}
        </span>
        <span className="text-[11px] text-foreground/40">{description}</span>
      </div>
    </button>
  );
}
