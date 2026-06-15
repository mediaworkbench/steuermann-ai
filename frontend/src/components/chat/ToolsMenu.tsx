"use client";

import { Wrench } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SystemConfig } from "@/lib/api";

const FALLBACK_TOOLS = [
  { id: "web_search_mcp", label: "Web Search" },
  { id: "extract_webpage_mcp", label: "Extract Webpage" },
  { id: "analyze_image_tool", label: "Analyze Image" },
  { id: "ocr_tool", label: "OCR" },
  { id: "analyze_document_tool", label: "Analyze Document" },
  { id: "analyze_chart_tool", label: "Analyze Chart" },
  { id: "image_metadata_tool", label: "Image Metadata" },
  { id: "read_barcodes_tool", label: "Read Barcodes" },
  { id: "datetime_tool", label: "Datetime" },
  { id: "calculator_tool", label: "Calculator" },
  { id: "map_tool", label: "Map" },
  { id: "csv_analyze_tool", label: "CSV Analyze" },
  { id: "file_ops_tool", label: "File Ops" },
] as const;

interface ToolsMenuProps {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  systemConfig: SystemConfig | null;
  toolToggles: Record<string, boolean>;
  onToolToggle: (toolId: string) => void;
  allowedTools: string[] | null; // role-allowed tool ids; null = no restriction
}

export function ToolsMenu({
  open,
  onToggle,
  onClose,
  systemConfig,
  toolToggles,
  onToolToggle,
  allowedTools,
}: ToolsMenuProps) {
  const visibleTools = (systemConfig?.available_tools ?? FALLBACK_TOOLS).filter(
    (tool) => !allowedTools || allowedTools.includes(tool.id)
  );
  return (
    <div className="relative">
      <Button
        type="button"
        onClick={onToggle}
        variant="ghost"
        size="sm"
        className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-surface-muted hover:text-foreground"
        aria-label="Tools"
      >
        <Wrench size={20} />
      </Button>
      {open && (
        <>
          <div aria-hidden="true" className="fixed inset-0 z-10" onClick={onClose} />
          <div className="absolute bottom-full left-0 z-20 mb-2 min-w-50 rounded-xl border border-border bg-surface py-2 shadow-lg">
            <p className="px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Tools</p>
            {visibleTools.map((tool) => {
              const enabled = toolToggles[tool.id] !== false;
              return (
                <Button
                  key={tool.id}
                  type="button"
                  onClick={() => onToolToggle(tool.id)}
                  variant="ghost"
                  size="sm"
                  className="w-full items-center justify-between gap-3 px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted"
                >
                  <span>{tool.label}</span>
                  <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wide transition-colors ${enabled ? "bg-primary/15 text-primary" : "bg-surface-muted text-muted-foreground"}`}>
                    {enabled ? "ON" : "OFF"}
                  </span>
                </Button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
