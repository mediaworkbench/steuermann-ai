import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import type { LLMCapabilityItem } from "@/lib/api";
import { CapabilitiesTable, type CapabilitiesTableLabels } from "@/components/product/CapabilitiesTable";
import { DiagnosticsLegend } from "@/components/product/DiagnosticsLegend";

interface DiagnosticsSectionCardProps {
  title: ReactNode;
  description: ReactNode;
  items: LLMCapabilityItem[];
  loading: boolean;
  error: string | null;
  expandedRows: Record<string, boolean>;
  onToggleRow: (key: string) => void;
  formatDateTime: (value: string) => string;
  labels: CapabilitiesTableLabels;
  legendTitle: string;
  legendNative: string;
  legendStructured: string;
  legendReact: string;
  loadingLabel: string;
  emptyLabel: string;
  copyLabel: string;
  copyingLabel: string;
  refreshLabel: string;
  copying: boolean;
  onCopy: () => void;
  onRefresh: () => void;
}

export function DiagnosticsSectionCard({
  title,
  description,
  items,
  loading,
  error,
  expandedRows,
  onToggleRow,
  formatDateTime,
  labels,
  legendTitle,
  legendNative,
  legendStructured,
  legendReact,
  loadingLabel,
  emptyLabel,
  copyLabel,
  copyingLabel,
  refreshLabel,
  copying,
  onCopy,
  onRefresh,
}: DiagnosticsSectionCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              type="button"
              onClick={onCopy}
              disabled={loading || copying || items.length === 0}
              variant="secondary"
              size="sm"
            >
              {copying ? copyingLabel : copyLabel}
            </Button>
            <Button
              type="button"
              onClick={onRefresh}
              disabled={loading}
              variant="secondary"
              size="sm"
            >
              {refreshLabel}
            </Button>
          </div>
        </div>
      </CardHeader>
      <div className="px-6 pb-6">
      <DiagnosticsLegend
        title={legendTitle}
        nativeLabel={legendNative}
        structuredLabel={legendStructured}
        reactLabel={legendReact}
      />

      {loading ? (
        <p className="text-sm text-muted-foreground">{loadingLabel}</p>
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-muted-foreground">{emptyLabel}</p>
      ) : (
        <CapabilitiesTable
          items={items}
          expandedRows={expandedRows}
          onToggleRow={onToggleRow}
          formatDateTime={formatDateTime}
          labels={labels}
        />
      )}
      </div>
    </Card>
  );
}
