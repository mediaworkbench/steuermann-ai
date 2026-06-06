import { Fragment } from "react";
import { Button } from "@/components/ui/Button";
import { CapabilityDetailsPanel } from "@/components/product/CapabilityDetailsPanel";
import { CapabilityModePill } from "@/components/product/CapabilityModePill";
import { CapabilityRolePill } from "@/components/product/CapabilityRolePill";
import type { LLMCapabilityItem } from "@/lib/api";

export interface CapabilitiesTableLabels {
  capabilityModel: string;
  capabilityRole: string;
  capabilityDesired: string;
  capabilityEffective: string;
  capabilityProbeStatus: string;
  capabilityReason: string;
  capabilityProbedAt: string;
  capabilityDetails: string;
  hideDetails: string;
  showDetails: string;
  detailConfiguredMode: string;
  detailApiBase: string;
  detailError: string;
  detailBindTools: string;
  detailVision: string;
  detailReasoning: string;
  detailMismatch: string;
  detailMetadata: string;
  na: string;
}

interface CapabilitiesTableProps {
  items: LLMCapabilityItem[];
  expandedRows: Record<string, boolean>;
  onToggleRow: (key: string) => void;
  labels: CapabilitiesTableLabels;
  formatDateTime: (value: string) => string;
}

export function CapabilitiesTable({
  items,
  expandedRows,
  onToggleRow,
  labels,
  formatDateTime,
}: CapabilitiesTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full rounded-lg border border-border text-sm">
        <thead className="bg-surface-muted">
          <tr>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityModel}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityRole}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityDesired}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityEffective}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityProbeStatus}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityReason}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityProbedAt}</th>
            <th className="px-3 py-2 text-left font-semibold text-foreground">{labels.capabilityDetails}</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const rowKey = `${item.provider_id}:${item.model_name}`;
            const expanded = !!expandedRows[rowKey];
            return (
              <Fragment key={rowKey}>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 text-foreground">
                    <div className="font-medium">{item.model_name}</div>
                    <div className="text-xs text-muted-foreground">{item.provider_id}</div>
                  </td>
                  <td className="px-3 py-2">
                    <CapabilityRolePill role={item.role} />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{item.desired_mode}</td>
                  <td className="px-3 py-2">
                    <CapabilityModePill mode={item.effective_mode} />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{item.probe_status}</td>
                  <td className="px-3 py-2 text-muted-foreground">{item.effective_mode_reason}</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {item.probed_at ? formatDateTime(item.probed_at) : labels.na}
                  </td>
                  <td className="px-3 py-2">
                    <Button
                      type="button"
                      onClick={() => onToggleRow(rowKey)}
                      variant="secondary"
                      size="sm"
                      className="px-2 py-1 text-xs"
                    >
                      {expanded ? labels.hideDetails : labels.showDetails}
                    </Button>
                  </td>
                </tr>
                {expanded ? (
                  <tr className="border-t border-border bg-surface-muted">
                    <td colSpan={8} className="px-3 py-3 text-xs text-muted-foreground">
                      <CapabilityDetailsPanel
                        item={item}
                        labels={{
                          configuredMode: labels.detailConfiguredMode,
                          apiBase: labels.detailApiBase,
                          error: labels.detailError,
                          bindTools: labels.detailBindTools,
                          vision: labels.detailVision,
                          reasoning: labels.detailReasoning,
                          mismatch: labels.detailMismatch,
                          metadata: labels.detailMetadata,
                          na: labels.na,
                        }}
                      />
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
