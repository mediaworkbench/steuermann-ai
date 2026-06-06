import type { CapabilitiesTableLabels } from "@/components/product/CapabilitiesTable";

export function buildCapabilitiesTableLabels(
  t: (key: string) => string
): CapabilitiesTableLabels {
  return {
    capabilityModel: t("settingsPanel.capabilityModel"),
    capabilityRole: t("settingsPanel.capabilityRole"),
    capabilityDesired: t("settingsPanel.capabilityDesired"),
    capabilityEffective: t("settingsPanel.capabilityEffective"),
    capabilityProbeStatus: t("settingsPanel.capabilityProbeStatus"),
    capabilityReason: t("settingsPanel.capabilityReason"),
    capabilityProbedAt: t("settingsPanel.capabilityProbedAt"),
    capabilityDetails: t("settingsPanel.capabilityDetails"),
    hideDetails: t("settingsPanel.hideDetails"),
    showDetails: t("settingsPanel.showDetails"),
    detailConfiguredMode: t("settingsPanel.detailConfiguredMode"),
    detailApiBase: t("settingsPanel.detailApiBase"),
    detailError: t("settingsPanel.detailError"),
    detailBindTools: t("settingsPanel.detailBindTools"),
    detailVision: t("settingsPanel.detailVision"),
    detailReasoning: t("settingsPanel.detailReasoning"),
    detailMismatch: t("settingsPanel.detailMismatch"),
    detailMetadata: t("settingsPanel.detailMetadata"),
    na: t("metrics.na"),
  };
}
