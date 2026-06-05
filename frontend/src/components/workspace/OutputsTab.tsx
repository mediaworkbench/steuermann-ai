"use client";

import { useI18n } from "@/hooks/useI18n";
import { EvidenceTabPlaceholder } from "./EvidenceTabPlaceholder";

/** Read-only evidence tab for tool / generation outputs. Placeholder in R1.1. */
export function OutputsTab() {
  const { t } = useI18n();
  return (
    <EvidenceTabPlaceholder
      icon="build"
      title={t("workspace.tabOutputs")}
      hint={t("workspace.outputsEmpty")}
    />
  );
}
