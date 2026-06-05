"use client";

import { useI18n } from "@/hooks/useI18n";
import { EvidenceTabPlaceholder } from "./EvidenceTabPlaceholder";

/** Read-only evidence tab for recalled memories. Placeholder in R1.1. */
export function MemoryTab() {
  const { t } = useI18n();
  return (
    <EvidenceTabPlaceholder
      icon="memory"
      title={t("workspace.tabMemory")}
      hint={t("workspace.memoryEmpty")}
    />
  );
}
