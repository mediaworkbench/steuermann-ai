"use client";

import { useI18n } from "@/hooks/useI18n";
import { WorkspaceTabState } from "./WorkspaceTabState";

/** Read-only evidence tab for recalled memories. Placeholder until R1.3. */
export function MemoryTab() {
  const { t } = useI18n();
  return (
    <WorkspaceTabState
      icon="memory"
      title={t("workspace.tabMemory")}
      hint={t("workspace.memoryEmpty")}
    />
  );
}
