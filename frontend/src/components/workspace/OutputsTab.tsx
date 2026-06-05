"use client";

import { useI18n } from "@/hooks/useI18n";
import { WorkspaceTabState } from "./WorkspaceTabState";

/** Read-only evidence tab for tool / generation outputs. Placeholder until R1.3. */
export function OutputsTab() {
  const { t } = useI18n();
  return (
    <WorkspaceTabState
      icon="build"
      title={t("workspace.tabOutputs")}
      hint={t("workspace.outputsEmpty")}
    />
  );
}
