"use client";

import { useI18n } from "@/hooks/useI18n";
import { WorkspaceTabState } from "./WorkspaceTabState";

/** Read-only evidence tab for RAG / knowledge-base hits. Placeholder until R1.3. */
export function KnowledgeTab() {
  const { t } = useI18n();
  return (
    <WorkspaceTabState
      icon="menu_book"
      title={t("workspace.tabKnowledge")}
      hint={t("workspace.knowledgeEmpty")}
    />
  );
}
