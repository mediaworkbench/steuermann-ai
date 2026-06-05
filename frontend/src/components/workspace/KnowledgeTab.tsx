"use client";

import { useI18n } from "@/hooks/useI18n";
import { EvidenceTabPlaceholder } from "./EvidenceTabPlaceholder";

/** Read-only evidence tab for RAG / knowledge-base hits. Placeholder in R1.1. */
export function KnowledgeTab() {
  const { t } = useI18n();
  return (
    <EvidenceTabPlaceholder
      icon="menu_book"
      title={t("workspace.tabKnowledge")}
      hint={t("workspace.knowledgeEmpty")}
    />
  );
}
