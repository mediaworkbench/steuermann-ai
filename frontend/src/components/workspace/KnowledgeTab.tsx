"use client";

import { useI18n } from "@/hooks/useI18n";
import { BookOpen, Globe } from "lucide-react";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { WorkspaceInlineBadge } from "./WorkspaceInlineBadge";
import { WorkspaceMutedCard } from "./WorkspaceMutedCard";
import { WorkspaceSectionLabel } from "./WorkspaceSectionLabel";
import type { AnswerEvidence } from "@/lib/answerEvidence";

/** Read-only evidence tab: RAG / knowledge-base hits and answer sources. */
export function KnowledgeTab({ evidence }: { evidence: AnswerEvidence }) {
  const { t } = useI18n();

  if (!evidence.knowledgeBaseUsed && evidence.sources.length === 0) {
    return (
      <WorkspaceTabState
        icon="menu_book"
        title={t("workspace.tabKnowledge")}
        hint={t("workspace.knowledgeEmpty")}
      />
    );
  }

  return (
    <div className="p-3 space-y-3">
      {evidence.knowledgeBaseUsed && (
        <WorkspaceMutedCard className="p-3">
          <WorkspaceSectionLabel className="px-0">{t("workspace.tabKnowledge")}</WorkspaceSectionLabel>
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <BookOpen size={14} className="text-muted-foreground shrink-0" />
            {evidence.ragDocCount > 0
              ? t("workspace.knowledgeRetrieved", { count: evidence.ragDocCount })
              : t("workspace.knowledgeNoResults")}
          </div>
        </WorkspaceMutedCard>
      )}

      {evidence.sources.length > 0 && (
        <section>
          <WorkspaceSectionLabel>{t("workspace.sourcesHeading")}</WorkspaceSectionLabel>
          <div className="flex flex-col gap-1.5">
            {evidence.sources.map((src, idx) => {
              const isWeb = src.type === "web";
              const body = (
                <>
                  {isWeb ? <Globe size={13} className="shrink-0" /> : <BookOpen size={13} className="shrink-0" />}
                  <span className="truncate">{src.label}</span>
                </>
              );
              return (
                <WorkspaceInlineBadge
                  key={`${src.label}-${idx}`}
                  tone={isWeb ? "primary" : "default"}
                  href={isWeb ? (src.url ?? undefined) : undefined}
                  className={!isWeb ? "bg-surface" : undefined}
                >
                  {body}
                </WorkspaceInlineBadge>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
