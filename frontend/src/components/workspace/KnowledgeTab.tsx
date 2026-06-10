"use client";

import { useI18n } from "@/hooks/useI18n";
import { BookOpen, FileText, FolderOpen, Globe } from "lucide-react";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { WorkspaceInlineBadge } from "./WorkspaceInlineBadge";
import { WorkspaceMutedCard } from "./WorkspaceMutedCard";
import { WorkspaceSectionLabel } from "./WorkspaceSectionLabel";
import type { AnswerEvidence } from "@/lib/answerEvidence";

/**
 * Derive the display citation number for a source at position `i` in
 * `evidence.sources`. Mirrors the logic in `lib/markdown.ts` `linkFootnotes`:
 * prefer the backend-assigned `index` field; fall back to 1-based position.
 * The fallback is applied per-source so partial-index data still renders sensibly.
 */
function sourceCitationNumber(
  src: AnswerEvidence["sources"][number],
  idx: number,
): number {
  return src.index ?? idx + 1;
}

/** Read-only evidence tab: RAG / knowledge-base hits, answer sources (with [N]
 *  citation indices), workspace documents, and attachments used as context. */
export function KnowledgeTab({ evidence }: { evidence: AnswerEvidence }) {
  const { t } = useI18n();

  const hasContent =
    evidence.knowledgeBaseUsed ||
    evidence.sources.length > 0 ||
    evidence.documentCount > 0 ||
    evidence.attachments.length > 0;

  if (!hasContent) {
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
      {/* RAG / knowledge-base summary */}
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

      {/* Sources — each prefixed with its [N] citation number so readers can
          cross-reference with the [N] superscripts in the answer text. */}
      {evidence.sources.length > 0 && (
        <section>
          <WorkspaceSectionLabel>{t("workspace.sourcesHeading")}</WorkspaceSectionLabel>
          <div className="flex flex-col gap-1.5">
            {evidence.sources.map((src, idx) => {
              const isWeb = src.type === "web";
              const n = sourceCitationNumber(src, idx);
              return (
                <WorkspaceInlineBadge
                  key={`${src.label}-${idx}`}
                  tone={isWeb ? "primary" : "default"}
                  href={isWeb ? (src.url ?? undefined) : undefined}
                  className={!isWeb ? "bg-surface" : undefined}
                >
                  {/* Citation index badge — matches the superscript in the answer text */}
                  <span
                    className="shrink-0 rounded bg-primary/10 px-1 font-mono text-[10px] font-semibold leading-4 text-primary"
                    aria-label={t("workspace.sourceCitationLabel", { n })}
                  >
                    {n}
                  </span>
                  {isWeb
                    ? <Globe size={13} className="shrink-0" />
                    : <BookOpen size={13} className="shrink-0" />}
                  <span className="truncate">{src.label}</span>
                </WorkspaceInlineBadge>
              );
            })}
          </div>
        </section>
      )}

      {/* Workspace documents used as context for this answer */}
      {evidence.documentCount > 0 && (
        <section>
          <WorkspaceSectionLabel>{t("workspace.documentsInContext")}</WorkspaceSectionLabel>
          <div className="flex flex-col gap-1.5">
            {evidence.documents.map((doc) => (
              <WorkspaceInlineBadge key={doc.id} tone="default">
                <FolderOpen size={13} className="shrink-0" />
                <span className="truncate">{doc.filename}</span>
                <span className="shrink-0 text-muted-foreground">v{doc.version}</span>
              </WorkspaceInlineBadge>
            ))}
          </div>
        </section>
      )}

      {/* File attachments used as context for this answer */}
      {evidence.attachments.length > 0 && (
        <section>
          <WorkspaceSectionLabel>{t("workspace.attachmentsInContext")}</WorkspaceSectionLabel>
          <div className="flex flex-col gap-1.5">
            {evidence.attachments.map((att) => (
              <WorkspaceInlineBadge key={att.id} tone="default">
                <FileText size={13} className="shrink-0" />
                <span className="truncate">{att.original_name}</span>
              </WorkspaceInlineBadge>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
