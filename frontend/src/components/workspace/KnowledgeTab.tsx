"use client";

import { useI18n } from "@/hooks/useI18n";
import { Icon } from "../Icon";
import { WorkspaceTabState } from "./WorkspaceTabState";
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
        <section className="rounded-lg border border-gray-200 bg-gray-50/60 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-evergreen/40 mb-1.5">
            {t("workspace.tabKnowledge")}
          </p>
          <div className="flex items-center gap-1.5 text-xs text-evergreen/70">
            <Icon name="menu_book" size={14} className="text-evergreen/40 shrink-0" />
            {evidence.ragDocCount > 0
              ? t("workspace.knowledgeRetrieved", { count: evidence.ragDocCount })
              : t("workspace.knowledgeNoResults")}
          </div>
        </section>
      )}

      {evidence.sources.length > 0 && (
        <section>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-evergreen/40 mb-1.5 px-0.5">
            {t("workspace.sourcesHeading")}
          </p>
          <div className="flex flex-col gap-1.5">
            {evidence.sources.map((src, idx) => {
              const isWeb = src.type === "web";
              const cls = isWeb
                ? "bg-pacific-blue/5 text-pacific-blue border-pacific-blue/20 hover:bg-pacific-blue/10"
                : "bg-amber-50 text-amber-800 border-amber-200";
              const body = (
                <>
                  <Icon name={isWeb ? "language" : "menu_book"} size={13} className="shrink-0" />
                  <span className="truncate">{src.label}</span>
                </>
              );
              return isWeb && src.url ? (
                <a
                  key={`${src.label}-${idx}`}
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border no-underline transition-colors ${cls}`}
                >
                  {body}
                </a>
              ) : (
                <span
                  key={`${src.label}-${idx}`}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border ${cls}`}
                >
                  {body}
                </span>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
