"use client";

import { useI18n } from "@/hooks/useI18n";
import { AlertCircle, CheckCircle, Map } from "lucide-react";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { WorkspaceInlineBadge } from "./WorkspaceInlineBadge";
import { WorkspaceSectionLabel } from "./WorkspaceSectionLabel";
import type { AnswerEvidence } from "@/lib/answerEvidence";

/** Read-only evidence tab: tool runs and generated outputs (e.g. maps). */
export function OutputsTab({ evidence }: { evidence: AnswerEvidence }) {
  const { t } = useI18n();
  const hasOutputs = evidence.tools.length > 0 || Boolean(evidence.mapData);

  if (!hasOutputs) {
    return (
      <WorkspaceTabState
        icon="build"
        title={t("workspace.tabOutputs")}
        hint={t("workspace.outputsEmpty")}
      />
    );
  }

  return (
    <div className="p-3 space-y-3">
      {evidence.tools.length > 0 && (
        <section>
          <WorkspaceSectionLabel>{t("chat.toolsInvoked")}</WorkspaceSectionLabel>
          <div className="flex flex-wrap gap-1.5">
            {evidence.tools.map((tool, idx) => {
              const isError = tool.status === "error";
              return (
                <WorkspaceInlineBadge
                  key={`${tool.name}-${idx}`}
                  tone={isError ? "destructive" : "primary"}
                  className="gap-1 rounded-full px-2 py-0.5"
                >
                  {isError ? <AlertCircle size={11} /> : <CheckCircle size={11} />}
                  {tool.name}
                </WorkspaceInlineBadge>
              );
            })}
          </div>
        </section>
      )}

      {evidence.mapData && (
        <section>
          <WorkspaceSectionLabel>{t("workspace.mapGenerated")}</WorkspaceSectionLabel>
          <WorkspaceInlineBadge tone="default">
            <Map size={13} className="shrink-0" />
            <span className="truncate">
              {evidence.mapData.label || evidence.mapData.summary || t("workspace.mapGenerated")}
            </span>
          </WorkspaceInlineBadge>
        </section>
      )}
    </div>
  );
}
