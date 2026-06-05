"use client";

import { useI18n } from "@/hooks/useI18n";
import { Icon } from "../Icon";
import { WorkspaceTabState } from "./WorkspaceTabState";
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
          <p className="text-[10px] font-semibold uppercase tracking-wider text-evergreen/40 mb-1.5 px-0.5">
            {t("chat.toolsInvoked")}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {evidence.tools.map((tool, idx) => {
              const isError = tool.status === "error";
              const cls = isError
                ? "bg-burnt-tangerine/10 text-burnt-tangerine border-burnt-tangerine/20"
                : "bg-pacific-blue/10 text-pacific-blue border-pacific-blue/20";
              return (
                <span
                  key={`${tool.name}-${idx}`}
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${cls}`}
                >
                  <Icon name={isError ? "error" : "check_circle"} size={11} />
                  {tool.name}
                </span>
              );
            })}
          </div>
        </section>
      )}

      {evidence.mapData && (
        <section>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-evergreen/40 mb-1.5 px-0.5">
            {t("workspace.mapGenerated")}
          </p>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-evergreen/5 text-evergreen/70 border border-gray-200">
            <Icon name="map" size={13} className="shrink-0" />
            <span className="truncate">
              {evidence.mapData.label || evidence.mapData.summary || t("workspace.mapGenerated")}
            </span>
          </div>
        </section>
      )}
    </div>
  );
}
