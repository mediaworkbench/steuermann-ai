"use client";

import { useState } from "react";
import { useI18n } from "@/hooks/useI18n";
import { AlertCircle, CheckCircle, ChevronDown, ChevronRight, Map } from "lucide-react";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { WorkspaceInlineBadge } from "./WorkspaceInlineBadge";
import { WorkspaceSectionLabel } from "./WorkspaceSectionLabel";
import type { AnswerEvidence } from "@/lib/answerEvidence";
import type { ToolResultDetail } from "@/lib/types";

/** Render an arg value (already bounded server-side) as a compact string. */
function formatArgValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** Expandable card for one tool invocation: name + status, with args/output on expand. */
function ToolResultRow({ detail }: { detail: ToolResultDetail }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const isError = detail.status === "error";
  const argEntries = detail.args ? Object.entries(detail.args) : [];
  const body = isError ? detail.error : detail.output;
  const hasDetail = argEntries.length > 0 || Boolean(body);

  return (
    <li className="rounded-lg border border-border bg-surface-muted overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        disabled={!hasDetail}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left enabled:hover:bg-surface-muted/60 disabled:cursor-default"
      >
        {isError
          ? <AlertCircle size={13} className="shrink-0 text-destructive" />
          : <CheckCircle size={13} className="shrink-0 text-primary" />}
        <span className="font-mono text-xs text-foreground truncate flex-1">{detail.name}</span>
        <span className="sr-only">
          {isError ? t("workspace.inspectorStatusError") : t("workspace.inspectorStatusSuccess")}
        </span>
        {hasDetail && (open
          ? <ChevronDown size={13} className="shrink-0 text-muted-foreground" />
          : <ChevronRight size={13} className="shrink-0 text-muted-foreground" />)}
      </button>

      {!open && detail.summary && (
        <p className="px-2.5 pb-1.5 ml-5.5 text-[11px] text-muted-foreground line-clamp-1">
          {detail.summary}
        </p>
      )}

      {open && hasDetail && (
        <div className="px-2.5 pb-2 pt-2 space-y-2 border-t border-border/60">
          {argEntries.length > 0 && (
            <div>
              <WorkspaceSectionLabel>{t("workspace.toolArgs")}</WorkspaceSectionLabel>
              <dl className="space-y-0.5">
                {argEntries.map(([key, value]) => (
                  <div key={key} className="flex gap-1.5 text-[11px]">
                    <dt className="font-mono text-muted-foreground shrink-0">{key}:</dt>
                    <dd className="font-mono text-foreground break-all">{formatArgValue(value)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
          {body && (
            <div>
              <WorkspaceSectionLabel>
                {isError ? t("workspace.toolError") : t("workspace.toolOutput")}
              </WorkspaceSectionLabel>
              <pre className={`whitespace-pre-wrap wrap-break-word font-mono text-[11px] leading-snug ${isError ? "text-destructive" : "text-foreground"}`}>
                {body}
              </pre>
            </div>
          )}
        </div>
      )}
    </li>
  );
}

/** Read-only evidence tab: tool invocations (args + result preview) and generated outputs (e.g. maps). */
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
          {evidence.toolResults.length > 0 ? (
            <ul className="space-y-1.5">
              {evidence.toolResults.map((detail, idx) => (
                <ToolResultRow key={`${detail.name}-${idx}`} detail={detail} />
              ))}
            </ul>
          ) : (
            // Older persisted answers carry tool names but no detail payload —
            // fall back to compact name badges so they still render.
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
          )}
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
