"use client";

import { useI18n } from "@/hooks/useI18n";
import { AlertCircle, ArrowRight, CheckCircle } from "lucide-react";
import { iconMap } from "@/lib/iconMap";
import { WorkspaceTabState } from "./WorkspaceTabState";
import { WorkspaceInlineBadge } from "./WorkspaceInlineBadge";
import { WorkspaceSectionLabel } from "./WorkspaceSectionLabel";
import type { NodeTraceEntry } from "@/lib/types";
import { POST_RESPONSE_NODES, isPostResponseNode } from "@/lib/nodeTrace";

// The three mutually-exclusive tool-calling strategies; exactly one runs per
// turn, so the others are collapsed into a single "Call tools" slot.
const CALL_TOOLS_SET = new Set(["call_tools_native", "call_tools_structured", "call_tools_react"]);

// Answer-path nodes in builder order (see CLAUDE.md / graph_builder.py).
const PRE_RESPONSE_NODES = [
  "load_tools",
  "prefilter_tools",
  "call_tools_native",
  "call_tools_structured",
  "call_tools_react",
  "after_tool_call",
  "memory_query_cache",
  "load_memory",
  "retrieve_knowledge",
  "memory_cache_store",
  "respond",
];

/** Humanize a graph node id (e.g. "retrieve_knowledge" → "Retrieve knowledge"). */
function humanizeNode(id: string): string {
  const spaced = id.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * Inspector: a semantic view of how the latest answer was produced — the ordered
 * nodes that fired (active path) with per-node status and timing, the answer-path
 * nodes that did not run this turn, and the post-response nodes that run after the
 * answer is sent. Persisted in message metadata, so it survives a conversation reload.
 */
export function InspectorTab({
  nodeTrace,
  isStreaming,
  isLatestAnswer = false,
  onOpenOutputs,
}: {
  nodeTrace: NodeTraceEntry[];
  isStreaming?: boolean;
  /** Latest (non-historical) answer: its post-response nodes are still running in
   *  the background, so pending ones are shown as in-progress rather than inert. */
  isLatestAnswer?: boolean;
  /** When provided, tool-calling node rows become deep-links into the Outputs tab. */
  onOpenOutputs?: () => void;
}) {
  const { t } = useI18n();

  if (nodeTrace.length === 0) {
    return (
      <WorkspaceTabState
        icon="account_tree"
        title={isStreaming ? t("workspace.inspectorRunning") : t("workspace.tabInspector")}
        hint={isStreaming ? t("workspace.inspectorWaiting") : t("workspace.inspectorEmpty")}
        tone={isStreaming ? "loading" : "idle"}
      />
    );
  }

  const firedIds = new Set(nodeTrace.map((n) => n.node));
  const toolCallingFired = nodeTrace.some((n) => CALL_TOOLS_SET.has(n.node));
  const totalMs = nodeTrace.reduce((sum, n) => sum + (n.durationMs ?? 0), 0);
  const maxMs = Math.max(1, ...nodeTrace.map((n) => n.durationMs ?? 0));

  // Post-response nodes run after [DONE]; their trace is captured by the backend
  // drain and persisted, so it arrives after the answer (on reload / auto-refresh).
  // Split it out so it renders as its own ordered list with full status + timing.
  const preTrace = nodeTrace.filter((n) => !isPostResponseNode(n.node));
  const postTrace = nodeTrace.filter((n) => isPostResponseNode(n.node));

  // Answer-path nodes that did not fire, collapsing the mutually-exclusive
  // tool-calling strategies into one "Call tools" slot (shown only if none ran).
  const skipped: string[] = [];
  let toolSlotAdded = false;
  for (const id of PRE_RESPONSE_NODES) {
    if (firedIds.has(id)) continue;
    if (CALL_TOOLS_SET.has(id)) {
      if (!toolCallingFired && !toolSlotAdded) {
        skipped.push("call_tools");
        toolSlotAdded = true;
      }
      continue;
    }
    skipped.push(id);
  }
  // Post-response nodes whose trace hasn't landed yet — shown as pending badges.
  const postResponsePending = POST_RESPONSE_NODES.filter((id) => !firedIds.has(id));

  return (
    <div className="p-3 space-y-3">
      {/* Summary */}
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-muted-foreground">
          {t("workspace.inspectorNodes", { count: nodeTrace.length })}
        </span>
        <span className="font-mono text-muted-foreground">{Math.round(totalMs)} ms</span>
      </div>

      {/* Active path — ordered nodes that fired, with status + timing */}
      <ol className="space-y-1">
        {preTrace.map((n, idx) => (
          <NodeRow key={`${n.node}-${n.sequence}-${idx}`} n={n} maxMs={maxMs} onOpenOutputs={onOpenOutputs} />
        ))}
      </ol>

      {/* Live indicator while the answer is still streaming */}
      {isStreaming && (
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground px-0.5">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          {t("workspace.inspectorRunning")}
        </div>
      )}

      {/* Answer-path nodes that did not run this turn */}
      {skipped.length > 0 && <NodeGroup heading={t("workspace.inspectorNotRun")} ids={skipped} />}

      {/* Nodes that run after the response is sent. Captured ones render with full
          status + timing once the backend drain persists them; the rest stay as
          pending badges until that trace lands. */}
      {(postTrace.length > 0 || postResponsePending.length > 0) && (
        <div className="pt-2 border-t border-border/60">
          <WorkspaceSectionLabel>{t("workspace.inspectorPostResponse")}</WorkspaceSectionLabel>
          {postTrace.length > 0 && (
            <ol className="space-y-1">
              {postTrace.map((n, idx) => (
                <NodeRow key={`${n.node}-${n.sequence}-${idx}`} n={n} maxMs={maxMs} />
              ))}
            </ol>
          )}
          {postResponsePending.length > 0 && (
            <>
              {/* Latest answer: these nodes are still executing in the ~40s
                  background drain, so flag them as in-progress (not skipped). */}
              {isLatestAnswer && postTrace.length === 0 && (
                <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground px-0.5 mt-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                  {t("workspace.inspectorPostResponseRunning")}
                </div>
              )}
              <div className="flex flex-wrap gap-1 mt-1">
                {postResponsePending.map((id) => {
                  const IconCmp = iconMap["schedule"];
                  return (
                    <WorkspaceInlineBadge
                      key={id}
                      size="compact"
                      className={`text-muted-foreground${isLatestAnswer && postTrace.length === 0 ? " animate-pulse" : ""}`}
                    >
                      {IconCmp && <IconCmp size={10} />}
                      {humanizeNode(id)}
                    </WorkspaceInlineBadge>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** One ordered node-trace row: sequence, status icon, label, duration, timing bar. */
function NodeRow({
  n,
  maxMs,
  onOpenOutputs,
}: {
  n: NodeTraceEntry;
  maxMs: number;
  onOpenOutputs?: () => void;
}) {
  const { t } = useI18n();
  const isError = n.status === "error";
  const pct = n.durationMs != null ? Math.round((n.durationMs / maxMs) * 100) : 0;
  const isToolNode = CALL_TOOLS_SET.has(n.node);
  const linkToOutputs = isToolNode && Boolean(onOpenOutputs);
  const rowContent = (
    <>
      <span className="font-mono text-[10px] text-muted-foreground w-5 shrink-0">{n.sequence}</span>
      {isError
        ? <AlertCircle size={13} className="shrink-0 text-destructive" />
        : <CheckCircle size={13} className="shrink-0 text-primary" />
      }
      <span className="text-xs text-foreground truncate flex-1">{humanizeNode(n.node)}</span>
      <span className="sr-only">
        {isError ? t("workspace.inspectorStatusError") : t("workspace.inspectorStatusSuccess")}
      </span>
      {n.durationMs != null && (
        <span className="font-mono text-[10px] text-muted-foreground shrink-0">{n.durationMs} ms</span>
      )}
      {linkToOutputs && (
        <>
          <ArrowRight size={12} className="shrink-0 text-muted-foreground group-hover/row:text-primary" />
          <span className="sr-only">{t("workspace.viewToolResults")}</span>
        </>
      )}
    </>
  );
  return (
    <li className="rounded-lg border border-border bg-surface-muted px-2.5 py-1.5">
      {linkToOutputs ? (
        <button
          type="button"
          onClick={onOpenOutputs}
          title={t("workspace.viewToolResults")}
          className="group/row flex w-full items-center gap-2 text-left"
        >
          {rowContent}
        </button>
      ) : (
        <div className="flex items-center gap-2">{rowContent}</div>
      )}
      <div className="mt-1 ml-7 h-1 rounded-full bg-border overflow-hidden">
        <div
          className={`h-full rounded-full ${isError ? "bg-destructive/50" : "bg-primary/40"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </li>
  );
}

function NodeGroup({ heading, ids, icon }: { heading: string; ids: string[]; icon?: string }) {
  const IconCmp = icon ? iconMap[icon] : null;
  return (
    <div className="pt-2 border-t border-border/60">
      <WorkspaceSectionLabel>{heading}</WorkspaceSectionLabel>
      <div className="flex flex-wrap gap-1">
        {ids.map((id) => (
          <WorkspaceInlineBadge
            key={id}
            size="compact"
            className="text-muted-foreground"
          >
            {IconCmp && <IconCmp size={10} />}
            {humanizeNode(id)}
          </WorkspaceInlineBadge>
        ))}
      </div>
    </div>
  );
}
