"use client";

import { useState, useCallback, useRef } from "react";
import { Icon } from "./Icon";
import type { MessageMetrics } from "@/lib/types";
import { useI18n } from "@/hooks/useI18n";

interface MetricsPanelProps {
  metrics?: MessageMetrics;
  messageContent: string;
  onRegenerate?: () => void;
}

const FINISH_REASON_COLORS: Record<string, string> = {
  stop: "text-pacific-blue",
  tool_use: "text-atomic-tangerine",
  max_tokens: "text-burnt-tangerine",
};

export function MetricsPanel({
  metrics,
  messageContent,
  onRegenerate,
}: MetricsPanelProps) {
  const { t, formatNumber } = useI18n();
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const totalTokens =
    (metrics?.input_tokens ?? 0) + (metrics?.output_tokens ?? 0);
  const responseTimeSec = metrics?.response_time_ms
    ? (metrics.response_time_ms / 1000).toFixed(1)
    : null;
  const toolCount = metrics?.tools_executed?.length ?? 0;

  const handleCopy = useCallback(() => {
    const doCopy = () => {
      setCopied(true);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 1500);
    };

    if (navigator.clipboard) {
      navigator.clipboard.writeText(messageContent).then(doCopy).catch(() => {
        fallbackCopy(messageContent);
        doCopy();
      });
    } else {
      fallbackCopy(messageContent);
      doCopy();
    }
  }, [messageContent]);

  return (
    <div className="mt-2 ml-1 w-full">
      {/* Metrics toggle + action icons in the same row */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-xs text-evergreen/40 hover:text-evergreen/70 transition-colors"
          aria-expanded={expanded}
        >
          <Icon
            name="expand_more"
            size={14}
            className={`metrics-chevron ${expanded ? "open" : ""}`}
          />
          <span className="flex items-center gap-1.5 font-mono">
            {responseTimeSec && (
              <>
                <span className="flex items-center gap-0.5">
                  <Icon name="timer" size={13} className="text-pacific-blue/70" />
                  {responseTimeSec}s
                </span>
                <span className="text-evergreen/20" aria-hidden="true">
                  ·
                </span>
              </>
            )}
            <span className="flex items-center gap-0.5">
              <Icon name="toll" size={13} className="text-pacific-blue/70" />
              {formatNumber(totalTokens)} {t("charts.tokens")}
            </span>
            {toolCount > 0 && (
              <>
                <span className="text-evergreen/20" aria-hidden="true">
                  ·
                </span>
                <span className="flex items-center gap-0.5">
                  <Icon name="build" size={13} className="text-pacific-blue/70" />
                  {toolCount} {t("chat.toolsInvoked")}
                </span>
              </>
            )}
          </span>
        </button>

        {/* Action icons — same row, pushed right */}
        <div className="flex items-center gap-0.5">
          <button
            onClick={handleCopy}
            className={`p-1.5 rounded transition-colors ${
              copied
                ? "text-pacific-blue copy-success"
                : "text-evergreen/30 hover:text-pacific-blue hover:bg-pacific-blue/10"
            }`}
            aria-label={t("chat.copyMessage")}
          >
            <Icon name={copied ? "check" : "content_copy"} size={16} />
          </button>
          <button
            onClick={onRegenerate}
            className="p-1.5 rounded text-evergreen/30 hover:text-atomic-tangerine hover:bg-atomic-tangerine/10 transition-colors"
            aria-label={t("chat.regenerateResponse")}
          >
            <Icon name="refresh" size={16} />
          </button>
        </div>
      </div>

      {/* Expanded metrics body */}
      <div
        className={`metrics-body ${expanded ? "open" : ""}`}
        role="region"
        aria-label={t("chat.responseMetrics")}
      >
        <div>
          <div className="mt-2 rounded-lg border border-light-cyan/60 bg-light-cyan/10 p-3 text-xs font-mono space-y-3">
            {/* Timing + tokens + finish reason */}
            <div className="flex flex-wrap gap-x-6 gap-y-2">
              {responseTimeSec && (
                <MetricCell
                  label={t("chat.responseTime")}
                  value={`${responseTimeSec}s`}
                />
              )}
              {metrics?.input_tokens != null && (
                <MetricCell
                  label={t("chat.inputTokens")}
                  value={formatNumber(metrics.input_tokens)}
                />
              )}
              {metrics?.output_tokens != null && (
                <MetricCell
                  label={t("chat.outputTokens")}
                  value={formatNumber(metrics.output_tokens)}
                />
              )}
              {metrics?.finish_reason && (
                <div className="flex flex-col gap-0.5">
                  <span className="text-evergreen/40 uppercase tracking-wider text-[10px]">
                    {t("chat.finishReason")}
                  </span>
                  <span
                    className={`font-semibold ${
                      FINISH_REASON_COLORS[metrics.finish_reason] ??
                      "text-evergreen"
                    }`}
                  >
                    {metrics.finish_reason}
                  </span>
                </div>
              )}
            </div>

            {/* Model + temperature */}
            {(metrics?.model || metrics?.temperature != null) && (
              <div className="flex flex-wrap gap-x-6 gap-y-2 pt-2 border-t border-light-cyan/40">
                {metrics.model && (
                  <MetricCell label={t("chat.model")} value={metrics.model} />
                )}
                {metrics.temperature != null && (
                  <MetricCell
                    label={t("chat.temperature")}
                    value={String(metrics.temperature)}
                  />
                )}
              </div>
            )}

            {/* Tools */}
            {metrics?.tools_executed && metrics.tools_executed.length > 0 && (
              <div className="pt-2 border-t border-light-cyan/40">
                <span className="text-evergreen/40 uppercase tracking-wider text-[10px] block mb-1.5">
                  {t("chat.toolsInvoked")}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {metrics.tools_executed.map((tool, idx) => {
                    const isRag = tool.name === "knowledge_base";
                    const isError = tool.status === "error";
                    const badgeClass = isRag
                      ? "bg-amber-100 text-amber-800 border border-amber-200"
                      : isError
                        ? "bg-burnt-tangerine/10 text-burnt-tangerine border border-burnt-tangerine/20"
                        : "bg-pacific-blue/15 text-pacific-blue border border-pacific-blue/20";
                    const iconName = isRag
                      ? "menu_book"
                      : tool.status === "success"
                        ? "check_circle"
                        : "error";
                    return (
                      <span
                        key={`${tool.name}-${idx}`}
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full ${badgeClass}`}
                      >
                        <Icon name={iconName} size={11} />
                        {isRag ? "RAG" : tool.name}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-evergreen/40 uppercase tracking-wider text-[10px]">
        {label}
      </span>
      <span className="text-evergreen font-semibold">{value}</span>
    </div>
  );
}

function fallbackCopy(text: string) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
  } catch {
    // silently fail
  }
  document.body.removeChild(ta);
}
