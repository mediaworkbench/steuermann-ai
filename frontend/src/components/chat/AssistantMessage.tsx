"use client";

import dynamic from "next/dynamic";
const MapWidget = dynamic(() => import("@/components/MapWidget").then((m) => m.MapWidget), { ssr: false });
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { MetricsPanel } from "@/components/MetricsPanel";
import { ReasoningBox } from "@/components/ReasoningBox";
import { EvidenceChips } from "@/components/workspace/EvidenceChips";
import { ChatMessageShell } from "@/components/product/ChatMessageShell";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/hooks/useI18n";
import type { Message } from "@/lib/types";
import type { WorkspaceTabId } from "@/components/workspace/types";

interface AssistantMessageProps {
  message: Message;
  index: number;
  /** True when the workspace panel is pinned to this (earlier) answer. */
  isFocused?: boolean;
  onSelectEvidence?: (tab: WorkspaceTabId, index: number) => void;
  onRegenerate: () => void;
  onFeedback: (index: number, value: "up" | "down") => void;
  loading: boolean;
  showMetrics?: boolean;
}

export function AssistantMessage({
  message,
  index,
  isFocused = false,
  onSelectEvidence,
  onRegenerate,
  onFeedback,
  loading,
  showMetrics = true,
}: AssistantMessageProps) {
  const { t } = useI18n();
  return (
    <ChatMessageShell
      messageRole="assistant"
      bodyClassName={isFocused ? "rounded-lg p-2 ring-1 ring-primary/30 ring-offset-2 ring-offset-background transition-all" : undefined}
    >
        {/* Name + timestamp */}
        <div className="flex items-center gap-2 ml-1">
          <span className="text-sm font-bold text-foreground">{t("chat.aiAgent")}</span>
          {message.timestamp && (
            <span className="msg-timestamp font-mono text-xs text-muted-foreground">
              {message.timestamp}
            </span>
          )}
        </div>

        {/* Reasoning chain (collapsed by default for completed messages) */}
        {message.thinking && (
          <ReasoningBox content={message.thinking} isStreaming={false} />
        )}

        {/* Message text */}
        <div className="px-1 text-base leading-relaxed text-foreground">
          <MarkdownMessage content={message.content} sources={message.metrics?.sources} />
        </div>

        {/* Map widget — rendered when map_tool was used */}
        {message.metrics?.map_data && (
          <div className="mt-2 px-1 w-full">
            <MapWidget data={message.metrics.map_data} />
          </div>
        )}

        {/* Single inline provenance summary (sources · memory · tools · attachments ·
            docs), interactive on every answer: clicking a chip pins the workspace
            panel to THIS answer and opens the matching tab. Citations stay inline as
            [N] superscripts; the full drill-down (source URLs, tool args/results,
            trace) lives in the tabs. */}
        <EvidenceChips
          metrics={message.metrics}
          onSelect={onSelectEvidence ? (tab) => onSelectEvidence(tab, index) : undefined}
        />

        {/* Metrics panel + feedback row */}
        <div className="w-full flex flex-col gap-1">
          <MetricsPanel
            metrics={message.metrics}
            messageContent={message.content}
            onRegenerate={loading ? undefined : onRegenerate}
            showMetrics={showMetrics}
          />

          {/* Feedback buttons */}
          <div className="flex items-center gap-1 ml-1">
            <Button
              type="button"
              onClick={() => onFeedback(index, "up")}
              disabled={loading}
              variant="ghost"
              size="sm"
              className={`p-1 rounded transition-colors cursor-pointer
                ${
                  message.feedback === "up"
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                } disabled:opacity-40`}
              aria-label="Thumbs up"
              title={t("chat.feedbackSaved")}
            >
              <ThumbsUp size={15} />
            </Button>
            <Button
              type="button"
              onClick={() => onFeedback(index, "down")}
              disabled={loading}
              variant="ghost"
              size="sm"
              className={`p-1 rounded transition-colors cursor-pointer
                ${
                  message.feedback === "down"
                    ? "bg-destructive/10 text-destructive"
                    : "text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                } disabled:opacity-40`}
              aria-label="Thumbs down"
              title="Poor response"
            >
              <ThumbsDown size={15} />
            </Button>
          </div>
        </div>
    </ChatMessageShell>
  );
}
