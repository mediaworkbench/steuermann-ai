"use client";

import {
  Brain,
  Calculator,
  Clock,
  Compass,
  ExternalLink,
  FileEdit,
  Search,
  Settings,
} from "lucide-react";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { ReasoningBox } from "@/components/ReasoningBox";
import { ChatMessageShell } from "@/components/product/ChatMessageShell";
import { useI18n } from "@/hooks/useI18n";

function nodeStatusIcon(
  nodeStatus: string | null,
  toolCallStatus: { name: string; status: "start" | "end"; label: string } | null,
) {
  if (nodeStatus?.includes("knowledge")) return <Search size={13} className="shrink-0" />;
  if (nodeStatus?.includes("memor")) return <Brain size={13} className="shrink-0" />;
  if (toolCallStatus?.name?.includes("search") || toolCallStatus?.name?.includes("web"))
    return <Compass size={13} className="shrink-0" />;
  if (toolCallStatus?.name?.includes("calc")) return <Calculator size={13} className="shrink-0" />;
  if (toolCallStatus?.name?.includes("date") || toolCallStatus?.name?.includes("time"))
    return <Clock size={13} className="shrink-0" />;
  if (toolCallStatus?.name?.includes("file") || toolCallStatus?.name?.includes("workspace"))
    return <FileEdit size={13} className="shrink-0" />;
  if (toolCallStatus?.name?.includes("webpage") || toolCallStatus?.name?.includes("extract"))
    return <ExternalLink size={13} className="shrink-0" />;
  return <Settings size={13} className="shrink-0" />;
}

interface StreamingMessageProps {
  isStreaming: boolean;
  loading: boolean;
  streamingContent: string;
  thinkingContent: string;
  isThinking: boolean;
  nodeStatus: string | null;
  toolCallStatus: { name: string; status: "start" | "end"; label: string } | null;
  /** Filename of the document being written back, or null if not a writeback turn. */
  compactWritebackDoc: string | null;
  /** Pre-split SUMMARY section for the compact writeback view. */
  writebackSummary: string;
}

export function StreamingMessage({
  isStreaming,
  loading,
  streamingContent,
  thinkingContent,
  isThinking,
  nodeStatus,
  toolCallStatus,
  compactWritebackDoc,
  writebackSummary,
}: StreamingMessageProps) {
  const { t } = useI18n();

  if (!isStreaming && !loading) return null;

  return (
    <ChatMessageShell messageRole="assistant">
        <div className="flex items-center gap-2 ml-1">
          <span className="text-sm font-bold text-foreground">
            {t("chat.aiAgent")}
          </span>
        </div>

        {/* Node / tool status indicator */}
        {(nodeStatus || toolCallStatus?.status === "start") && (
          <div className="mb-1 ml-1 flex items-center gap-1.5 text-xs text-muted-foreground animate-pulse">
            {nodeStatusIcon(nodeStatus, toolCallStatus)}
            <span>{nodeStatus ?? toolCallStatus?.label}</span>
          </div>
        )}

        {(thinkingContent || isThinking) && (
          <ReasoningBox content={thinkingContent} isStreaming={isThinking} />
        )}

        {isStreaming && streamingContent && compactWritebackDoc ? (
          /* Compact writeback view: SUMMARY + "Updating document…" — the
             full DOCUMENT body is saved, not streamed into the chat. */
          <div
            className="px-1 text-base leading-relaxed text-foreground"
            aria-live="polite"
            aria-busy="true"
          >
            {writebackSummary && (
              <MarkdownMessage content={writebackSummary} />
            )}
            <div className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground animate-pulse">
              <FileEdit size={13} className="shrink-0" />
              <span>{t("chat.updatingDocument", { name: compactWritebackDoc })}</span>
            </div>
          </div>
        ) : isStreaming && streamingContent ? (
          /* Live streaming text with cursor */
          <div
            className="px-1 text-base leading-relaxed text-foreground"
            aria-live="polite"
            aria-busy="true"
          >
            <MarkdownMessage content={streamingContent} />
            <span
              className="ml-0.5 inline-block h-[1.15em] w-[0.55em] animate-cursor-blink rounded-[1px] bg-primary/70 align-middle"
              aria-hidden="true"
            />
          </div>
        ) : (
          /* Fallback three-dot loader (before first token arrives) */
          <div
            className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-border bg-surface-muted px-4 py-3"
            role="status"
            aria-label={t("chat.aiThinking")}
          >
            <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
            <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
            <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
          </div>
        )}
    </ChatMessageShell>
  );
}
