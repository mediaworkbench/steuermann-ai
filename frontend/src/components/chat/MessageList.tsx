"use client";

import { Bot } from "lucide-react";
import { AssistantMessage } from "./AssistantMessage";
import { UserMessage } from "./UserMessage";
import { QueuedMessageBubble } from "./QueuedMessageBubble";
import { StreamingMessage } from "./StreamingMessage";
import { ScrollToBottomButton } from "@/components/ScrollToBottomButton";
import { useI18n } from "@/hooks/useI18n";
import type { Message, NodeTraceEntry } from "@/lib/types";
import type { WorkspaceTabId } from "@/components/workspace/types";
import type { QueuedMessage } from "@/context/ChatSessionContext";

interface MessageListProps {
  messages: Message[];
  loading: boolean;
  isStreaming: boolean;
  streamingContent: string;
  thinkingContent: string;
  isThinking: boolean;
  nodeStatus: string | null;
  toolCallStatus: { name: string; status: "start" | "end"; label: string } | null;
  nodeTrace: NodeTraceEntry[];
  compactWritebackDoc: string | null;
  writebackSummary: string;
  queuedMessage: QueuedMessage | null;
  /** Whether the workspace sidebar is currently open — used for the isFocused ring. */
  workspaceSidebarOpen: boolean;
  focusedAnswerIndex: number | null;
  lastAssistantIndex: number;
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  isAtBottom: boolean;
  unreadCount: number;
  onScrollToBottom: () => void;
  onSelectEvidence: (tab: WorkspaceTabId, index: number) => void;
  onRegenerate: () => void;
  onFeedback: (index: number, value: "up" | "down") => void;
  onEditAndResend: (index: number, newContent: string) => void;
  onClearQueue: () => void;
  onSendQueuedNow: () => void;
  onEditQueued: () => void;
}

export function MessageList({
  messages,
  loading,
  isStreaming,
  streamingContent,
  thinkingContent,
  isThinking,
  nodeStatus,
  toolCallStatus,
  compactWritebackDoc,
  writebackSummary,
  queuedMessage,
  workspaceSidebarOpen,
  focusedAnswerIndex,
  lastAssistantIndex,
  scrollContainerRef,
  messagesEndRef,
  isAtBottom,
  unreadCount,
  onScrollToBottom,
  onSelectEvidence,
  onRegenerate,
  onFeedback,
  onEditAndResend,
  onClearQueue,
  onSendQueuedNow,
  onEditQueued,
}: MessageListProps) {
  const { t } = useI18n();

  return (
    <div
      ref={scrollContainerRef}
      className="flex-1 overflow-y-auto space-y-8 bg-surface p-4 scroll-smooth md:p-6 lg:px-12"
      id="chat-container"
      role="log"
      aria-live="polite"
      aria-label={t("sidebar.chatHistory")}
    >
      {messages.length === 0 && !loading ? (
        <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
          <Bot size={48} className="mb-4 opacity-50" />
          <p className="text-lg font-medium">{t("chat.noMessagesYet")}</p>
        </div>
      ) : (
        <>
          {messages.map((msg, i) =>
            msg.role === "user" ? (
              <UserMessage
                key={i}
                message={msg}
                index={i}
                onEdit={onEditAndResend}
                loading={loading}
              />
            ) : (
              <AssistantMessage
                key={i}
                message={msg}
                index={i}
                isFocused={
                  workspaceSidebarOpen &&
                  focusedAnswerIndex !== null &&
                  i === focusedAnswerIndex &&
                  i !== lastAssistantIndex
                }
                onSelectEvidence={onSelectEvidence}
                onRegenerate={onRegenerate}
                onFeedback={onFeedback}
                loading={loading}
              />
            ),
          )}
        </>
      )}

      {/* Streaming / Typing indicator */}
      <StreamingMessage
        isStreaming={isStreaming}
        loading={loading}
        streamingContent={streamingContent}
        thinkingContent={thinkingContent}
        isThinking={isThinking}
        nodeStatus={nodeStatus}
        toolCallStatus={toolCallStatus}
        compactWritebackDoc={compactWritebackDoc}
        writebackSummary={writebackSummary}
      />

      {/* Queued follow-up (pending bubble) */}
      {queuedMessage && (
        <QueuedMessageBubble
          text={queuedMessage.text}
          idle={!isStreaming && !loading}
          onDiscard={onClearQueue}
          onSendNow={onSendQueuedNow}
          onEdit={onEditQueued}
        />
      )}

      {/* Scroll-to-bottom floating button */}
      <div className="sticky bottom-4 z-10 flex justify-center pointer-events-none">
        <div className="pointer-events-auto">
          <ScrollToBottomButton
            visible={!isAtBottom}
            unreadCount={unreadCount}
            onClick={onScrollToBottom}
          />
        </div>
      </div>

      <div className="h-12" ref={messagesEndRef} />
    </div>
  );
}
