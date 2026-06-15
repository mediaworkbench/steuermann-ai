"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { ActiveDocumentPaneSlot } from "./workspace/ActiveDocumentPane";
import { ActiveDocumentProvider, type ActiveDocumentEditorApi } from "@/context/ActiveDocumentContext";
import { ChatComposer, type ChatComposerHandle } from "./chat/ChatComposer";
import { MessageList } from "./chat/MessageList";
import { WorkspaceSidebar } from "./WorkspaceSidebar";
import { useConversationContext } from "./LayoutShell";
import { useChatSession } from "@/context/ChatSessionContext";
import { useEdgeEffect } from "@/hooks/useEdgeEffect";
import { usePanelAnswer } from "@/hooks/usePanelAnswer";
import { useWorkspaceDocuments } from "@/hooks/useWorkspaceDocuments";
import { useComposerSettings } from "@/hooks/useComposerSettings";
import { useConversationAttachments } from "@/hooks/useConversationAttachments";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { useI18n } from "@/hooks/useI18n";
import { setMessageFeedback } from "@/lib/api";
import { splitWritebackStream, looksLikeWriteback } from "@/lib/writeback";

export function ChatInterface() {
  const { t } = useI18n();
  const [input, setInput] = useState("");
  const [activeWorkspaceDocId, setActiveWorkspaceDocId] = useState<string | null>(null);
  const [writebackSavedDocId, setWritebackSavedDocId] = useState<string | null>(null);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false);
  const [contextMenuOpen, setContextMenuOpen] = useState(false);
  const [isCompacting, setIsCompacting] = useState(false);
  const [hasNewMessage, setHasNewMessage] = useState(false);

  const activeDocEditorRef = useRef<ActiveDocumentEditorApi | null>(null);
  const composerApiRef = useRef<ChatComposerHandle | null>(null);
  const plopAudioRef = useRef<HTMLAudioElement | null>(null);

  const {
    messages,
    setMessages,
    streamingContent,
    isStreaming,
    streamError,
    streamWarning,
    toolCallStatus,
    nodeStatus,
    nodeTrace,
    finalMetadata,
    writebackPending,
    wasCancelled,
    thinkingContent,
    isThinking,
    contextTokens,
    setContextTokens,
    contextBreakdown,
    loading,
    queuedMessage,
    sendMessage,
    enqueueMessage,
    clearQueue,
    ensureConversation,
    cancelStream,
  } = useChatSession();

  const { activeId, refresh: refreshConversations, workspaceSidebarOpen, setWorkspaceSidebarOpen } =
    useConversationContext();

  const { documents, documentsLoading, documentsError, refresh: refreshDocs } =
    useWorkspaceDocuments();

  const {
    attachments,
    uploadingAttachment,
    addAttachment,
    handleAttachmentUpload: _handleAttachmentUpload,
    handleAttachmentDelete,
  } = useConversationAttachments(activeId);

  const {
    ragEnabled,
    handleRagToggle,
    toolToggles,
    handleToolToggle,
    allowedTools,
    selectedChatModel,
    availableChatModels,
    handleModelChange,
    systemConfig,
    soundEnabled,
    showMetrics,
    maxContextTokens,
  } = useComposerSettings();

  const {
    focusedAnswerIndex,
    lastAssistantIndex,
    panelMetrics,
    panelNodeTrace,
    panelIsStreaming,
    isHistoricalAnswer,
    handleSelectEvidence,
    handleJumpToLatest,
  } = usePanelAnswer(messages, isStreaming, nodeTrace);

  const { scrollContainerRef, messagesEndRef, isAtBottom, unreadCount, scrollToBottom, shouldAutoScroll } =
    useScrollToBottom(messages.length);

  // ── Compact writeback view ────────────────────────────────────────────
  const compactWritebackDoc = writebackPending
    ? writebackPending.filename
    : isStreaming && looksLikeWriteback(streamingContent) && activeWorkspaceDocId
      ? documents.find((d) => d.id === activeWorkspaceDocId)?.filename ?? null
      : null;
  const writebackSummary = compactWritebackDoc
    ? splitWritebackStream(streamingContent).summary
    : "";

  // ── Scroll behaviour ─────────────────────────────────────────────────

  // Conversation switch: jump to bottom immediately regardless of scroll position
  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => scrollToBottom("instant"), 0);
    }
  }, [activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Streaming / new message: auto-scroll only when already at bottom
  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom("smooth");
    }
  }, [messages, loading, isStreaming, streamingContent, queuedMessage, shouldAutoScroll]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Tab title badge ──────────────────────────────────────────────────

  useEffect(() => {
    const BASE = "Steuermann";
    document.title = hasNewMessage ? `🔴 ${BASE}` : BASE;
  }, [hasNewMessage]);

  useEffect(() => {
    const clear = () => { if (!document.hidden) setHasNewMessage(false); };
    document.addEventListener("visibilitychange", clear);
    window.addEventListener("focus", clear);
    return () => {
      document.removeEventListener("visibilitychange", clear);
      window.removeEventListener("focus", clear);
    };
  }, []);

  // ── Stream-end UX (sound, unread badge, writeback toast/refresh) ─────
  // Fires only while the chat view is mounted. Durable effects (message
  // commit, persisted-id backfill, context-token update) live in
  // ChatSessionProvider so they run even when the user is on another page.
  useEdgeEffect(isStreaming, {
    onFalling: () => {
      if (!(streamingContent || wasCancelled)) return;
      if (finalMetadata?.workspace_document_writeback?.status === "saved") {
        const wb = finalMetadata.workspace_document_writeback;
        refreshDocs();
        setWritebackSavedDocId(wb.document_id ?? null);
        setTimeout(() => setWritebackSavedDocId(null), 200);
        toast.success(t("chat.workspaceDocumentSaved"), {
          description: `${wb.filename} updated to v${wb.version}`,
        });
      }
      if (soundEnabled) {
        plopAudioRef.current?.play().catch(() => {});
      }
      if (document.hidden || !document.hasFocus()) {
        setHasNewMessage(true);
      }
    },
  });

  // ── Stream error / warning toasts ────────────────────────────────────

  useEffect(() => {
    if (streamError) {
      toast.error(t("chat.messageFailed"), { description: streamError });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamError]);

  useEffect(() => {
    if (streamWarning) {
      toast.warning(streamWarning);
    }
  }, [streamWarning]);

  // Preload plop audio on mount
  useEffect(() => {
    plopAudioRef.current = new Audio("/plop.mp3");
    plopAudioRef.current.load();
  }, []);

  // Tab key focuses the composer from anywhere on the page
  useEffect(() => {
    const onTabKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      if (document.activeElement?.closest('[role="dialog"]')) return;
      e.preventDefault();
      composerApiRef.current?.focus();
    };
    document.addEventListener("keydown", onTabKey);
    return () => document.removeEventListener("keydown", onTabKey);
  }, []);

  // ── Derived state ────────────────────────────────────────────────────

  const queueFull = queuedMessage != null;
  const userMessageCount = messages.filter((m) => m.role === "user").length;
  const assistantMessageCount = messages.filter((m) => m.role === "assistant").length;

  // ── Handlers ─────────────────────────────────────────────────────────

  const buildSendOptions = useCallback(
    () => ({
      attachmentIds: attachments.map((a) => a.id),
      documentIds: activeWorkspaceDocId ? [activeWorkspaceDocId] : [],
      ragEnabled,
    }),
    [attachments, activeWorkspaceDocId, ragEnabled],
  );

  const flushActiveDocBeforeSend = useCallback(async (): Promise<boolean> => {
    const api = activeDocEditorRef.current;
    if (api?.isDirty) return api.flushSave();
    return true;
  }, []);

  async function handleSend() {
    if (!input.trim()) return;
    if (!(await flushActiveDocBeforeSend())) return;
    const userMessage = input;
    setInput("");
    setTimeout(() => composerApiRef.current?.resize(), 0);
    if (isStreaming || loading) {
      enqueueMessage(userMessage, buildSendOptions());
    } else {
      sendMessage(userMessage, buildSendOptions());
    }
  }

  const handleRegenerate = useCallback(async () => {
    if (!(await flushActiveDocBeforeSend())) return;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        sendMessage(messages[i].content, { ...buildSendOptions(), replaceFromIndex: i });
        return;
      }
    }
  }, [messages, sendMessage, buildSendOptions, flushActiveDocBeforeSend]);

  const handleEditAndResend = useCallback(
    async (index: number, newContent: string) => {
      if (!(await flushActiveDocBeforeSend())) return;
      sendMessage(newContent, { ...buildSendOptions(), replaceFromIndex: index });
    },
    [sendMessage, buildSendOptions, flushActiveDocBeforeSend],
  );

  const handleFeedback = useCallback(
    async (index: number, value: "up" | "down") => {
      const msg = messages[index];
      if (!msg || !activeId) return;
      const newFeedback = msg.feedback === value ? undefined : value;
      setMessages((prev) =>
        prev.map((m, i) => (i === index ? { ...m, feedback: newFeedback } : m)),
      );
      if (msg.persistedId) {
        await setMessageFeedback(activeId, msg.persistedId, newFeedback ?? null);
        toast.success(newFeedback ? t("chat.feedbackSaved") : t("chat.feedbackRemoved"));
      }
    },
    [messages, setMessages, activeId, t],
  );

  const handleCompactContext = useCallback(async () => {
    if (!activeId) return;
    setIsCompacting(true);
    try {
      const res = await fetch(`/api/proxy/api/conversations/${activeId}/compact`, {
        method: "POST",
        headers: { "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "" },
      });
      if (res.ok) {
        const data = await res.json() as {
          status: string;
          estimated_tokens?: number;
          messages_before?: number;
          messages_after?: number;
        };
        if (data.status === "ok") {
          if ((data.estimated_tokens ?? 0) > 0) setContextTokens(data.estimated_tokens!);
          setContextMenuOpen(false);
          const removed =
            data.messages_before != null && data.messages_after != null
              ? data.messages_before - data.messages_after
              : 0;
          toast.success(
            removed > 0
              ? `Compacted ${removed} messages into a summary — the visible transcript is unchanged`
              : "Context compacted — the visible transcript is unchanged",
          );
        } else if (data.status === "skipped") {
          toast.info("Nothing to compact — context is already small");
        } else if (data.status === "error") {
          toast.error("Compaction failed — conversation unchanged");
        } else {
          setContextMenuOpen(false);
        }
      } else {
        toast.error("Compact failed");
      }
    } finally {
      setIsCompacting(false);
    }
  }, [activeId, setContextTokens]);

  // Wrap the attachment upload handler to supply the options it needs.
  const handleAttachmentUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      void _handleAttachmentUpload(event, {
        loading,
        ensureConversation,
        onUploaded: () => setWorkspaceSidebarOpen(true),
        refresh: () => { refreshDocs(); refreshConversations(); },
      });
    },
    [_handleAttachmentUpload, loading, ensureConversation, setWorkspaceSidebarOpen, refreshDocs, refreshConversations],
  );

  const handleAttachmentPillClick = useCallback((attachment: { id: string; original_name: string }) => {
    const ref = `"${attachment.original_name}" (id: ${attachment.id})`;
    setInput((prev) => {
      // Can't access selectionStart here (no DOM access in state updater),
      // so just append — the full cursor-aware insertion is only possible via
      // an imperative textarea ref, which the composer owns.
      return prev + ref;
    });
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape" && isStreaming) {
      cancelStream();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  }

  return (
    <ActiveDocumentProvider
      documents={documents}
      conversationId={activeId}
      writebackSavedDocId={writebackSavedDocId}
      editorApiRef={activeDocEditorRef}
      onActiveDocumentChange={setActiveWorkspaceDocId}
      onDocumentsRefresh={refreshDocs}
      onAttachmentUploaded={(attachment) => {
        addAttachment(attachment);
        setWorkspaceSidebarOpen(true);
      }}
    >
      <div className="flex h-full min-h-0">
        {/* ─── Main chat area ─── */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          <MessageList
            messages={messages}
            loading={loading}
            isStreaming={isStreaming}
            streamingContent={streamingContent}
            thinkingContent={thinkingContent}
            isThinking={isThinking}
            nodeStatus={nodeStatus}
            toolCallStatus={toolCallStatus}
            nodeTrace={nodeTrace}
            compactWritebackDoc={compactWritebackDoc}
            writebackSummary={writebackSummary}
            queuedMessage={queuedMessage}
            workspaceSidebarOpen={workspaceSidebarOpen}
            focusedAnswerIndex={focusedAnswerIndex}
            lastAssistantIndex={lastAssistantIndex}
            scrollContainerRef={scrollContainerRef}
            messagesEndRef={messagesEndRef}
            isAtBottom={isAtBottom}
            unreadCount={unreadCount}
            onScrollToBottom={() => scrollToBottom("smooth")}
            onSelectEvidence={handleSelectEvidence}
            onRegenerate={handleRegenerate}
            onFeedback={handleFeedback}
            onEditAndResend={handleEditAndResend}
            onClearQueue={clearQueue}
            onSendQueuedNow={() => {
              const q = queuedMessage;
              if (!q) return;
              clearQueue();
              sendMessage(q.text, q.opts);
            }}
            onEditQueued={() => {
              if (input.trim()) return;
              if (!queuedMessage) return;
              setInput(queuedMessage.text);
              clearQueue();
              setTimeout(() => {
                composerApiRef.current?.focus();
                composerApiRef.current?.resize();
              }, 0);
            }}
            showMetrics={showMetrics}
          />

          <ChatComposer
            input={input}
            onInputChange={setInput}
            onSend={() => void handleSend()}
            onCancel={cancelStream}
            onKeyDown={handleKeyDown}
            isStreaming={isStreaming}
            loading={loading}
            queueFull={queueFull}
            attachments={attachments}
            uploadingAttachment={uploadingAttachment}
            onAttachmentUpload={handleAttachmentUpload}
            onAttachmentDelete={handleAttachmentDelete}
            onAttachmentPillClick={handleAttachmentPillClick}
            ragEnabled={ragEnabled}
            onRagToggle={handleRagToggle}
            workspaceSidebarOpen={workspaceSidebarOpen}
            onWorkspaceSidebarToggle={() => setWorkspaceSidebarOpen(!workspaceSidebarOpen)}
            toolToggles={toolToggles}
            onToolToggle={handleToolToggle}
            allowedTools={allowedTools}
            systemConfig={systemConfig}
            selectedChatModel={selectedChatModel}
            availableChatModels={availableChatModels}
            onModelChange={handleModelChange}
            contextTokens={contextTokens}
            maxContextTokens={maxContextTokens}
            contextBreakdown={contextBreakdown}
            userMessageCount={userMessageCount}
            assistantMessageCount={assistantMessageCount}
            isCompacting={isCompacting}
            activeId={activeId}
            onCompactContext={handleCompactContext}
            attachMenuOpen={attachMenuOpen}
            onAttachMenuToggle={() => setAttachMenuOpen((v) => !v)}
            toolsMenuOpen={toolsMenuOpen}
            onToolsMenuToggle={() => setToolsMenuOpen((v) => !v)}
            contextMenuOpen={contextMenuOpen}
            onContextMenuToggle={() => setContextMenuOpen((v) => !v)}
            composerApiRef={composerApiRef}
          />
        </div>

        {/* ─── Active document split-view pane ─── */}
        {/* Mounts when a doc is open for editing OR its version history is open
            (the slot reads editor/history state from the provider it sits in). */}
        <ActiveDocumentPaneSlot isLoading={loading} />

        {/* ─── Workspace sidebar ─── */}
        <WorkspaceSidebar
          isOpen={workspaceSidebarOpen}
          onToggle={() => setWorkspaceSidebarOpen(!workspaceSidebarOpen)}
          conversationId={activeId}
          documents={documents}
          isLoading={loading}
          onDocumentsRefresh={refreshDocs}
          onRetryDocuments={refreshDocs}
          documentsLoading={documentsLoading}
          documentsError={documentsError}
          onEnsureConversation={() => ensureConversation()}
          answerMetrics={panelMetrics}
          nodeTrace={panelNodeTrace}
          isStreaming={panelIsStreaming}
          historicalAnswer={isHistoricalAnswer}
          onJumpToLatest={handleJumpToLatest}
          onAttachmentUploaded={(attachment) => {
            addAttachment(attachment);
            setWorkspaceSidebarOpen(true);
          }}
        />
      </div>
    </ActiveDocumentProvider>
  );
}
