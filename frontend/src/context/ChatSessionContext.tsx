"use client";

import {
  createContext,
  useContext,
  useState,
  useRef,
  useEffect,
  useCallback,
} from "react";
import { useConversationContext } from "@/components/LayoutShell";
import { useI18n } from "@/hooks/useI18n";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { fetchConversation } from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type { ChatResponse, MapData, Message, PersistedMessage, Source } from "@/lib/types";

// ── Helpers (moved from ChatInterface) ──────────────────────────────────

function generateTitle(message: string): string {
  const clean = message.replace(/\s+/g, " ").trim();
  if (!clean) return "New conversation";
  if (clean.length <= 50) return clean;
  const sentenceEnd = clean.substring(0, 60).search(/[.!?]\s/);
  if (sentenceEnd > 10) return clean.substring(0, sentenceEnd + 1);
  const truncated = clean.substring(0, 50);
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace > 20) return truncated.substring(0, lastSpace) + "…";
  return truncated + "…";
}

/** Convert a persisted (DB) message into the local UI Message shape. */
function toUiMessage(
  pm: PersistedMessage,
  formatTime: (value: Date | string | number) => string,
): Message {
  return {
    role: pm.role === "system" ? "assistant" : pm.role,
    content: pm.content,
    thinking: (pm.metadata?.thinking_content as string | undefined) ?? undefined,
    timestamp: pm.created_at ? formatTime(pm.created_at) : undefined,
    persistedId: pm.id,
    feedback: pm.feedback ?? undefined,
    metrics: {
      output_tokens: (pm.metadata?.output_tokens as number | undefined) ?? pm.tokens_used ?? undefined,
      input_tokens: (pm.metadata?.input_tokens as number | undefined) ?? undefined,
      response_time_ms: pm.response_time_ms ?? undefined,
      model: pm.model_name ?? undefined,
      tools_executed: [
        ...(pm.tools_used?.map((t) => ({ name: t.name, status: t.status })) ?? []),
        ...(pm.metadata?.rag_attempted
          ? [{ name: "knowledge_base" as const, status: "success" as const }]
          : []),
      ],
      sources: pm.metadata?.sources as Source[] | undefined,
      rag_attempted: (pm.metadata?.rag_attempted as boolean | undefined) ?? undefined,
      rag_doc_count: (pm.metadata?.rag_doc_count as number | undefined) ?? undefined,
      attachments_used: pm.metadata?.attachments_used as
        | Array<{ id: string; original_name: string }>
        | undefined,
      documents_used: pm.metadata?.documents_used as
        | Array<{ id: string; filename: string; version: number }>
        | undefined,
      memories_used: pm.metadata?.memories_used as
        | Array<{
            memory_id: string;
            text?: string;
            user_rating?: number | null;
            importance_score?: number | null;
            is_related?: boolean;
          }>
        | undefined,
      map_data: pm.metadata?.map_data as MapData | undefined,
    },
  };
}

// ── Context ─────────────────────────────────────────────────────────────

export interface SendMessageOptions {
  attachmentIds?: string[];
  documentIds?: string[];
  ragEnabled?: boolean;
  replaceFromIndex?: number;
}

/** A follow-up message queued by the user while an inference is still streaming. */
export interface QueuedMessage {
  text: string;
  opts: SendMessageOptions;
}

interface ChatSessionValue {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  isStreaming: boolean;
  streamingContent: string;
  thinkingContent: string;
  isThinking: boolean;
  nodeStatus: string | null;
  toolCallStatus: { name: string; status: "start" | "end"; label: string } | null;
  streamError: string | null;
  streamWarning: string | null;
  finalMetadata: ChatResponse["metadata"] | null;
  wasCancelled: boolean;
  contextTokens: number;
  setContextTokens: React.Dispatch<React.SetStateAction<number>>;
  loading: boolean;
  queuedMessage: QueuedMessage | null;
  sendMessage: (userMessage: string, opts?: SendMessageOptions) => Promise<void>;
  enqueueMessage: (text: string, opts?: SendMessageOptions) => void;
  clearQueue: () => void;
  ensureConversation: (seedText?: string) => Promise<string | null>;
  cancelStream: () => void;
  resetStream: () => void;
}

const ChatSessionContext = createContext<ChatSessionValue | null>(null);

export function useChatSession(): ChatSessionValue {
  const ctx = useContext(ChatSessionContext);
  if (!ctx) throw new Error("useChatSession must be used within ChatSessionProvider");
  return ctx;
}

/**
 * Owns the live chat runtime (messages + streaming) so it survives in-app
 * navigation. Mounted in the persistent LayoutShell, it is never unmounted on
 * route changes — an in-flight inference keeps running while the user is on
 * another page, and returning to the chat shows the streaming/completed result.
 */
export function ChatSessionProvider({ children }: { children: React.ReactNode }) {
  const { t, formatTime } = useI18n();
  const { activeId, create, refresh, rename, activeConversation, setWorkspaceSidebarOpen } =
    useConversationContext();

  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [contextTokens, setContextTokens] = useState<number>(0);
  const [queuedMessage, setQueuedMessage] = useState<QueuedMessage | null>(null);

  const {
    streamingContent,
    isStreaming,
    streamError,
    streamWarning,
    toolCallStatus,
    nodeStatus,
    finalMetadata,
    wasCancelled,
    thinkingContent,
    isThinking,
    sendMessage: startStream,
    cancel: rawCancel,
    reset: resetStream,
  } = useStreamingChat();

  // Flag: when true the next activeId change came from create() and the
  // message-fetch effect should skip reloading (no persisted messages yet).
  const skipNextFetchRef = useRef(false);
  const startTimeRef = useRef<number>(0);
  const wasStreamingRef = useRef(false);
  // Conversation a running stream belongs to — guards against committing a
  // response into a conversation the user has switched to.
  const streamConversationRef = useRef<string | null>(null);
  // Live activeId, so the deferred queue auto-fire can bail if the user
  // switched conversations during the setTimeout(0) window.
  const activeIdRef = useRef<string | null>(activeId);
  activeIdRef.current = activeId;

  // ── Follow-up queue ────────────────────────────────────────────────────
  // One queued message the user typed while a stream was in flight. Read from
  // a ref inside the commit effect (latest value, no stale closure / extra dep).
  const queuedMessageRef = useRef<QueuedMessage | null>(null);
  // True when the just-finished stream ended via an explicit user cancel.
  // Set synchronously *before* the hook's cancel because the hook's own
  // `wasCancelled` is only set later (in catch/finally), too late for the
  // commit effect to read on the isStreaming true→false transition.
  const manualStopRef = useRef(false);
  // Always points at the latest sendMessage so the commit effect can fire the
  // queue without taking sendMessage as a dependency (correctness relies on the
  // setMessages(prev => …) updater form, not on this closure being fresh).
  const sendMessageRef = useRef<((text: string, opts?: SendMessageOptions) => Promise<void>) | undefined>(undefined);

  const enqueueMessage = useCallback((text: string, opts: SendMessageOptions = {}) => {
    const q: QueuedMessage = { text, opts };
    queuedMessageRef.current = q;
    setQueuedMessage(q);
  }, []);

  const clearQueue = useCallback(() => {
    queuedMessageRef.current = null;
    setQueuedMessage(null);
  }, []);

  // Wrap the hook's cancel so an explicit user stop is recorded before the
  // stream tears down (see manualStopRef rationale above).
  const cancelStream = useCallback(() => {
    manualStopRef.current = true;
    rawCancel();
  }, [rawCancel]);

  // ── Load messages when the active conversation changes ─────────────────
  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setWorkspaceSidebarOpen(false);
      setContextTokens(0);
      cancelStream();
      clearQueue();
      return;
    }
    // When a conversation was just created the DB has no messages yet.
    // Skip the fetch so the optimistic user message is not overwritten.
    if (skipNextFetchRef.current) {
      skipNextFetchRef.current = false;
      return;
    }
    // Switching to a different existing conversation: abandon any in-flight
    // stream from the previous one so it can't bleed into this view, and drop
    // any follow-up queued against the previous conversation.
    cancelStream();
    clearQueue();
    let cancelled = false;
    (async () => {
      const detail = await fetchConversation(activeId);
      if (cancelled || !detail) return;
      const loadedMessages = detail.messages.map((msg) => toUiMessage(msg, formatTime));
      // Restore the ring to the most recent inference's prompt size (live fill).
      const lastAssistantTokens =
        [...loadedMessages]
          .reverse()
          .find((m) => m.role === "assistant" && (m.metrics?.input_tokens ?? 0) > 0)
          ?.metrics?.input_tokens ?? 0;
      setContextTokens(lastAssistantTokens);
      setMessages(loadedMessages);
      if (detail.messages.length === 0) {
        setWorkspaceSidebarOpen(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId, formatTime]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Commit completed streaming message (durable — runs even while the chat
  //     view is unmounted) ────────────────────────────────────────────────
  useEffect(() => {
    const was = wasStreamingRef.current;
    wasStreamingRef.current = isStreaming;
    if (was && !isStreaming) {
      const belongsToActive = streamConversationRef.current === activeId;
      if (belongsToActive && (streamingContent || wasCancelled)) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant" as const,
            content: wasCancelled
              ? streamingContent + (streamingContent ? "\n\n*(generation stopped)*" : "*(generation stopped)*")
              : streamingContent,
            thinking: thinkingContent || undefined,
            timestamp: formatTime(new Date()),
            metrics: finalMetadata
              ? {
                  response_time_ms: Date.now() - startTimeRef.current,
                  input_tokens: finalMetadata.input_tokens,
                  output_tokens: finalMetadata.output_tokens,
                  finish_reason: "stop",
                  model: finalMetadata.model_used,
                  tools_executed: finalMetadata.tools_executed?.map((name) => ({
                    name,
                    status: "success" as const,
                  })),
                  sources: finalMetadata.sources,
                  memories_used: finalMetadata.memories_used,
                  rag_attempted: finalMetadata.rag_attempted,
                  rag_doc_count: finalMetadata.rag_doc_count,
                  map_data: finalMetadata.map_data,
                }
              : undefined,
          },
        ]);

        // Fetch the conversation to attach DB message IDs so feedback (thumbs
        // up/down) can be persisted. _run_persistence completes before [DONE].
        if (activeId) {
          fetchConversation(activeId).then((detail) => {
            if (!detail) return;
            setMessages((prev) => {
              const dbMsgs = detail.messages;
              let dbIdx = 0;
              return prev.map((msg) => {
                if (msg.persistedId != null) {
                  while (dbIdx < dbMsgs.length && dbMsgs[dbIdx].id !== msg.persistedId) dbIdx++;
                  dbIdx++;
                  return msg;
                }
                while (dbIdx < dbMsgs.length && dbMsgs[dbIdx].role !== msg.role) dbIdx++;
                const dbMsg = dbMsgs[dbIdx];
                dbIdx++;
                return dbMsg ? { ...msg, persistedId: dbMsg.id } : msg;
              });
            });
          });
        }
      }
      resetStream();

      // ── Auto-fire a queued follow-up ───────────────────────────────────
      // Only on a normal completion: an explicit Stop (manualStopRef) or a
      // stream error keeps the queued message pending so the user can decide.
      // streamError is reliable here (set in catch, batched with the finally's
      // setIsStreaming(false)); wasCancelled is NOT — hence manualStopRef.
      const completedNormally = !manualStopRef.current && !streamError;
      manualStopRef.current = false;
      if (
        completedNormally &&
        queuedMessageRef.current &&
        streamConversationRef.current === activeId
      ) {
        const q = queuedMessageRef.current;
        const convAtSchedule = activeId;
        queuedMessageRef.current = null;
        setQueuedMessage(null);
        // Defer to a macrotask so the assistant-bubble commit above and the
        // finishing turn's setLoading(false) flush first, then the queued turn
        // starts cleanly (avoids a loading-race where the new stream's
        // loading=true is clobbered by the prior turn's trailing setLoading).
        // Bail if the user switched conversations during the window.
        setTimeout(() => {
          if (activeIdRef.current !== convAtSchedule) return;
          sendMessageRef.current?.(q.text, q.opts);
        }, 0);
      }
    }
  }, [isStreaming, activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Live context-window fill: latest inference's prompt size ───────────
  useEffect(() => {
    const tokens = finalMetadata?.input_tokens;
    if (!isStreaming && tokens != null && tokens > 0) {
      setContextTokens(tokens);
    }
  }, [isStreaming, finalMetadata]);

  const ensureConversation = useCallback(
    async (seedText: string = t("chat.newConversation")) => {
      if (activeId) return activeId;
      const autoTitle = generateTitle(seedText);
      skipNextFetchRef.current = true;
      const conv = await create(autoTitle);
      if (!conv) {
        skipNextFetchRef.current = false;
        return null;
      }
      return conv.id;
    },
    [activeId, create, t],
  );

  const sendMessage = useCallback(
    async (userMessage: string, opts: SendMessageOptions = {}) => {
      const { attachmentIds = [], documentIds = [], ragEnabled = true, replaceFromIndex } = opts;
      let convId = activeId;
      const isFirstMessage = messages.length === 0 || replaceFromIndex === 0;
      if (!convId) {
        convId = await ensureConversation(userMessage);
        if (!convId) return;
      }
      streamConversationRef.current = convId;

      if (replaceFromIndex != null) {
        setMessages((prev) => [
          ...prev.slice(0, replaceFromIndex),
          { role: "user", content: userMessage, timestamp: formatTime(new Date()) },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "user", content: userMessage, timestamp: formatTime(new Date()) },
        ]);
      }
      setLoading(true);
      startTimeRef.current = Date.now();
      // Every new inference starts clean: a prior manual stop must not bleed
      // into this turn's completion check.
      manualStopRef.current = false;

      // Transient preview only: show at least the prior turn's fill (or a rough
      // chars/4 estimate) while the LLM generates. The real per-inference
      // input_tokens overwrites this via direct set when metadata arrives.
      const _estTokens = Math.round(
        messages.reduce((s, m) => s + m.content.length, 0) / 4 + userMessage.length / 4,
      );
      setContextTokens((prev) => Math.max(prev, _estTokens));

      await startStream({
        message: userMessage,
        userId: CURRENT_USER_ID,
        conversationId: convId,
        attachmentIds,
        documentIds,
        ragEnabled,
      });

      setLoading(false);
      refresh();
      if (
        isFirstMessage &&
        convId &&
        (activeConversation?.title === "New conversation" ||
          activeConversation?.title === t("chat.newConversation"))
      ) {
        const betterTitle = generateTitle(userMessage);
        if (betterTitle !== t("chat.newConversation")) rename(convId, betterTitle);
      }
    },
    [
      activeId,
      messages,
      ensureConversation,
      refresh,
      rename,
      activeConversation?.title,
      startStream,
      t,
      formatTime,
    ],
  );

  // Keep the ref pointing at the latest sendMessage for the commit effect's
  // deferred queue auto-fire.
  sendMessageRef.current = sendMessage;

  const value: ChatSessionValue = {
    messages,
    setMessages,
    isStreaming,
    streamingContent,
    thinkingContent,
    isThinking,
    nodeStatus,
    toolCallStatus,
    streamError,
    streamWarning,
    finalMetadata,
    wasCancelled,
    contextTokens,
    setContextTokens,
    loading,
    queuedMessage,
    sendMessage,
    enqueueMessage,
    clearQueue,
    ensureConversation,
    cancelStream,
    resetStream,
  };

  return <ChatSessionContext.Provider value={value}>{children}</ChatSessionContext.Provider>;
}
