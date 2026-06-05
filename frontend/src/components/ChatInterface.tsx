"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { Icon } from "./Icon";
import { MarkdownMessage } from "./MarkdownMessage";
import { ContextRingIndicator } from "./ContextRingIndicator";
import { MetricsPanel } from "./MetricsPanel";
import { ReasoningBox } from "./ReasoningBox";
import dynamic from "next/dynamic";
const MapWidget = dynamic(() => import("./MapWidget").then((m) => m.MapWidget), { ssr: false });
import { WorkspaceSidebar, type WorkspaceDocument } from "./WorkspaceSidebar";
import { EvidenceChips } from "./workspace/EvidenceChips";
import { useConversationContext } from "./LayoutShell";
import { useChatSession } from "@/context/ChatSessionContext";
import { useI18n } from "@/hooks/useI18n";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { ScrollToBottomButton } from "@/components/ScrollToBottomButton";
import {
  deleteConversationAttachment,
  fetchConversationAttachments,
  fetchSystemConfig,
  fetchUserSettings,
  setMessageFeedback,
  updateUserSettings,
  uploadConversationAttachment,
} from "@/lib/api";
import type { SystemConfig } from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type {
  ConversationAttachment,
  Message,
  MessageMetrics,
  Source,
} from "@/lib/types";

const FALLBACK_TOOLS = [
  { id: "web_search_mcp", label: "Web Search" },
  { id: "extract_webpage_mcp", label: "Extract Webpage" },
  { id: "analyze_image_tool", label: "Analyze Image" },
  { id: "ocr_tool", label: "OCR" },
  { id: "analyze_document_tool", label: "Analyze Document" },
  { id: "analyze_chart_tool", label: "Analyze Chart" },
  { id: "image_metadata_tool", label: "Image Metadata" },
  { id: "read_barcodes_tool", label: "Read Barcodes" },
  { id: "datetime_tool", label: "Datetime" },
  { id: "calculator_tool", label: "Calculator" },
  { id: "map_tool", label: "Map" },
  { id: "file_ops_tool", label: "File Ops" },
] as const;

function formatModelName(model: string | null | undefined, fallback = "Model"): string {
  const m = model || fallback;
  const parts = m.split("/");
  return parts.length > 1 ? parts.slice(1).join("/") : m;
}

/** Render source badges below a message — blue for web (clickable), amber for RAG */
function SourceBadges({ sources }: { sources?: Source[] }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 px-1 mt-2">
      {sources.map((src, idx) =>
        src.type === "web" && src.url ? (
          <a
            key={`${src.label}-${idx}`}
            href={src.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                       bg-pacific-blue/10 text-pacific-blue border border-pacific-blue/20
                       hover:bg-pacific-blue/20 transition-colors no-underline"
          >
            <Icon name="language" size={12} />
            {src.label}
          </a>
        ) : (
          <span
            key={`${src.label}-${idx}`}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                       bg-amber-100 text-amber-800 border border-amber-200"
          >
            <Icon name="menu_book" size={12} />
            {src.label}
          </span>
        ),
      )}
    </div>
  );
}

/** Render small file chips below an assistant message for each document used as context. */
function AttachmentUsedBadges({
  attachments,
}: {
  attachments?: Array<{ id: string; original_name: string }>;
}) {
  if (!attachments || attachments.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 px-1 mt-1.5">
      {attachments.map((att) => (
        <span
          key={att.id}
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                     bg-pacific-blue/5 text-evergreen/55 border border-gray-200"
          title={`Context from: ${att.original_name}`}
        >
          <Icon name="description" size={12} />
          {att.original_name}
        </span>
      ))}
    </div>
  );
}

function DocumentUsedBadges({
  documents,
}: {
  documents?: Array<{ id: string; filename: string; version: number }>;
}) {
  if (!documents || documents.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 px-1 mt-1.5">
      {documents.map((doc) => (
        <span
          key={doc.id}
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                     bg-emerald-50 text-emerald-800 border border-emerald-200"
          title={`Workspace context: ${doc.filename} (v${doc.version})`}
        >
          <Icon name="folder_open" size={12} />
          {doc.filename}
          <span className="text-emerald-700/75">v{doc.version}</span>
        </span>
      ))}
    </div>
  );
}



function CtxRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-evergreen/60">{label}</span>
      <span className="text-evergreen tabular-nums">{value}</span>
    </div>
  );
}

export function ChatInterface() {
  const { t } = useI18n();
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<ConversationAttachment[]>([]);
  const [uploadingAttachment, setUploadingAttachment] = useState(false);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [writebackSavedDocId, setWritebackSavedDocId] = useState<string | null>(null);
  const [activeWorkspaceDocId, setActiveWorkspaceDocId] = useState<string | null>(null);
  const [ragEnabled, setRagEnabled] = useState<boolean>(true);
  const [ragConfig, setRagConfig] = useState<Record<string, unknown>>({ collection: "", top_k: 5, enabled: true });
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [toolToggles, setToolToggles] = useState<Record<string, boolean>>({});
  const [chatModel, setChatModel] = useState<string | null>(null);
  const [availableChatModels, setAvailableChatModels] = useState<string[]>([]);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [contextMenuOpen, setContextMenuOpen] = useState(false);
  const [isCompacting, setIsCompacting] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [hasNewMessage, setHasNewMessage] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const preferredModelsRef = useRef<Record<string, string | null>>({});
  const plopAudioRef = useRef<HTMLAudioElement | null>(null);

  // Live chat runtime lives in a persistent provider so it survives in-app
  // navigation (the stream keeps running while the user is on another page).
  const {
    messages,
    setMessages,
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
    contextTokens,
    setContextTokens,
    loading,
    queuedMessage,
    sendMessage,
    enqueueMessage,
    clearQueue,
    ensureConversation,
    cancelStream,
  } = useChatSession();

  const { activeId, refresh, workspaceSidebarOpen, setWorkspaceSidebarOpen } =
    useConversationContext();

  // Latest assistant answer in the active conversation. Sourced from the
  // messages array (already active-scoped), so the evidence tabs and the
  // latest-answer chips cannot bleed from a backgrounded stream.
  const lastAssistantIndex = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  }, [messages]);
  const latestAnswerMetrics = useMemo<MessageMetrics | null>(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.metrics) return m.metrics;
    }
    return null;
  }, [messages]);

  // ── Load workspace documents ─────────────────
  const fetchWorkspaceDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const response = await fetch("/api/proxy/api/workspace/documents", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "x-chat-token": process.env.NEXT_PUBLIC_API_TOKEN || "",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to load documents: ${response.statusText}`);
      }

      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      // Documents are optional; surface the failure in the panel rather than crash.
      console.warn("Failed to load workspace documents:", err);
      setDocumentsError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  // Load documents on mount
  useEffect(() => {
    fetchWorkspaceDocuments();
  }, [fetchWorkspaceDocuments]);

  // Attachments are conversation-scoped and re-fetched on conversation change.
  useEffect(() => {
    if (!activeId) {
      setAttachments([]);
      return;
    }
    let cancelled = false;
    (async () => {
      const data = await fetchConversationAttachments(activeId);
      if (cancelled) return;
      setAttachments(data);
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  const { scrollContainerRef, messagesEndRef, isAtBottom, unreadCount, scrollToBottom, shouldAutoScroll } =
    useScrollToBottom(messages.length);

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

  // Stream-end UX (sound, unread badge, workspace-writeback toast/refresh) —
  // fires only while the chat view is mounted. The durable message commit, the
  // persisted-id backfill, and the context-token update all live in
  // ChatSessionProvider, so they run even when the user is on another page.
  const uxWasStreamingRef = useRef(false);
  useEffect(() => {
    const was = uxWasStreamingRef.current;
    uxWasStreamingRef.current = isStreaming;
    if (was && !isStreaming && (streamingContent || wasCancelled)) {
      if (finalMetadata?.workspace_document_writeback?.status === "saved") {
        const wb = finalMetadata.workspace_document_writeback;
        fetchWorkspaceDocuments();
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
    }
  }, [isStreaming, soundEnabled]); // eslint-disable-line react-hooks/exhaustive-deps

  // Surface stream errors as toasts (loading is reset by sendMessage in the provider)
  useEffect(() => {
    if (streamError) {
      toast.error(t("chat.messageFailed"), { description: streamError });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamError]);

  // Surface model validation warnings as toasts
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

  // Tab title badge: prefix with • when a new message arrives while the tab is hidden
  useEffect(() => {
    const BASE = "Steuermann";
    document.title = hasNewMessage ? `🔴 ${BASE}` : BASE;
  }, [hasNewMessage]);

  // Clear the badge whenever the tab becomes visible or regains focus
  useEffect(() => {
    const clear = () => { if (!document.hidden) setHasNewMessage(false); };
    document.addEventListener("visibilitychange", clear);
    window.addEventListener("focus", clear);
    return () => {
      document.removeEventListener("visibilitychange", clear);
      window.removeEventListener("focus", clear);
    };
  }, []);

  // Load user settings on mount: RAG config, tool toggles, chat model, sound
  useEffect(() => {
    fetchUserSettings(CURRENT_USER_ID).then((s) => {
      if (!s) return;
      const cfg = (s.rag_config as Record<string, unknown>) || { collection: "", top_k: 5, enabled: true };
      setRagConfig(cfg);
      setRagEnabled((cfg.enabled as boolean) !== false);
      if (s.tool_toggles) setToolToggles(s.tool_toggles);
      const model = s.preferred_models?.chat ?? s.preferred_model ?? null;
      setChatModel(model);
      preferredModelsRef.current = s.preferred_models ?? {};
      const prefs = s.analytics_preferences as Record<string, unknown> | undefined;
      setSoundEnabled((prefs?.sound_enabled as boolean) ?? true);
    }).catch(() => {});
  }, []);

  // Load system config for tools list and available models
  useEffect(() => {
    fetchSystemConfig().then((config) => {
      if (!config) return;
      setSystemConfig(config);
      const chatRole = config.model_roles?.find((r) => r.role === "chat");
      if (chatRole) {
        setAvailableChatModels(
          Array.from(new Set([chatRole.default_model, ...chatRole.available_models].filter(Boolean)))
        );
      }
    });
  }, []);

  const handleRagToggle = useCallback(async () => {
    const next = !ragEnabled;
    const updated = { ...ragConfig, enabled: next };
    setRagEnabled(next);
    setRagConfig(updated);
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || "/api/proxy";
    await fetch(`${apiBase}/api/settings/user/${CURRENT_USER_ID}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rag_config: updated }),
    });
  }, [ragEnabled, ragConfig]);

  const handleToolToggle = useCallback(async (toolId: string) => {
    const next = { ...toolToggles, [toolId]: toolToggles[toolId] !== false ? false : true };
    setToolToggles(next);
    await updateUserSettings(CURRENT_USER_ID, { tool_toggles: next });
  }, [toolToggles]);

  const handleModelChange = useCallback(async (model: string) => {
    setChatModel(model);
    setModelMenuOpen(false);
    const merged = { ...preferredModelsRef.current, chat: model };
    preferredModelsRef.current = merged;
    await updateUserSettings(CURRENT_USER_ID, { preferred_model: model, preferred_models: merged });
  }, []);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxH = 260; // ~10 lines
    el.style.height = Math.min(el.scrollHeight, maxH) + "px";
    el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
  }, []);

  // Build the per-send options from the composer's current UI state. The send
  // orchestration itself (messages append, stream start, auto-title) lives in
  // ChatSessionProvider so it survives navigation.
  const buildSendOptions = useCallback(
    () => ({
      attachmentIds: attachments.map((a) => a.id),
      documentIds: activeWorkspaceDocId ? [activeWorkspaceDocId] : [],
      ragEnabled,
    }),
    [attachments, activeWorkspaceDocId, ragEnabled],
  );

  const handleAttachmentUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file || uploadingAttachment || loading) return;

      setUploadingAttachment(true);
      try {
        const convId = await ensureConversation(file.name);
        if (!convId) throw new Error(t("chat.couldNotCreateConversationForAttachment"));

        const uploaded = await uploadConversationAttachment(convId, file, CURRENT_USER_ID);
        if (!uploaded) throw new Error(t("chat.attachmentUploadFailed"));

        setAttachments((prev) => [...prev, uploaded]);
        setWorkspaceSidebarOpen(true);
        fetchWorkspaceDocuments();
        refresh();
        toast.success(t("chat.attachmentUploaded"), { description: uploaded.original_name });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : t("chat.attachmentUploadFailed");
        toast.error(t("chat.attachmentUploadFailed"), { description: errorMessage });
      } finally {
        setUploadingAttachment(false);
      }
    },
    [ensureConversation, loading, refresh, uploadingAttachment, fetchWorkspaceDocuments, t],
  );

  const handleAttachmentDelete = useCallback(
    async (attachmentId: string) => {
      if (!activeId) return;
      const target = attachments.find((attachment) => attachment.id === attachmentId);
      const deleted = await deleteConversationAttachment(activeId, attachmentId);
      if (!deleted) {
        toast.error(t("chat.attachmentDeleteFailed"));
        return;
      }

      setAttachments((prev) => prev.filter((attachment) => attachment.id !== attachmentId));
      toast.success(t("chat.attachmentRemoved"), { description: target?.original_name || attachmentId });
    },
    [activeId, attachments, t],
  );

  const handleAttachmentPillClick = useCallback((attachment: ConversationAttachment) => {
    const ref = `"${attachment.original_name}" (id: ${attachment.id})`;
    const el = textareaRef.current;
    if (!el) { setInput((prev) => prev + ref); return; }
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? el.value.length;
    const next = el.value.slice(0, start) + ref + el.value.slice(end);
    setInput(next);
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(start + ref.length, start + ref.length);
    });
  }, []);

  function handleSend() {
    if (!input.trim()) return;
    const userMessage = input;
    setInput("");
    setTimeout(() => autoResize(), 0);
    // Busy → queue the follow-up (replaces any existing slot); the provider
    // auto-fires it when the current inference completes normally.
    if (isStreaming || loading) {
      enqueueMessage(userMessage, buildSendOptions());
    } else {
      sendMessage(userMessage, buildSendOptions());
    }
  }

  // ── Regenerate: resend the last user message ───────────────────────

  const handleRegenerate = useCallback(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        sendMessage(messages[i].content, { ...buildSendOptions(), replaceFromIndex: i });
        return;
      }
    }
  }, [messages, sendMessage, buildSendOptions]);

  // ── Edit user message & resend ─────────────────────────────────────

  const handleEditAndResend = useCallback(
    (index: number, newContent: string) => {
      sendMessage(newContent, { ...buildSendOptions(), replaceFromIndex: index });
    },
    [sendMessage, buildSendOptions],
  );

  // ── Feedback handler ───────────────────────────────────────────────

  const handleFeedback = useCallback(
    async (index: number, value: "up" | "down") => {
      const msg = messages[index];
      if (!msg || !activeId) return;
      const newFeedback = msg.feedback === value ? undefined : value;

      // Optimistic UI update
      setMessages((prev) =>
        prev.map((m, i) =>
          i === index ? { ...m, feedback: newFeedback } : m,
        ),
      );

      // Persist if we have a persisted ID
      if (msg.persistedId) {
        await setMessageFeedback(
          activeId,
          msg.persistedId,
          newFeedback ?? null,
        );
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
        const data = await res.json() as { status: string; estimated_tokens?: number };
        if (data.status === "ok" && (data.estimated_tokens ?? 0) > 0) {
          setContextTokens(data.estimated_tokens!);
          setContextMenuOpen(false);
          toast.success("Context compacted");
        } else if (data.status === "skipped") {
          toast.info("Nothing to compact — context is already small");
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

  const prevIsStreamingRef = useRef(false);
  useEffect(() => {
    if (prevIsStreamingRef.current && !isStreaming) {
      textareaRef.current?.focus();
    }
    prevIsStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    const onTabKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      if (document.activeElement?.closest('[role="dialog"]')) return;
      e.preventDefault();
      textareaRef.current?.focus();
    };
    document.addEventListener("keydown", onTabKey);
    return () => document.removeEventListener("keydown", onTabKey);
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape" && isStreaming) {
      cancelStream();
      return;
    }
    // Enter sends, or queues a follow-up while a stream is in flight.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // Only one follow-up may be queued. While a message occupies the slot the
  // composer is locked so a second send can't silently replace ("swallow") it —
  // the user must send-now, remove, or edit the queued message first.
  const queueFull = queuedMessage != null;

  const _chatRole = systemConfig?.model_roles?.find((r) => r.role === "chat");
  // Only the true context window is a valid denominator. Do NOT fall back to max_tokens
  // (the output cap) — dividing prompt tokens by the output budget gives a meaningless %.
  // When unknown, the indicator shows a raw token count instead of a percentage.
  const maxContextTokens = _chatRole?.context_window_tokens ?? null;

  const userMessageCount = messages.filter((m) => m.role === "user").length;
  const assistantMessageCount = messages.filter((m) => m.role === "assistant").length;
  const _ctxPct = maxContextTokens ? Math.min(100, Math.round((contextTokens / maxContextTokens) * 100)) : 0;
  const _ctxBarColor =
    _ctxPct >= 85 ? "bg-red-500"
    : _ctxPct >= 60 ? "bg-amber-500"
    : "bg-evergreen/60";

  return (
    <>
      <div className="flex h-full min-h-0">
        {/* ─── Main chat area ─── */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          {/* ─── Chat messages ─── */}
          <div
            ref={scrollContainerRef}
            className="flex-1 overflow-y-auto p-4 md:p-6 lg:px-12 space-y-8 scroll-smooth bg-white"
            id="chat-container"
            role="log"
            aria-live="polite"
            aria-label={t("sidebar.chatHistory")}
          >
        {messages.length === 0 && !loading ? (
          <div className="flex flex-col items-center justify-center h-full text-evergreen/40">
            <Icon name="smart_toy" size={48} className="mb-4 opacity-50" />
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
                  onEdit={handleEditAndResend}
                  loading={loading}
                />
              ) : (
                <AssistantMessage
                  key={i}
                  message={msg}
                  index={i}
                  isLatest={i === lastAssistantIndex}
                  onRegenerate={handleRegenerate}
                  onFeedback={handleFeedback}
                  loading={loading}
                />
              ),
            )}
          </>
        )}

        {/* ─── Streaming / Typing indicator ─── */}
        {(isStreaming || (loading && !isStreaming)) && (
          <div className="flex gap-4 max-w-5xl mx-auto">
            <div
              className="w-8 h-8 rounded-full bg-evergreen flex items-center justify-center shrink-0 mt-1"
              aria-hidden="true"
            >
              <Icon name="smart_toy" size={18} className="text-white" />
            </div>
            <div className="flex flex-col gap-1 items-start w-full max-w-[85%]">
              <div className="flex items-center gap-2 ml-1">
                <span className="text-sm font-bold text-evergreen">
                  {t("chat.aiAgent")}
                </span>
              </div>

              {/* Node / tool status indicator */}
              {(nodeStatus || toolCallStatus?.status === "start") && (
                <div className="text-xs text-evergreen/50 ml-1 flex items-center gap-1.5 animate-pulse mb-1">
                  <Icon
                    name={
                      nodeStatus?.includes("knowledge") ? "search"
                      : nodeStatus?.includes("memor") ? "psychology"
                      : toolCallStatus?.name?.includes("search") || toolCallStatus?.name?.includes("web") ? "travel_explore"
                      : toolCallStatus?.name?.includes("calc") ? "calculate"
                      : toolCallStatus?.name?.includes("date") || toolCallStatus?.name?.includes("time") ? "schedule"
                      : toolCallStatus?.name?.includes("file") || toolCallStatus?.name?.includes("workspace") ? "edit_document"
                      : toolCallStatus?.name?.includes("webpage") || toolCallStatus?.name?.includes("extract") ? "open_in_browser"
                      : "settings"
                    }
                    size={13}
                    className="shrink-0"
                  />
                  <span>{nodeStatus ?? toolCallStatus?.label}</span>
                </div>
              )}

              {(thinkingContent || isThinking) && (
                <ReasoningBox content={thinkingContent} isStreaming={isThinking} />
              )}

              {isStreaming && streamingContent ? (
                /* Live streaming text with cursor */
                <div
                  className="text-evergreen text-base leading-relaxed px-1"
                  aria-live="polite"
                  aria-busy="true"
                >
                  <MarkdownMessage content={streamingContent} />
                  <span
                    className="inline-block w-[0.55em] h-[1.15em] bg-evergreen/70 ml-0.5 align-middle rounded-[1px] animate-cursor-blink"
                    aria-hidden="true"
                  />
                </div>
              ) : (
                /* Fallback three-dot loader (before first token arrives) */
                <div
                  className="px-4 py-3 rounded-2xl rounded-tl-sm bg-light-cyan/30 border border-light-cyan/50 flex items-center gap-1.5"
                  role="status"
                  aria-label={t("chat.aiThinking")}
                >
                  <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
                  <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
                  <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── Queued follow-up (pending bubble) ─── */}
        {queuedMessage && (
          <QueuedMessageBubble
            text={queuedMessage.text}
            idle={!isStreaming && !loading}
            onDiscard={clearQueue}
            onSendNow={() => {
              const q = queuedMessage;
              clearQueue();
              sendMessage(q.text, q.opts);
            }}
            onEdit={() => {
              // Only reclaim into the composer when it's empty, so an
              // in-progress follow-up is never clobbered.
              if (input.trim()) return;
              setInput(queuedMessage.text);
              clearQueue();
              setTimeout(() => {
                textareaRef.current?.focus();
                autoResize();
              }, 0);
            }}
          />
        )}

        {/* Scroll-to-bottom floating button — sticks to visible bottom when user scrolls up */}
        <div className="sticky bottom-4 z-10 flex justify-center pointer-events-none">
          <div className="pointer-events-auto">
            <ScrollToBottomButton
              visible={!isAtBottom}
              unreadCount={unreadCount}
              onClick={() => scrollToBottom("smooth")}
            />
          </div>
        </div>

        <div className="h-12" ref={messagesEndRef} />
      </div>

      {/* ═══════════ COMPOSER ═══════════ */}
      <div className="p-4 md:px-6 lg:px-12 md:pb-8 bg-white shrink-0 border-t border-gray-100">
        <div className="max-w-5xl mx-auto">

          {/* Attachment chips */}
          {(attachments.length > 0 || uploadingAttachment) && (
            <div className="flex flex-wrap gap-2 mb-3 px-1">
              {attachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className="inline-flex items-center gap-1 rounded-full border px-3 py-1.5 text-xs transition-colors bg-light-cyan/20 border-pacific-blue/25 text-evergreen"
                >
                  <button
                    type="button"
                    onClick={() => handleAttachmentPillClick(attachment)}
                    className="inline-flex items-center gap-1 cursor-pointer"
                    title={t("chat.insertReference")}
                  >
                    <Icon name="description" size={14} className="text-pacific-blue" />
                    <span className="font-medium">{attachment.original_name}</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleAttachmentDelete(attachment.id)}
                    className="rounded-full p-0.5 hover:bg-black/5 cursor-pointer"
                    aria-label={`${t("chat.deleteAttachment")} ${attachment.original_name}`}
                    title={t("chat.deleteAttachment")}
                  >
                    <Icon name="close" size={14} className="text-evergreen/45" />
                  </button>
                </div>
              ))}
              {uploadingAttachment && (
                <div className="inline-flex items-center gap-2 rounded-full border border-gray-300 bg-white px-3 py-1.5 text-xs text-evergreen/55">
                  <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
                  <span>{t("chat.uploadingAttachment")}</span>
                </div>
              )}
            </div>
          )}

          {/* Composer box */}
          <div className="flex flex-col rounded-xl border border-gray-200 bg-white shadow-sm transition-all">

            {/* Textarea */}
            <label htmlFor="message-input" className="sr-only">{t("chat.message")}</label>
            <textarea
              id="message-input"
              ref={textareaRef}
              value={input}
              onChange={(e) => { setInput(e.target.value); autoResize(); }}
              onKeyDown={handleKeyDown}
              disabled={queueFull}
              className="w-full bg-transparent border-0 outline-none focus:ring-0 resize-none text-evergreen placeholder-gray-400 px-4 pt-3 pb-2 text-base disabled:opacity-60 disabled:cursor-not-allowed"
              placeholder={queueFull ? t("chat.queuedSlotFull") : isStreaming ? t("chat.queuedHint") : t("chat.typeYourMessage")}
              aria-label={t("chat.typeYourMessage")}
              rows={2}
            />

            {/* Bottom toolbar */}
            <div className="flex items-center gap-1 px-3 py-2 border-t border-gray-100">

              {/* Left group: +, tools, RAG */}
              <div className="flex items-center gap-0.5">

                {/* + button → attach menu */}
                <div className="relative">
                  <button
                    type="button"
                    disabled={loading || uploadingAttachment || isStreaming}
                    onClick={() => setAttachMenuOpen((v) => !v)}
                    className="p-1.5 rounded-lg text-evergreen/50 hover:text-evergreen hover:bg-gray-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    aria-label={t("chat.addAttachment")}
                  >
                    <Icon name="add" size={20} />
                  </button>
                  {attachMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setAttachMenuOpen(false)} />
                      <div className="absolute bottom-full left-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-1 min-w-40 z-20">
                        <button
                          type="button"
                          onClick={() => { fileInputRef.current?.click(); setAttachMenuOpen(false); }}
                          className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-evergreen hover:bg-gray-50 transition-colors"
                        >
                          <Icon name="description" size={16} className="text-evergreen/60" />
                          {t("chat.addFile")}
                        </button>
                        <button
                          type="button"
                          onClick={() => { imageInputRef.current?.click(); setAttachMenuOpen(false); }}
                          className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-evergreen hover:bg-gray-50 transition-colors"
                        >
                          <Icon name="image" size={16} className="text-evergreen/60" />
                          {t("chat.addImage")}
                        </button>
                      </div>
                    </>
                  )}
                </div>

                {/* Tools button */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setToolsMenuOpen((v) => !v)}
                    className="p-1.5 rounded-lg text-evergreen/50 hover:text-evergreen hover:bg-gray-100 transition-colors"
                    aria-label="Tools"
                  >
                    <Icon name="build" size={20} />
                  </button>
                  {toolsMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setToolsMenuOpen(false)} />
                      <div className="absolute bottom-full left-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-2 min-w-50 z-20">
                        <p className="px-3 pb-1.5 text-[11px] font-semibold text-evergreen/40 uppercase tracking-wide">Tools</p>
                        {(systemConfig?.available_tools ?? FALLBACK_TOOLS).map((tool) => {
                          const enabled = toolToggles[tool.id] !== false;
                          return (
                            <button
                              key={tool.id}
                              type="button"
                              onClick={() => handleToolToggle(tool.id)}
                              className="w-full flex items-center justify-between gap-3 px-3 py-2 text-sm text-evergreen hover:bg-gray-50 transition-colors"
                            >
                              <span>{tool.label}</span>
                              <span className={`shrink-0 text-[10px] font-bold tracking-wide px-2 py-0.5 rounded-full transition-colors ${enabled ? "bg-evergreen text-white" : "bg-gray-100 text-gray-400"}`}>
                                {enabled ? "ON" : "OFF"}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </>
                  )}
                </div>

                {/* RAG toggle */}
                <button
                  type="button"
                  onClick={handleRagToggle}
                  disabled={isStreaming}
                  title={ragEnabled ? t("chat.knowledgeBaseOn") : t("chat.knowledgeBaseOff")}
                  className={`p-1.5 rounded-lg transition-colors disabled:opacity-40 ${
                    ragEnabled
                      ? "text-pacific-blue hover:bg-pacific-blue/10"
                      : "text-evergreen/30 hover:text-evergreen/60 hover:bg-gray-100"
                  }`}
                >
                  <Icon name="database" size={20} />
                </button>
              </div>

              {/* Spacer */}
              <div className="flex-1" />

              {/* Right group: model, mic, send */}
              <div className="flex items-center gap-1.5">

                {/* Context usage ring + overlay */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setContextMenuOpen((v) => !v)}
                    className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                    aria-label="Context window details"
                  >
                    <ContextRingIndicator
                      contextTokens={contextTokens}
                      maxContextTokens={maxContextTokens}
                    />
                  </button>

                  {contextMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setContextMenuOpen(false)} />
                      <div className="absolute bottom-full right-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-2 min-w-60 z-20">
                        <p className="px-3 pb-1.5 text-[11px] font-semibold text-evergreen/40 uppercase tracking-wide">
                          Context Window
                        </p>

                        {/* Usage bar + numbers */}
                        <div className="px-3 pb-2">
                          <div className="flex items-center justify-between text-xs text-evergreen mb-1">
                            <span>{contextTokens.toLocaleString()} tokens</span>
                            {maxContextTokens && <span className="text-evergreen/50">{_ctxPct}%</span>}
                          </div>
                          {maxContextTokens ? (
                            <>
                              <div className="h-1 rounded-full bg-gray-100 overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all ${_ctxBarColor}`}
                                  style={{ width: `${_ctxPct}%` }}
                                />
                              </div>
                              <p className="mt-1 text-[10px] text-evergreen/40">of {maxContextTokens.toLocaleString()} max</p>
                            </>
                          ) : (
                            <p className="mt-1 text-[10px] text-evergreen/40">context window size unknown</p>
                          )}
                        </div>

                        <div className="border-t border-gray-100 my-1" />

                        {/* Message counts */}
                        <div className="px-3 py-1 space-y-0.5">
                          <p className="text-[10px] font-semibold text-evergreen/40 uppercase tracking-wide mb-1">Messages</p>
                          <CtxRow label="User" value={userMessageCount} />
                          <CtxRow label="Assistant" value={assistantMessageCount} />
                        </div>

                        <div className="border-t border-gray-100 my-1" />

                        {/* Compact button */}
                        <div className="px-2 pt-1">
                          <button
                            type="button"
                            disabled={isStreaming || isCompacting || !activeId || contextTokens === 0}
                            onClick={handleCompactContext}
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-evergreen rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            <Icon name="compress" size={16} className="text-evergreen/60" />
                            {isCompacting ? "Compacting…" : "Compact Context"}
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {/* Model selector */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setModelMenuOpen((v) => !v)}
                    className="flex items-center gap-0.5 rounded-lg px-2.5 py-2 text-xs text-evergreen/60 hover:text-evergreen hover:bg-gray-100 transition-colors max-w-35"
                    aria-label="Select model"
                  >
                    <span className="truncate">{formatModelName(chatModel, systemConfig?.default_model)}</span>
                    <Icon name="keyboard_arrow_down" size={14} className="shrink-0" />
                  </button>
                  {modelMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setModelMenuOpen(false)} />
                      <div className="absolute bottom-full right-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-1 min-w-55 max-h-52 overflow-y-auto z-20">
                        <p className="px-3 pb-1.5 text-[11px] font-semibold text-evergreen/40 uppercase tracking-wide">Chat model</p>
                        {availableChatModels.map((model) => (
                          <button
                            key={model}
                            type="button"
                            onClick={() => handleModelChange(model)}
                            className={`w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors hover:bg-gray-50 ${chatModel === model ? "text-pacific-blue font-bold" : "text-evergreen"}`}
                          >
                            {chatModel === model && <Icon name="check" size={14} className="shrink-0" />}
                            <span className="truncate">{formatModelName(model)}</span>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>

                {/* Microphone — inactive placeholder */}
                <button
                  type="button"
                  disabled
                  aria-label="Voice input (not available)"
                  className="p-1.5 rounded-lg text-evergreen/25 cursor-not-allowed"
                >
                  <Icon name="mic" size={20} />
                </button>

                {/* Send / Cancel — while busy, Stop stays reachable and a Send
                    (queue) button appears once the user has typed a follow-up. */}
                {isStreaming || loading ? (
                  <>
                    <button
                      type="button"
                      onClick={cancelStream}
                      aria-label={t("chat.stopGenerating") ?? "Stop generating"}
                      className="w-8 h-8 flex items-center justify-center rounded-lg bg-burnt-tangerine hover:bg-burnt-tangerine/85 text-white transition-colors"
                    >
                      <Icon name="stop_circle" size={20} />
                    </button>
                    {input.trim() && (
                      <button
                        type="button"
                        onClick={handleSend}
                        aria-label={t("chat.queueMessage")}
                        title={t("chat.queueMessage")}
                        className="w-8 h-8 flex items-center justify-center rounded-lg bg-pacific-blue hover:bg-pacific-blue/85 text-white transition-colors"
                      >
                        <Icon name="arrow_upward" size={20} />
                      </button>
                    )}
                  </>
                ) : (
                  <button
                    type="button"
                    onClick={handleSend}
                    disabled={!input.trim()}
                    aria-label={t("chat.sendMessage")}
                    className="w-8 h-8 flex items-center justify-center rounded-lg bg-burnt-tangerine hover:bg-burnt-tangerine/85 text-white disabled:opacity-30 transition-colors"
                  >
                    <Icon name="arrow_upward" size={20} />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,text/plain,text/markdown"
            className="sr-only"
            onChange={handleAttachmentUpload}
          />
          <input
            ref={imageInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.gif,.webp,image/jpeg,image/png,image/gif,image/webp"
            className="sr-only"
            onChange={handleAttachmentUpload}
          />
        </div>
      </div>
      </div>

      {/* ─── Workspace sidebar ─── */}
      <WorkspaceSidebar
        isOpen={workspaceSidebarOpen}
        onToggle={() => setWorkspaceSidebarOpen(!workspaceSidebarOpen)}
        conversationId={activeId}
        documents={documents}
        isLoading={loading}
        onDocumentsRefresh={fetchWorkspaceDocuments}
        onRetryDocuments={fetchWorkspaceDocuments}
        documentsLoading={documentsLoading}
        documentsError={documentsError}
        onEnsureConversation={() => ensureConversation()}
        writebackSavedDocId={writebackSavedDocId}
        onActiveDocumentChange={setActiveWorkspaceDocId}
        answerMetrics={latestAnswerMetrics}
        onAttachmentUploaded={(attachment) => {
          setAttachments((prev) => {
            if (prev.some((item) => item.id === attachment.id)) return prev;
            return [...prev, attachment];
          });
          setWorkspaceSidebarOpen(true);
        }}
      />
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   Assistant Message — with metrics, feedback, copy, regenerate
   ═══════════════════════════════════════════════════════════════════════ */

function AssistantMessage({
  message,
  index,
  isLatest = false,
  onRegenerate,
  onFeedback,
  loading,
}: {
  message: Message;
  index: number;
  isLatest?: boolean;
  onRegenerate: () => void;
  onFeedback: (index: number, value: "up" | "down") => void;
  loading: boolean;
}) {
  const { t } = useI18n();
  return (
    <div className="msg-row flex gap-4 max-w-5xl mx-auto">
      <div
        className="w-8 h-8 rounded-full bg-evergreen flex items-center justify-center shrink-0 mt-1"
        aria-hidden="true"
      >
        <Icon name="smart_toy" size={18} className="text-white" />
      </div>
      <div className="flex flex-col gap-1 items-start w-full max-w-[85%]">
        {/* Name + timestamp */}
        <div className="flex items-center gap-2 ml-1">
          <span className="text-sm font-bold text-evergreen">{t("chat.aiAgent")}</span>
          {message.timestamp && (
            <span className="msg-timestamp text-xs text-evergreen/40 font-mono">
              {message.timestamp}
            </span>
          )}
        </div>

        {/* Reasoning chain (collapsed by default for completed messages) */}
        {message.thinking && (
          <ReasoningBox content={message.thinking} isStreaming={false} />
        )}

        {/* Message text */}
        <div className="text-evergreen text-base leading-relaxed px-1">
          <MarkdownMessage content={message.content} sources={message.metrics?.sources} />
        </div>

        {/* Map widget — rendered when map_tool was used */}
        {message.metrics?.map_data && (
          <div className="mt-2 px-1 w-full">
            <MapWidget data={message.metrics.map_data} />
          </div>
        )}

        {/* Source badges — only show RAG sources when docs were actually injected */}
        <SourceBadges
          sources={message.metrics?.sources?.filter(
            (s) => s.type !== "rag" || (message.metrics?.rag_doc_count ?? 0) > 0,
          )}
        />

        {/* Attachment context badges */}
        <AttachmentUsedBadges attachments={message.metrics?.attachments_used} />

        {/* Workspace document context badges */}
        <DocumentUsedBadges documents={message.metrics?.documents_used} />

        {/* Latest-answer evidence summary — glanceable bridge to the workspace tabs */}
        {isLatest && <EvidenceChips metrics={message.metrics} />}

        {/* Metrics panel + feedback row */}
        <div className="w-full flex flex-col gap-1">
          <MetricsPanel
            metrics={message.metrics}
            messageContent={message.content}
            onRegenerate={loading ? undefined : onRegenerate}
          />

          {/* Feedback buttons */}
          <div className="flex items-center gap-1 ml-1">
            <button
              onClick={() => onFeedback(index, "up")}
              disabled={loading}
              className={`p-1 rounded transition-colors cursor-pointer
                ${
                  message.feedback === "up"
                    ? "text-pacific-blue bg-pacific-blue/10"
                    : "text-evergreen/25 hover:text-pacific-blue hover:bg-pacific-blue/10"
                } disabled:opacity-40`}
              aria-label="Thumbs up"
              title={t("chat.feedbackSaved")}
            >
              <Icon name="thumb_up" size={15} />
            </button>
            <button
              onClick={() => onFeedback(index, "down")}
              disabled={loading}
              className={`p-1 rounded transition-colors cursor-pointer
                ${
                  message.feedback === "down"
                    ? "text-burnt-tangerine bg-burnt-tangerine/10"
                    : "text-evergreen/25 hover:text-burnt-tangerine hover:bg-burnt-tangerine/10"
                } disabled:opacity-40`}
              aria-label="Thumbs down"
              title="Poor response"
            >
              <Icon name="thumb_down" size={15} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   User Message — with edit & copy actions
   ═══════════════════════════════════════════════════════════════════════ */

function UserMessage({
  message,
  index,
  onEdit,
  loading,
}: {
  message: Message;
  index: number;
  onEdit: (index: number, newContent: string) => void;
  loading: boolean;
}) {
  const { t } = useI18n();
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(message.content);
  const [copied, setCopied] = useState(false);
  const editRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing && editRef.current) {
      editRef.current.focus();
      editRef.current.setSelectionRange(
        editRef.current.value.length,
        editRef.current.value.length,
      );
      editRef.current.style.height = "auto";
      editRef.current.style.height = editRef.current.scrollHeight + "px";
    }
  }, [editing]);

  const handleCopy = useCallback(() => {
    navigator.clipboard?.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [message.content]);

  const handleSubmitEdit = () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== message.content) {
      onEdit(index, trimmed);
    }
    setEditing(false);
  };

  return (
    <div className="msg-row flex gap-4 max-w-5xl mx-auto flex-row-reverse">
      <div
        className="w-8 h-8 rounded-full bg-linear-to-tr from-pacific-blue to-light-cyan
                    flex items-center justify-center shrink-0"
        aria-hidden="true"
      >
        <Icon name="person" size={18} className="text-white" />
      </div>
      <div className="flex flex-col gap-1 items-end max-w-[85%]">
        <div className="flex items-center gap-2 mr-1">
          {message.timestamp && (
            <span className="msg-timestamp text-xs text-evergreen/40 font-mono">
              {message.timestamp}
            </span>
          )}
          <span className="text-sm font-bold text-pacific-blue">You</span>
        </div>

        {editing ? (
          /* ── Edit mode ── */
          <div className="w-full min-w-75">
            <textarea
              ref={editRef}
              value={editValue}
              onChange={(e) => {
                setEditValue(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = e.target.scrollHeight + "px";
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmitEdit();
                }
                if (e.key === "Escape") {
                  setEditValue(message.content);
                  setEditing(false);
                }
              }}
              className="w-full rounded-xl border border-pacific-blue/40 bg-pacific-blue/5 p-4
                         text-evergreen text-base leading-relaxed resize-none focus:outline-none
                         focus:border-pacific-blue"
              rows={1}
            />
            <div className="flex items-center gap-2 mt-1.5 justify-end">
              <button
                onClick={() => {
                  setEditValue(message.content);
                  setEditing(false);
                }}
                className="text-xs text-evergreen/50 hover:text-evergreen px-2 py-1 rounded
                           hover:bg-gray-100 transition-colors cursor-pointer"
              >
                {t("common.cancel")}
              </button>
              <button
                onClick={handleSubmitEdit}
                className="text-xs text-white bg-pacific-blue hover:bg-pacific-blue/80 px-3 py-1
                           rounded font-medium transition-colors cursor-pointer"
              >
                {t("workspace.saveChanges")}
              </button>
            </div>
          </div>
        ) : (
          /* ── Normal view ── */
          <>
            <div className="bg-pacific-blue/10 p-5 rounded-2xl rounded-tr-sm text-evergreen text-base leading-relaxed border border-pacific-blue/20">
              <p className="whitespace-pre-wrap m-0">{message.content}</p>
            </div>

            {/* User message action bar */}
            <div className="flex items-center gap-0.5 mr-1 mt-0.5">
              <button
                onClick={handleCopy}
                className={`p-1 rounded transition-colors cursor-pointer ${
                  copied
                    ? "text-pacific-blue"
                    : "text-evergreen/25 hover:text-pacific-blue hover:bg-pacific-blue/10"
                }`}
                aria-label={t("chat.copyMessage")}
                title={t("chat.copyMessage")}
              >
                <Icon name={copied ? "check" : "content_copy"} size={14} />
              </button>
              <button
                onClick={() => {
                  setEditValue(message.content);
                  setEditing(true);
                }}
                disabled={loading}
                className="p-1 rounded text-evergreen/25 hover:text-pacific-blue hover:bg-pacific-blue/10
                           transition-colors disabled:opacity-40 cursor-pointer"
                aria-label="Edit message"
                title={t("workspace.edit")}
              >
                <Icon name="edit" size={14} />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   Queued Message — dimmed pending user bubble for a follow-up typed while the
   model is still streaming. Not part of the `messages` array (like the live
   streaming indicator), so it can't disturb message ordering or the
   persisted-id backfill. Auto-fires on normal completion; on a manual Stop /
   error it stays put (idle) with explicit Send-now / discard controls.
   ═══════════════════════════════════════════════════════════════════════ */

function QueuedMessageBubble({
  text,
  idle,
  onDiscard,
  onSendNow,
  onEdit,
}: {
  text: string;
  idle: boolean;
  onDiscard: () => void;
  onSendNow: () => void;
  onEdit: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="msg-row flex gap-4 max-w-5xl mx-auto flex-row-reverse opacity-60">
      <div
        className="w-8 h-8 rounded-full bg-linear-to-tr from-pacific-blue to-light-cyan
                    flex items-center justify-center shrink-0"
        aria-hidden="true"
      >
        <Icon name="person" size={18} className="text-white" />
      </div>
      <div className="flex flex-col gap-1 items-end max-w-[85%]">
        {/* Queued tag */}
        <div className="flex items-center gap-1.5 mr-1 text-xs text-evergreen/45">
          <Icon name="schedule" size={13} className="animate-pulse" />
          <span className="font-medium">{t("chat.queued")}</span>
        </div>

        {/* Bubble — click to reclaim into the composer for editing */}
        <button
          type="button"
          onClick={onEdit}
          title={t("chat.editQueued")}
          aria-label={t("chat.editQueued")}
          className="text-left bg-pacific-blue/10 p-5 rounded-2xl rounded-tr-sm text-evergreen
                     text-base leading-relaxed border border-dashed border-pacific-blue/30
                     hover:border-pacific-blue/50 transition-colors cursor-text"
        >
          <p className="whitespace-pre-wrap m-0">{text}</p>
        </button>

        {/* Controls */}
        <div className="flex items-center gap-0.5 mr-1 mt-0.5">
          {idle && (
            <button
              type="button"
              onClick={onSendNow}
              className="p-1 rounded text-evergreen/35 hover:text-pacific-blue hover:bg-pacific-blue/10
                         transition-colors cursor-pointer"
              aria-label={t("chat.sendQueuedNow")}
              title={t("chat.sendQueuedNow")}
            >
              <Icon name="send" size={14} />
            </button>
          )}
          <button
            type="button"
            onClick={onDiscard}
            className="p-1 rounded text-evergreen/35 hover:text-burnt-tangerine hover:bg-burnt-tangerine/10
                       transition-colors cursor-pointer"
            aria-label={t("chat.cancelQueued")}
            title={t("chat.cancelQueued")}
          >
            <Icon name="close" size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
