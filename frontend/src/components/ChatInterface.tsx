"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { ArrowUp, BookOpen, Bot, Brain, Calculator, Check, Clock, Compass, Copy, Database, ExternalLink, FileEdit, FileText, FolderOpen, Globe, Image as ImageIcon, Mic, Minimize2, Pencil, Plus, Search, Send, Settings, StopCircle, ThumbsDown, ThumbsUp, Wrench, X } from "lucide-react";
import { MarkdownMessage } from "./MarkdownMessage";
import { ContextRingIndicator } from "./ContextRingIndicator";
import { MetricsPanel } from "./MetricsPanel";
import { ReasoningBox } from "./ReasoningBox";
import dynamic from "next/dynamic";
const MapWidget = dynamic(() => import("./MapWidget").then((m) => m.MapWidget), { ssr: false });
import { WorkspaceSidebar, type WorkspaceDocument } from "./WorkspaceSidebar";
import { EvidenceChips } from "./workspace/EvidenceChips";
import type { WorkspaceTabId } from "./workspace/types";
import { ChatMessageShell } from "./product/ChatMessageShell";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import { Textarea } from "@/components/ui/Textarea";
import { useConversationContext } from "./LayoutShell";
import { useWorkspacePanel } from "@/context/WorkspacePanelContext";
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
  NodeTraceEntry,
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
                         bg-primary/10 text-primary border border-primary/20
                         hover:bg-primary/20 transition-colors no-underline"
          >
            <Globe size={12} />
            {src.label}
          </a>
        ) : (
          <span
            key={`${src.label}-${idx}`}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                       bg-warning/10 text-warning border border-warning/20"
          >
            <BookOpen size={12} />
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
                     bg-surface-muted text-muted-foreground border border-border"
          title={`Context from: ${att.original_name}`}
        >
          <FileText size={12} />
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
                     bg-success/10 text-success border border-success/20"
          title={`Workspace context: ${doc.filename} (v${doc.version})`}
        >
          <FolderOpen size={12} />
          {doc.filename}
          <span className="text-success/75">v{doc.version}</span>
        </span>
      ))}
    </div>
  );
}



function CtxRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums text-foreground">{value}</span>
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
  const [contextMenuOpen, setContextMenuOpen] = useState(false);
  const [isCompacting, setIsCompacting] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [hasNewMessage, setHasNewMessage] = useState(false);
  const selectedChatModel = chatModel || systemConfig?.default_model || availableChatModels[0] || "";
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
    nodeTrace,
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

  // Inspector trace: the live trace while the active answer is streaming,
  // otherwise the committed trace on the latest assistant message (the live one
  // is cleared by resetStream on completion).
  const committedNodeTrace = useMemo<NodeTraceEntry[]>(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.nodeTrace && m.nodeTrace.length > 0) return m.nodeTrace;
    }
    return [];
  }, [messages]);
  const inspectorNodeTrace = isStreaming
    ? nodeTrace
    : committedNodeTrace.length > 0
      ? committedNodeTrace
      : nodeTrace;

  // Clicking an evidence chip opens the workspace panel on the matching tab.
  const { setActiveTab: setWorkspaceTab } = useWorkspacePanel();
  const handleSelectEvidence = useCallback(
    (tab: WorkspaceTabId) => {
      setWorkspaceTab(tab);
      setWorkspaceSidebarOpen(true);
    },
    [setWorkspaceTab, setWorkspaceSidebarOpen],
  );

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
    [ensureConversation, loading, refresh, uploadingAttachment, fetchWorkspaceDocuments, setWorkspaceSidebarOpen, t],
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
    _ctxPct >= 85 ? "bg-destructive"
    : _ctxPct >= 60 ? "bg-warning"
    : "bg-primary/70";

  return (
    <>
      <div className="flex h-full min-h-0">
        {/* ─── Main chat area ─── */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          {/* ─── Chat messages ─── */}
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
                  onEdit={handleEditAndResend}
                  loading={loading}
                />
              ) : (
                <AssistantMessage
                  key={i}
                  message={msg}
                  index={i}
                  isLatest={i === lastAssistantIndex}
                  onSelectEvidence={handleSelectEvidence}
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
          <ChatMessageShell role="assistant">
              <div className="flex items-center gap-2 ml-1">
                <span className="text-sm font-bold text-foreground">
                  {t("chat.aiAgent")}
                </span>
              </div>

              {/* Node / tool status indicator */}
              {(nodeStatus || toolCallStatus?.status === "start") && (
                <div className="mb-1 ml-1 flex items-center gap-1.5 text-xs text-muted-foreground animate-pulse">
                  {nodeStatus?.includes("knowledge") ? <Search size={13} className="shrink-0" />
                  : nodeStatus?.includes("memor") ? <Brain size={13} className="shrink-0" />
                  : toolCallStatus?.name?.includes("search") || toolCallStatus?.name?.includes("web") ? <Compass size={13} className="shrink-0" />
                  : toolCallStatus?.name?.includes("calc") ? <Calculator size={13} className="shrink-0" />
                  : toolCallStatus?.name?.includes("date") || toolCallStatus?.name?.includes("time") ? <Clock size={13} className="shrink-0" />
                  : toolCallStatus?.name?.includes("file") || toolCallStatus?.name?.includes("workspace") ? <FileEdit size={13} className="shrink-0" />
                  : toolCallStatus?.name?.includes("webpage") || toolCallStatus?.name?.includes("extract") ? <ExternalLink size={13} className="shrink-0" />
                  : <Settings size={13} className="shrink-0" />}
                  <span>{nodeStatus ?? toolCallStatus?.label}</span>
                </div>
              )}

              {(thinkingContent || isThinking) && (
                <ReasoningBox content={thinkingContent} isStreaming={isThinking} />
              )}

              {isStreaming && streamingContent ? (
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
      <div className="shrink-0 border-t border-border bg-surface p-4 md:px-6 md:pb-8 lg:px-12">
        <div className="max-w-5xl mx-auto">

          {/* Attachment chips */}
          {(attachments.length > 0 || uploadingAttachment) && (
            <div className="flex flex-wrap gap-2 mb-3 px-1">
              {attachments.map((attachment) => (
                <div
                  key={attachment.id}
                  className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-muted px-3 py-1.5 text-xs text-foreground transition-colors"
                >
                  <Button
                    type="button"
                    onClick={() => handleAttachmentPillClick(attachment)}
                    variant="ghost"
                    size="sm"
                    className="inline-flex h-auto items-center gap-1 cursor-pointer rounded-full px-0 py-0 text-inherit hover:bg-transparent"
                    title={t("chat.insertReference")}
                  >
                    <FileText size={14} className="text-primary" />
                    <span className="font-medium">{attachment.original_name}</span>
                  </Button>
                  <Button
                    type="button"
                    onClick={() => handleAttachmentDelete(attachment.id)}
                    variant="ghost"
                    size="sm"
                    className="rounded-full p-0.5 hover:bg-black/5 cursor-pointer"
                    aria-label={`${t("chat.deleteAttachment")} ${attachment.original_name}`}
                    title={t("chat.deleteAttachment")}
                  >
                    <X size={14} className="text-muted-foreground" />
                  </Button>
                </div>
              ))}
              {uploadingAttachment && (
                <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5 text-xs text-muted-foreground">
                  <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
                  <span>{t("chat.uploadingAttachment")}</span>
                </div>
              )}
            </div>
          )}

          {/* Composer box */}
          <div className="flex flex-col rounded-xl border border-border bg-surface shadow-sm transition-all">

            {/* Textarea */}
            <label htmlFor="message-input" className="sr-only">{t("chat.message")}</label>
            <Textarea
              id="message-input"
              ref={textareaRef}
              value={input}
              onChange={(e) => { setInput(e.target.value); autoResize(); }}
              onKeyDown={handleKeyDown}
              disabled={queueFull}
              className="resize-none rounded-none border-0 bg-transparent px-4 pb-2 pt-3 text-base text-foreground shadow-none focus:ring-0"
              placeholder={queueFull ? t("chat.queuedSlotFull") : isStreaming ? t("chat.queuedHint") : t("chat.typeYourMessage")}
              aria-label={t("chat.typeYourMessage")}
              rows={2}
            />

            {/* Bottom toolbar */}
            <div className="flex items-center gap-1 border-t border-border px-3 py-2">

              {/* Left group: +, tools, RAG */}
              <div className="flex items-center gap-0.5">

                {/* + button → attach menu */}
                <div className="relative">
                  <Button
                    type="button"
                    disabled={loading || uploadingAttachment || isStreaming}
                    onClick={() => setAttachMenuOpen((v) => !v)}
                    variant="ghost"
                    size="sm"
                    className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-surface-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label={t("chat.addAttachment")}
                  >
                    <Plus size={20} />
                  </Button>
                  {attachMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setAttachMenuOpen(false)} />
                      <div className="absolute bottom-full left-0 z-20 mb-2 min-w-40 rounded-xl border border-border bg-surface py-1 shadow-lg">
                        <Button
                          type="button"
                          onClick={() => { fileInputRef.current?.click(); setAttachMenuOpen(false); }}
                          variant="ghost"
                          size="sm"
                          className="w-full items-center gap-2.5 px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted"
                        >
                          <FileText size={16} className="text-muted-foreground" />
                          {t("chat.addFile")}
                        </Button>
                        <Button
                          type="button"
                          onClick={() => { imageInputRef.current?.click(); setAttachMenuOpen(false); }}
                          variant="ghost"
                          size="sm"
                          className="w-full items-center gap-2.5 px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted"
                        >
                          <ImageIcon size={16} className="text-muted-foreground" />
                          {t("chat.addImage")}
                        </Button>
                      </div>
                    </>
                  )}
                </div>

                {/* Tools button */}
                <div className="relative">
                  <Button
                    type="button"
                    onClick={() => setToolsMenuOpen((v) => !v)}
                    variant="ghost"
                    size="sm"
                    className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-surface-muted hover:text-foreground"
                    aria-label="Tools"
                  >
                    <Wrench size={20} />
                  </Button>
                  {toolsMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setToolsMenuOpen(false)} />
                      <div className="absolute bottom-full left-0 z-20 mb-2 min-w-50 rounded-xl border border-border bg-surface py-2 shadow-lg">
                        <p className="px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Tools</p>
                        {(systemConfig?.available_tools ?? FALLBACK_TOOLS).map((tool) => {
                          const enabled = toolToggles[tool.id] !== false;
                          return (
                            <Button
                              key={tool.id}
                              type="button"
                              onClick={() => handleToolToggle(tool.id)}
                              variant="ghost"
                              size="sm"
                              className="w-full items-center justify-between gap-3 px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted"
                            >
                              <span>{tool.label}</span>
                              <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wide transition-colors ${enabled ? "bg-primary/15 text-primary" : "bg-surface-muted text-muted-foreground"}`}>
                                {enabled ? "ON" : "OFF"}
                              </span>
                            </Button>
                          );
                        })}
                      </div>
                    </>
                  )}
                </div>

                {/* RAG toggle */}
                <Button
                  type="button"
                  onClick={handleRagToggle}
                  disabled={isStreaming}
                  title={ragEnabled ? t("chat.knowledgeBaseOn") : t("chat.knowledgeBaseOff")}
                  variant="ghost"
                  size="sm"
                  className={`p-1.5 rounded-lg transition-colors disabled:opacity-40 ${
                    ragEnabled
                      ? "text-primary hover:bg-primary/10"
                      : "text-muted-foreground hover:bg-surface-muted hover:text-foreground"
                  }`}
                >
                  <Database size={20} />
                </Button>
              </div>

              {/* Spacer */}
              <div className="flex-1" />

              {/* Right group: model, mic, send */}
              <div className="flex items-center gap-1.5">

                {/* Context usage ring + overlay */}
                <div className="relative">
                  <Button
                    type="button"
                    onClick={() => setContextMenuOpen((v) => !v)}
                    variant="ghost"
                    size="sm"
                    className="rounded-lg p-1.5 transition-colors hover:bg-surface-muted"
                    aria-label="Context window details"
                  >
                    <ContextRingIndicator
                      contextTokens={contextTokens}
                      maxContextTokens={maxContextTokens}
                    />
                  </Button>

                  {contextMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setContextMenuOpen(false)} />
                      <div className="absolute bottom-full right-0 z-20 mb-2 min-w-60 rounded-xl border border-border bg-surface py-2 shadow-lg">
                        <p className="px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                          Context Window
                        </p>

                        {/* Usage bar + numbers */}
                        <div className="px-3 pb-2">
                          <div className="mb-1 flex items-center justify-between text-xs text-foreground">
                            <span>{contextTokens.toLocaleString()} tokens</span>
                            {maxContextTokens && <span className="text-muted-foreground">{_ctxPct}%</span>}
                          </div>
                          {maxContextTokens ? (
                            <>
                              <div className="h-1 overflow-hidden rounded-full bg-surface-muted">
                                <div
                                  className={`h-full rounded-full transition-all ${_ctxBarColor}`}
                                  style={{ width: `${_ctxPct}%` }}
                                />
                              </div>
                              <p className="mt-1 text-[10px] text-muted-foreground">of {maxContextTokens.toLocaleString()} max</p>
                            </>
                          ) : (
                            <p className="mt-1 text-[10px] text-muted-foreground">context window size unknown</p>
                          )}
                        </div>

                        <div className="my-1 border-t border-border" />

                        {/* Message counts */}
                        <div className="px-3 py-1 space-y-0.5">
                          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Messages</p>
                          <CtxRow label="User" value={userMessageCount} />
                          <CtxRow label="Assistant" value={assistantMessageCount} />
                        </div>

                        <div className="my-1 border-t border-border" />

                        {/* Compact button */}
                        <div className="px-2 pt-1">
                          <Button
                            type="button"
                            disabled={isStreaming || isCompacting || !activeId || contextTokens === 0}
                            onClick={handleCompactContext}
                            variant="ghost"
                            size="sm"
                            className="w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            <Minimize2 size={16} className="text-muted-foreground" />
                            {isCompacting ? "Compacting…" : "Compact Context"}
                          </Button>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {/* Model selector */}
                <div className="relative">
                  <Select
                    value={selectedChatModel}
                    onChange={(e) => void handleModelChange(e.target.value)}
                    aria-label="Select model"
                    className="max-w-35 rounded-lg px-2.5 py-2 text-xs text-foreground"
                  >
                    {availableChatModels.map((model) => (
                      <option key={model} value={model}>
                        {formatModelName(model)}
                      </option>
                    ))}
                  </Select>
                </div>

                {/* Microphone — inactive placeholder */}
                <Button
                  type="button"
                  disabled
                  aria-label="Voice input (not available)"
                  variant="ghost"
                  size="sm"
                  className="cursor-not-allowed rounded-lg p-1.5 text-muted-foreground"
                >
                  <Mic size={20} />
                </Button>

                {/* Send / Cancel — while busy, Stop stays reachable and a Send
                    (queue) button appears once the user has typed a follow-up. */}
                {isStreaming || loading ? (
                  <>
                    <Button
                      type="button"
                      onClick={cancelStream}
                      aria-label={t("chat.stopGenerating") ?? "Stop generating"}
                      variant="destructive"
                      size="sm"
                      className="h-8 w-8 rounded-lg p-0"
                    >
                      <StopCircle size={20} />
                    </Button>
                    {input.trim() && (
                      <Button
                        type="button"
                        onClick={handleSend}
                        aria-label={t("chat.queueMessage")}
                        title={t("chat.queueMessage")}
                        variant="primary"
                        size="sm"
                        className="h-8 w-8 rounded-lg p-0"
                      >
                        <ArrowUp size={20} />
                      </Button>
                    )}
                  </>
                ) : (
                  <Button
                    type="button"
                    onClick={handleSend}
                    disabled={!input.trim()}
                    aria-label={t("chat.sendMessage")}
                    variant="primary"
                    size="sm"
                    className="h-8 w-8 rounded-lg p-0 disabled:opacity-30"
                  >
                    <ArrowUp size={20} />
                  </Button>
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
        nodeTrace={inspectorNodeTrace}
        isStreaming={isStreaming}
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
  onSelectEvidence,
  onRegenerate,
  onFeedback,
  loading,
}: {
  message: Message;
  index: number;
  isLatest?: boolean;
  onSelectEvidence?: (tab: WorkspaceTabId) => void;
  onRegenerate: () => void;
  onFeedback: (index: number, value: "up" | "down") => void;
  loading: boolean;
}) {
  const { t } = useI18n();
  return (
    <ChatMessageShell role="assistant">
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
        {isLatest && <EvidenceChips metrics={message.metrics} onSelect={onSelectEvidence} />}

        {/* Metrics panel + feedback row */}
        <div className="w-full flex flex-col gap-1">
          <MetricsPanel
            metrics={message.metrics}
            messageContent={message.content}
            onRegenerate={loading ? undefined : onRegenerate}
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
    <ChatMessageShell role="user">
        <div className="flex items-center gap-2 mr-1">
          {message.timestamp && (
            <span className="msg-timestamp font-mono text-xs text-muted-foreground">
              {message.timestamp}
            </span>
          )}
          <span className="text-sm font-bold text-primary">You</span>
        </div>

        {editing ? (
          /* ── Edit mode ── */
          <div className="w-full min-w-75">
            <Textarea
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
              className="w-full rounded-xl border border-primary/40 bg-primary/5 p-4
                         text-foreground text-base leading-relaxed resize-none focus:outline-none
                         focus:border-primary"
              rows={1}
            />
            <div className="flex items-center gap-2 mt-1.5 justify-end">
              <Button
                type="button"
                onClick={() => {
                  setEditValue(message.content);
                  setEditing(false);
                }}
                variant="ghost"
                size="sm"
                className="rounded px-2 py-1 text-xs text-muted-foreground transition-colors cursor-pointer
                           hover:bg-surface-muted hover:text-foreground"
              >
                {t("common.cancel")}
              </Button>
              <Button
                type="button"
                onClick={handleSubmitEdit}
                variant="primary"
                size="sm"
                className="rounded px-3 py-1 text-xs font-medium transition-colors cursor-pointer"
              >
                {t("workspace.saveChanges")}
              </Button>
            </div>
          </div>
        ) : (
          /* ── Normal view ── */
          <>
            <div className="rounded-2xl rounded-tr-sm border border-primary/20 bg-primary/10 p-5 text-base leading-relaxed text-foreground">
              <p className="whitespace-pre-wrap m-0">{message.content}</p>
            </div>

            {/* User message action bar */}
            <div className="flex items-center gap-0.5 mr-1 mt-0.5">
              <Button
                onClick={handleCopy}
                type="button"
                variant="ghost"
                size="sm"
                className={`p-1 rounded transition-colors cursor-pointer ${
                  copied
                    ? "text-primary"
                    : "text-muted-foreground hover:text-primary hover:bg-primary/10"
                }`}
                aria-label={t("chat.copyMessage")}
                title={t("chat.copyMessage")}
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </Button>
              <Button
                type="button"
                onClick={() => {
                  setEditValue(message.content);
                  setEditing(true);
                }}
                disabled={loading}
                variant="ghost"
                size="sm"
                className="p-1 rounded text-muted-foreground hover:text-primary hover:bg-primary/10
                           transition-colors disabled:opacity-40 cursor-pointer"
                aria-label="Edit message"
                title={t("workspace.edit")}
              >
                <Pencil size={14} />
              </Button>
            </div>
          </>
        )}
    </ChatMessageShell>
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
    <ChatMessageShell role="user" className="opacity-60">
        {/* Queued tag */}
        <div className="mr-1 flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock size={13} className="animate-pulse" />
          <span className="font-medium">{t("chat.queued")}</span>
        </div>

        {/* Bubble — click to reclaim into the composer for editing */}
        <Button
          type="button"
          onClick={onEdit}
          title={t("chat.editQueued")}
          aria-label={t("chat.editQueued")}
          variant="ghost"
          className="text-left rounded-2xl rounded-tr-sm border border-dashed border-primary/30 bg-primary/10 p-5
                     text-base leading-relaxed text-foreground transition-colors cursor-text hover:border-primary/50"
        >
          <p className="whitespace-pre-wrap m-0">{text}</p>
        </Button>

        {/* Controls */}
        <div className="flex items-center gap-0.5 mr-1 mt-0.5">
          {idle && (
            <Button
              type="button"
              onClick={onSendNow}
              variant="ghost"
              size="sm"
              className="p-1 rounded text-muted-foreground hover:text-primary hover:bg-primary/10
                         transition-colors cursor-pointer"
              aria-label={t("chat.sendQueuedNow")}
              title={t("chat.sendQueuedNow")}
            >
              <Send size={14} />
            </Button>
          )}
          <Button
            type="button"
            onClick={onDiscard}
            variant="ghost"
            size="sm"
            className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10
                       transition-colors cursor-pointer"
            aria-label={t("chat.cancelQueued")}
            title={t("chat.cancelQueued")}
          >
            <X size={14} />
          </Button>
        </div>
    </ChatMessageShell>
  );
}
