"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Icon } from "./Icon";
import { ContextRingIndicator } from "./ContextRingIndicator";
import { MetricsPanel } from "./MetricsPanel";
import { WorkspaceSidebar, type WorkspaceDocument } from "./WorkspaceSidebar";
import { useConversationContext } from "./LayoutShell";
import { useI18n } from "@/hooks/useI18n";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { useScrollToBottom } from "@/hooks/useScrollToBottom";
import { ScrollToBottomButton } from "@/components/ScrollToBottomButton";
import {
  deleteConversationAttachment,
  fetchConversation,
  fetchConversationAttachments,
  fetchSystemConfig,
  fetchUserSettings,
  setMessageFeedback,
  updateUserSettings,
  uploadConversationAttachment,
} from "@/lib/api";
import type { SystemConfig } from "@/lib/api";
import { selectActiveAttachmentIds } from "@/lib/attachments";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type {
  ConversationAttachment,
  Message,
  PersistedMessage,
  Source,
} from "@/lib/types";

/**
 * Replace [N] footnote references with clickable markdown links using the sources array.
 * E.g. "[1]" becomes "[<sup>1</sup>](url)" if source 1 has a URL, or bold "[<sup>1</sup>]" for RAG.
 */
const FALLBACK_TOOLS = [
  { id: "web_search_mcp", label: "Web Search" },
  { id: "extract_webpage_mcp", label: "Extract Webpage" },
  { id: "datetime_tool", label: "Datetime" },
  { id: "calculator_tool", label: "Calculator" },
  { id: "file_ops_tool", label: "File Ops" },
] as const;

function formatModelName(model: string | null | undefined, fallback = "Model"): string {
  const m = model || fallback;
  const parts = m.split("/");
  return parts.length > 1 ? parts.slice(1).join("/") : m;
}

function linkFootnotes(text: string, sources?: Source[]): string {
  if (!sources || sources.length === 0) return text;
  // Build a map from index (1-based from backend) to source
  const indexMap = new Map<number, Source>();
  sources.forEach((s) => {
    if (s.index) indexMap.set(s.index, s);
  });
  // Also fall back to position-based if no index field
  if (indexMap.size === 0) {
    sources.forEach((s, i) => indexMap.set(i + 1, s));
  }

  const SUPERSCRIPT = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
  const toSup = (n: number) =>
    String(n).split("").map((d) => SUPERSCRIPT[parseInt(d)]).join("");

  // Match [N], [N, M], [N, M, O] patterns
  return text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_match, nums: string) => {
    const numbers = nums.split(",").map((n: string) => parseInt(n.trim(), 10));
    const parts = numbers.map((n) => {
      const src = indexMap.get(n);
      if (!src) return `[${n}]`;
      if (src.url) return `[${toSup(n)}](${src.url})`;
      return `**[${toSup(n)}]**`;
    });
    return parts.join(" ");
  });
}

/** Render markdown content with proper styling and footnote linking */
function MarkdownMessage({ content, sources }: { content: string; sources?: Source[] }) {
  const processed = useMemo(() => linkFootnotes(content, sources), [content, sources]);
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children, ...props }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-pacific-blue underline hover:text-pacific-blue/80 break-all"
            {...props}
          >
            {children}
          </a>
        ),
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        h1: ({ children }) => <h1 className="text-xl font-bold mb-2 mt-4 first:mt-0">{children}</h1>,
        h2: ({ children }) => <h2 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="text-base font-semibold mb-1.5 mt-3 first:mt-0">{children}</h3>,
        ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        code: ({ children, className }) => {
          const isBlock = className?.includes("language-");
          if (isBlock) {
            return (
              <code className="block bg-evergreen/5 rounded-lg p-3 text-sm font-mono overflow-x-auto my-2 border border-evergreen/10">
                {children}
              </code>
            );
          }
          return (
            <code className="bg-evergreen/5 rounded px-1.5 py-0.5 text-sm font-mono">{children}</code>
          );
        },
        pre: ({ children }) => <pre className="my-2">{children}</pre>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-3 border-pacific-blue/40 pl-3 my-2 text-evergreen/70 italic">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto my-3">
            <table className="min-w-full text-sm border border-evergreen/10 rounded">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="px-3 py-1.5 text-left font-semibold bg-light-cyan/20 border-b border-evergreen/10">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-3 py-1.5 border-b border-evergreen/5">{children}</td>
        ),
        hr: () => <hr className="my-4 border-evergreen/10" />,
      }}
    >
      {processed}
    </ReactMarkdown>
  );
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


/** Generate a concise conversation title from the first user message. */
function generateTitle(message: string): string {
  const clean = message.replace(/\s+/g, " ").trim();
  if (!clean) return "New conversation";
  if (clean.length <= 50) return clean;
  const sentenceEnd = clean.substring(0, 60).search(/[.!?]\s/);
  if (sentenceEnd > 10) return clean.substring(0, sentenceEnd + 1);
  const truncated = clean.substring(0, 50);
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace > 20) return truncated.substring(0, lastSpace) + "\u2026";
  return truncated + "\u2026";
}

/** Convert a persisted (DB) message into the local UI Message shape. */
function toUiMessage(pm: PersistedMessage, formatTime: (value: Date | string | number) => string): Message {
  return {
    role: pm.role === "system" ? "assistant" : pm.role,
    content: pm.content,
    timestamp: pm.created_at
      ? formatTime(pm.created_at)
      : undefined,
    persistedId: pm.id,
    feedback: pm.feedback ?? undefined,
    metrics: {
      output_tokens: pm.tokens_used ?? undefined,
      input_tokens: (pm.metadata?.input_tokens as number | undefined) ?? undefined,
      response_time_ms: pm.response_time_ms ?? undefined,
      model: pm.model_name ?? undefined,
      tools_executed: pm.tools_used?.map((t) => ({
        name: t.name,
        status: t.status,
      })),
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
    },
  };
}

export function ChatInterface() {
  const { t, formatTime } = useI18n();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [attachments, setAttachments] = useState<ConversationAttachment[]>([]);
  const [selectedAttachmentIds, setSelectedAttachmentIds] = useState<string[]>([]);
  const [uploadingAttachment, setUploadingAttachment] = useState(false);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
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
  const [contextTokens, setContextTokens] = useState<number>(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const startTimeRef = useRef<number>(0);
  const wasStreamingRef = useRef(false);
  const preferredModelsRef = useRef<Record<string, string | null>>({});

  const {
    streamingContent,
    isStreaming,
    streamError,
    streamWarning,
    toolCallStatus,
    nodeStatus,
    finalMetadata,
    wasCancelled,
    sendMessage: startStream,
    cancel: cancelStream,
    reset: resetStream,
  } = useStreamingChat();

  const { activeId, create, refresh, rename, activeConversation, workspaceSidebarOpen, setWorkspaceSidebarOpen } =
    useConversationContext();


  // Flag: when true the next activeId change came from create() and the
  // message-fetch useEffect should skip reloading (no persisted messages yet).
  const skipNextFetchRef = useRef(false);

  // ── Load workspace documents ─────────────────
  const fetchWorkspaceDocuments = useCallback(async () => {
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
      // Silently fail - documents are optional
      console.warn("Failed to load workspace documents:", err);
    }
  }, []);

  // Load documents on mount
  useEffect(() => {
    fetchWorkspaceDocuments();
  }, [fetchWorkspaceDocuments]);

  // ── Load messages when active conversation changes ─────────────────

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setAttachments([]);
      setSelectedAttachmentIds([]);
      setWorkspaceSidebarOpen(false);
      setContextTokens(0);
      return;
    }
    // When a conversation was just created the DB has no messages yet.
    // Skip the fetch so the optimistic user message is not overwritten.
    if (skipNextFetchRef.current) {
      skipNextFetchRef.current = false;
      return;
    }
    let cancelled = false;
    (async () => {
      const detail = await fetchConversation(activeId);
      if (cancelled || !detail) return;
      const loadedMessages = detail.messages.map((msg) => toUiMessage(msg, formatTime));
      // Restore context ring to the high-water mark from persisted input_tokens.
      const hwm = loadedMessages
        .filter((m) => m.role === "assistant")
        .reduce((max, m) => Math.max(max, m.metrics?.input_tokens ?? 0), 0);
      setContextTokens(hwm);
      setMessages(loadedMessages);
      // New chats should start with a collapsed workspace unless explicitly opened.
      if (detail.messages.length === 0) {
        setWorkspaceSidebarOpen(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId, formatTime]);

  useEffect(() => {
    if (!activeId) {
      setAttachments([]);
      setSelectedAttachmentIds([]);
      return;
    }
    let cancelled = false;
    (async () => {
      const data = await fetchConversationAttachments(activeId);
      if (cancelled) return;
      setAttachments(data);
      // Auto-select all active attachments for this conversation.
      // Any previously selected ID no longer in the list (deleted/expired) is
      // naturally dropped because we derive the new selection from `data`.
      setSelectedAttachmentIds(selectActiveAttachmentIds(data));
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
  }, [messages, loading, isStreaming, streamingContent, shouldAutoScroll]); // eslint-disable-line react-hooks/exhaustive-deps

  // Commit completed streaming message to messages list
  useEffect(() => {
    const was = wasStreamingRef.current;
    wasStreamingRef.current = isStreaming;
    if (was && !isStreaming && (streamingContent || wasCancelled)) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant" as const,
          content: wasCancelled
            ? streamingContent + (streamingContent ? "\n\n*(generation stopped)*" : "*(generation stopped)*")
            : streamingContent,
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
              }
            : undefined,
        },
      ]);
      if (finalMetadata?.workspace_document_writeback?.status === "saved") {
        const wb = finalMetadata.workspace_document_writeback;
        fetchWorkspaceDocuments();
        setWritebackSavedDocId(wb.document_id ?? null);
        setTimeout(() => setWritebackSavedDocId(null), 200);
        toast.success(t("chat.workspaceDocumentSaved"), {
          description: `${wb.filename} updated to v${wb.version}`,
        });
      }
      resetStream();

      // After streaming, fetch the conversation to get DB message IDs so that
      // feedback (thumbs up/down) can be persisted. _run_persistence on the
      // backend completes before [DONE] is emitted, so the rows are ready.
      if (activeId) {
        fetchConversation(activeId).then((detail) => {
          if (!detail) return;
          setMessages((prev) => {
            const dbMsgs = detail.messages;
            let dbIdx = 0;
            return prev.map((msg) => {
              if (msg.persistedId != null) {
                // Already has an ID — advance the DB cursor past the matching row.
                while (dbIdx < dbMsgs.length && dbMsgs[dbIdx].id !== msg.persistedId) dbIdx++;
                dbIdx++;
                return msg;
              }
              // Find the next DB message with the same role.
              while (dbIdx < dbMsgs.length && dbMsgs[dbIdx].role !== msg.role) dbIdx++;
              const dbMsg = dbMsgs[dbIdx];
              dbIdx++;
              return dbMsg ? { ...msg, persistedId: dbMsg.id } : msg;
            });
          });
        });
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming]);

  // Update context token count — use high-water mark so the ring never goes backwards.
  // Per-turn prompt size fluctuates (RAG results and tool outputs vary) but conversation
  // history only grows, so the peak value best reflects cumulative context growth.
  // contextTokens is already reset to 0 when the active conversation changes.
  useEffect(() => {
    const tokens = finalMetadata?.input_tokens;
    if (!isStreaming && tokens) {
      setContextTokens((prev) => Math.max(prev, tokens));
    }
  }, [isStreaming, finalMetadata]);

  // Surface stream errors as toasts
  useEffect(() => {
    if (streamError) {
      toast.error(t("chat.messageFailed"), { description: streamError });
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamError]);

  // Surface model validation warnings as toasts
  useEffect(() => {
    if (streamWarning) {
      toast.warning(streamWarning);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamWarning]);

  // Cancel any in-flight stream on unmount
  useEffect(() => () => cancelStream(), [cancelStream]);

  // Load user settings on mount: RAG config, tool toggles, chat model
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

  // ── Core send logic (reused by send, regenerate, edit) ─────────────

  const sendMessage = useCallback(
    async (userMessage: string, replaceFromIndex?: number) => {
      let convId = activeId;
      const isFirstMessage = messages.length === 0 || replaceFromIndex === 0;
      if (!convId) {
        convId = await ensureConversation(userMessage);
        if (!convId) return;
      }

      if (replaceFromIndex != null) {
        setMessages((prev) => [
          ...prev.slice(0, replaceFromIndex),
          {
            role: "user",
            content: userMessage,
            timestamp: formatTime(new Date()),
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "user",
            content: userMessage,
            timestamp: formatTime(new Date()),
          },
        ]);
      }
      setLoading(true);
      startTimeRef.current = Date.now();

      await startStream({
        message: userMessage,
        userId: CURRENT_USER_ID,
        conversationId: convId,
        attachmentIds: selectedAttachmentIds,
        documentIds: activeWorkspaceDocId ? [activeWorkspaceDocId] : [],
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
      messages.length,
      ensureConversation,
      refresh,
      rename,
      activeConversation?.title,
      selectedAttachmentIds,
      activeWorkspaceDocId,
      ragEnabled,
      startStream,
      t,
      formatTime,
    ],
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
        setSelectedAttachmentIds((prev) => (prev.includes(uploaded.id) ? prev : [...prev, uploaded.id]));
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
      setSelectedAttachmentIds((prev) => prev.filter((id) => id !== attachmentId));
      toast.success(t("chat.attachmentRemoved"), { description: target?.original_name || attachmentId });
    },
    [activeId, attachments, t],
  );

  const toggleAttachmentSelection = useCallback((attachmentId: string) => {
    setSelectedAttachmentIds((prev) =>
      prev.includes(attachmentId)
        ? prev.filter((id) => id !== attachmentId)
        : [...prev, attachmentId],
    );
  }, []);

  async function handleSend() {
    if (!input.trim() || loading || isStreaming) return;
    const userMessage = input;
    setInput("");
    setTimeout(() => autoResize(), 0);
    sendMessage(userMessage);
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }

  // ── Regenerate: resend the last user message ───────────────────────

  const handleRegenerate = useCallback(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        sendMessage(messages[i].content, i);
        return;
      }
    }
  }, [messages, sendMessage]);

  // ── Edit user message & resend ─────────────────────────────────────

  const handleEditAndResend = useCallback(
    (index: number, newContent: string) => {
      sendMessage(newContent, index);
    },
    [sendMessage],
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
    [messages, activeId, t],
  );

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape" && isStreaming) {
      cancelStream();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey && !loading && !isStreaming) {
      e.preventDefault();
      handleSend();
    }
  }

  const _chatRole = systemConfig?.model_roles?.find((r) => r.role === "chat");
  const maxContextTokens = _chatRole?.context_window_tokens ?? _chatRole?.max_tokens ?? null;

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

              {isStreaming && streamingContent ? (
                /* Live streaming text with cursor */
                <div
                  className="text-evergreen text-base leading-relaxed px-1"
                  aria-live="polite"
                  aria-busy="true"
                >
                  <MarkdownMessage content={streamingContent} />
                  <span
                    className="inline-block w-0.5 h-[1.1em] bg-evergreen/60 ml-0.5 align-middle animate-cursor-blink"
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
              {attachments.map((attachment) => {
                const selected = selectedAttachmentIds.includes(attachment.id);
                return (
                  <div
                    key={attachment.id}
                    className={`inline-flex items-center gap-1 rounded-full border px-3 py-1.5 text-xs transition-colors ${
                      selected
                        ? "bg-light-cyan/30 border-pacific-blue/30 text-evergreen"
                        : "bg-white border-gray-300 text-evergreen/55"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => toggleAttachmentSelection(attachment.id)}
                      className="inline-flex items-center gap-1 cursor-pointer"
                      title={selected ? t("chat.excludeFromNextMessage") : t("chat.includeInNextMessage")}
                    >
                      <Icon name="description" size={14} className={selected ? "text-pacific-blue" : "text-evergreen/40"} />
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
                );
              })}
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
              disabled={isStreaming}
              className="w-full bg-transparent border-0 focus:ring-0 resize-none text-evergreen placeholder-gray-400 px-4 pt-3 pb-2 text-base disabled:opacity-60 disabled:cursor-not-allowed"
              placeholder={isStreaming ? (t("chat.aiThinking") ?? "Generating…") : t("chat.typeYourMessage")}
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
                      <div className="absolute bottom-full left-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-1 min-w-[160px] z-20">
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
                          disabled
                          className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-evergreen/30 cursor-not-allowed"
                        >
                          <Icon name="image" size={16} />
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
                      <div className="absolute bottom-full left-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-2 min-w-[200px] z-20">
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

                {/* Context usage ring */}
                <ContextRingIndicator
                  contextTokens={contextTokens}
                  maxContextTokens={maxContextTokens}
                />

                {/* Model selector */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setModelMenuOpen((v) => !v)}
                    className="flex items-center gap-0.5 rounded-lg px-2.5 py-2 text-xs text-evergreen/60 hover:text-evergreen hover:bg-gray-100 transition-colors max-w-[140px]"
                    aria-label="Select model"
                  >
                    <span className="truncate">{formatModelName(chatModel, systemConfig?.default_model)}</span>
                    <Icon name="keyboard_arrow_down" size={14} className="shrink-0" />
                  </button>
                  {modelMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={() => setModelMenuOpen(false)} />
                      <div className="absolute bottom-full right-0 mb-2 bg-white rounded-xl border border-gray-100 shadow-lg py-1 min-w-[220px] max-h-52 overflow-y-auto z-20">
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

                {/* Send / Cancel */}
                {isStreaming ? (
                  <button
                    type="button"
                    onClick={cancelStream}
                    aria-label={t("chat.stopGenerating") ?? "Stop generating"}
                    className="w-8 h-8 flex items-center justify-center rounded-lg bg-burnt-tangerine hover:bg-burnt-tangerine/85 text-white transition-colors"
                  >
                    <Icon name="stop_circle" size={20} />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                    aria-label={t("chat.sendMessage")}
                    className="w-8 h-8 flex items-center justify-center rounded-lg bg-burnt-tangerine hover:bg-burnt-tangerine/85 text-white disabled:opacity-30 transition-colors"
                  >
                    <Icon name="arrow_upward" size={20} />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Attachment count hint */}
          {selectedAttachmentIds.length > 0 && (
            <p className="mt-2 px-1 text-xs text-evergreen/45">
              {selectedAttachmentIds.length === 1
                ? t("chat.attachmentCountOne", { count: selectedAttachmentIds.length })
                : t("chat.attachmentCountOther", { count: selectedAttachmentIds.length })}
            </p>
          )}

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,text/plain,text/markdown"
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
        writebackSavedDocId={writebackSavedDocId}
        onActiveDocumentChange={setActiveWorkspaceDocId}
        onInsertCommand={(command) => {
          setInput(command);
          requestAnimationFrame(() => {
            textareaRef.current?.focus();
            autoResize();
          });
        }}
        onAttachmentUploaded={(attachment) => {
          setAttachments((prev) => {
            if (prev.some((item) => item.id === attachment.id)) return prev;
            return [...prev, attachment];
          });
          setSelectedAttachmentIds((prev) => {
            if (prev.includes(attachment.id)) return prev;
            return [...prev, attachment.id];
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
  onRegenerate,
  onFeedback,
  loading,
}: {
  message: Message;
  index: number;
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

        {/* Message text */}
        <div className="text-evergreen text-base leading-relaxed px-1">
          <MarkdownMessage content={message.content} sources={message.metrics?.sources} />
        </div>

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
                    flex items-center justify-center text-evergreen font-bold shrink-0"
        aria-hidden="true"
      >
        JS
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
