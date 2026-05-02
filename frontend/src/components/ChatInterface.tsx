"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Icon } from "./Icon";
import { MemoryRating } from "./MemoryRating";
import { MetricsPanel } from "./MetricsPanel";
import { WorkspaceSidebar, type WorkspaceDocument } from "./WorkspaceSidebar";
import { useConversationContext } from "./LayoutShell";
import { useI18n } from "@/hooks/useI18n";
import {
  deleteConversationAttachment,
  fetchConversation,
  fetchConversationAttachments,
  setMessageFeedback,
  uploadConversationAttachment,
} from "@/lib/api";
import { selectActiveAttachmentIds } from "@/lib/attachments";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type {
  ChatResponse,
  ConversationAttachment,
  Message,
  PersistedMessage,
  Source,
} from "@/lib/types";

/**
 * Replace [N] footnote references with clickable markdown links using the sources array.
 * E.g. "[1]" becomes "[<sup>1</sup>](url)" if source 1 has a URL, or bold "[<sup>1</sup>]" for RAG.
 */
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

function MemoryUsedList({
  memories,
}: {
  memories?: Array<{
    memory_id: string;
    text?: string;
    user_rating?: number | null;
    importance_score?: number | null;
    is_related?: boolean;
  }>;
}) {
  if (!memories || memories.length === 0) return null;

  return (
    <div className="mt-2 px-1 w-full max-w-3xl">
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-evergreen/45">Memories used</div>
      <div className="space-y-1.5">
        {memories.map((memory) => (
          <div
            key={memory.memory_id}
            className="flex items-start justify-between gap-3 rounded-lg border border-gray-200 bg-white/80 px-2.5 py-2"
          >
            <div className="min-w-0">
              <div className="truncate text-xs text-evergreen/70">{memory.text || memory.memory_id}</div>
              <div className="mt-0.5 text-[11px] text-evergreen/45">
                {memory.is_related ? "Related" : "Primary"}
                {typeof memory.importance_score === "number" ? ` • score ${memory.importance_score.toFixed(2)}` : ""}
              </div>
            </div>
            <MemoryRating
              memoryId={memory.memory_id}
              initialRating={typeof memory.user_rating === "number" ? memory.user_rating : 0}
            />
          </div>
        ))}
      </div>
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
      response_time_ms: pm.response_time_ms ?? undefined,
      model: pm.model_name ?? undefined,
      tools_executed: pm.tools_used?.map((t) => ({
        name: t.name,
        status: t.status,
      })),
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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { activeId, create, refresh, rename, activeConversation, workspaceSidebarOpen, setWorkspaceSidebarOpen } =
    useConversationContext();

  const templates = useMemo(
    () => [
      { icon: "lightbulb", label: t("chat.templates.explainConcept"), prompt: t("chat.templates.explainPrompt") },
      { icon: "code", label: t("chat.templates.helpCode"), prompt: t("chat.templates.codePrompt") },
      { icon: "summarize", label: t("chat.templates.summarizeText"), prompt: t("chat.templates.summarizePrompt") },
      { icon: "translate", label: t("chat.templates.translate"), prompt: t("chat.templates.translatePrompt") },
      { icon: "psychology", label: t("chat.templates.brainstormIdeas"), prompt: t("chat.templates.brainstormPrompt") },
      { icon: "troubleshoot", label: t("chat.templates.debugIssue"), prompt: t("chat.templates.debugPrompt") },
    ],
    [t],
  );

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
      setMessages(detail.messages.map((msg) => toUiMessage(msg, formatTime)));
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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
    el.style.overflowY = el.scrollHeight > 200 ? "auto" : "hidden";
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

      const apiBase = process.env.NEXT_PUBLIC_API_BASE || "/api/proxy";
      const startTime = Date.now();

      try {
        const response = await fetch(`${apiBase}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: userMessage,
            user_id: CURRENT_USER_ID,
            conversation_id: convId,
            attachment_ids: selectedAttachmentIds,
          }),
        });

        if (!response.ok) throw new Error(`API error: ${response.status}`);

        const data: ChatResponse = await response.json();
        const elapsed = Date.now() - startTime;

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.response,
            timestamp: formatTime(new Date()),
            metrics: {
              response_time_ms: elapsed,
              input_tokens: data.metadata?.input_tokens,
              output_tokens: data.metadata?.output_tokens,
              finish_reason: "stop",
              model: data.metadata?.model_used,
              temperature: undefined,
              tools_executed: data.metadata?.tools_executed?.map((name) => ({
                name,
                status: "success" as const,
              })),
              sources: data.metadata?.sources,
              attachments_used: data.metadata?.attachments_used,
              documents_used: data.metadata?.documents_used,
              memories_used: data.metadata?.memories_used,
            },
          },
        ]);

        if (data.metadata?.workspace_document_writeback?.status === "saved") {
          fetchWorkspaceDocuments();
          toast.success(t("chat.workspaceDocumentSaved"), {
            description: `${data.metadata.workspace_document_writeback.filename} updated to v${data.metadata.workspace_document_writeback.version}`,
          });
        }

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
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : t("chat.failedToSendMessage");
        toast.error(t("chat.messageFailed"), { description: errorMessage });
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${errorMessage}`,
            timestamp: formatTime(new Date()),
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [
      activeId,
      messages.length,
      ensureConversation,
      fetchWorkspaceDocuments,
      refresh,
      rename,
      activeConversation?.title,
      selectedAttachmentIds,
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
    if (!input.trim() || loading) return;
    const userMessage = input;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
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
    if (e.key === "Enter" && !e.shiftKey && !loading) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <>
      <div className="flex h-full min-h-0">
        {/* ─── Main chat area ─── */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          {/* ─── Chat messages ─── */}
          <div
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
            <p className="text-sm mb-8">
              {t("chat.startConversationHint")}
            </p>

            {/* ── Template starters ── */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 max-w-2xl w-full">
              {templates.map((template) => (
                <button
                  key={template.label}
                  onClick={() => {
                    setInput(template.prompt);
                    textareaRef.current?.focus();
                  }}
                  className="flex items-center gap-2.5 px-4 py-3.5 rounded-xl border border-light-cyan
                             bg-light-cyan/10 hover:bg-light-cyan/30 text-evergreen/70 hover:text-evergreen
                               transition-colors text-sm text-left group cursor-pointer min-h-11"
                >
                  <Icon
                    name={template.icon}
                    size={18}
                    className="text-pacific-blue shrink-0 group-hover:scale-110 transition-transform"
                  />
                  <span className="font-medium">{template.label}</span>
                </button>
              ))}
            </div>
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

        {/* ─── Typing indicator ─── */}
        {loading && (
          <div className="flex gap-4 max-w-5xl mx-auto">
            <div
              className="w-8 h-8 rounded-full bg-evergreen flex items-center justify-center shrink-0 mt-1"
              aria-hidden="true"
            >
              <Icon name="smart_toy" size={18} className="text-white" />
            </div>
            <div className="flex flex-col gap-1 items-start">
              <div className="flex items-center gap-2 ml-1">
                <span className="text-sm font-bold text-evergreen">
                  {t("chat.aiAgent")}
                </span>
              </div>
              <div
                className="px-4 py-3 rounded-2xl rounded-tl-sm bg-light-cyan/30 border border-light-cyan/50 flex items-center gap-1.5"
                role="status"
                aria-label={t("chat.aiThinking")}
              >
                <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
                <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
                <span className="typing-dot w-2 h-2 rounded-full bg-pacific-blue" />
              </div>
            </div>
          </div>
        )}

        <div className="h-12" ref={messagesEndRef} />
      </div>

      {/* ═══════════ INPUT BAR ═══════════ */}
      <div className="p-4 md:px-6 lg:px-12 md:pb-8 bg-white shrink-0 border-t border-gray-100">
        <div className="max-w-5xl mx-auto">
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

          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,text/plain,text/markdown"
            className="sr-only"
            onChange={handleAttachmentUpload}
          />

          <div
            className="relative flex items-center gap-2 rounded-lg border border-gray-300 p-1 px-2
                        bg-white transition-colors duration-300 shadow-sm"
          >
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading || uploadingAttachment}
              className="shrink-0 rounded px-2 py-2 text-evergreen/45 hover:text-pacific-blue hover:bg-pacific-blue/5 disabled:opacity-40 disabled:cursor-not-allowed"
              aria-label={t("chat.uploadingAttachment")}
              title={t("chat.uploadingAttachment")}
            >
              <Icon name="attach_file" size={20} />
            </button>
            <label htmlFor="message-input" className="sr-only">
              {t("chat.message")}
            </label>
            <textarea
              id="message-input"
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                autoResize();
              }}
              onKeyDown={handleKeyDown}
              className="w-full bg-transparent border-0 focus:ring-0 text-evergreen
                         placeholder-gray-400 py-3 max-h-50 min-h-11 resize-none text-base"
              placeholder={t("chat.typeYourMessage")}
              rows={1}
              style={{ fieldSizing: "content" } as React.CSSProperties}
              aria-label={t("chat.typeYourMessage")}
            />
            <div className="flex items-center gap-2 pr-1 border-l border-gray-200 pl-2 h-8 my-auto shrink-0">
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="text-evergreen/40 hover:text-burnt-tangerine transition-colors rounded
                           hover:bg-burnt-tangerine/5 shrink-0 flex items-center gap-1 px-2 py-1
                           disabled:opacity-40 disabled:cursor-not-allowed"
                aria-label={t("chat.sendMessage")}
              >
                <Icon name="send" size={20} className="text-burnt-tangerine" />
                <span className="text-xs font-medium text-evergreen/60 hidden sm:inline">
                  {t("chat.send")}
                </span>
              </button>
            </div>
          </div>

          {attachments.length > 0 && (
            <div className="flex items-center gap-2 mt-2 px-1">
                <span className="text-xs text-evergreen/45">
                  {selectedAttachmentIds.length === 1
                    ? t("chat.attachmentCountOne", { count: selectedAttachmentIds.length })
                    : t("chat.attachmentCountOther", { count: selectedAttachmentIds.length })}
                </span>
            </div>
          )}
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

        {/* Source badges */}
        <SourceBadges sources={message.metrics?.sources} />

        {/* Attachment context badges */}
        <AttachmentUsedBadges attachments={message.metrics?.attachments_used} />

        {/* Workspace document context badges */}
        <DocumentUsedBadges documents={message.metrics?.documents_used} />

        {/* Memory context list + rating controls */}
        <MemoryUsedList memories={message.metrics?.memories_used} />

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
