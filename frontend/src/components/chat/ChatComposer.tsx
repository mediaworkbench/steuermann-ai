"use client";

import { useRef, useCallback, useLayoutEffect } from "react";
import { ArrowUp, Database, FileText, Mic, PanelRightClose, PanelRightOpen, StopCircle, X } from "lucide-react";
import { AttachMenu } from "./AttachMenu";
import { ToolsMenu } from "./ToolsMenu";
import { ContextWindowMenu } from "./ContextWindowMenu";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { formatModelName } from "@/components/product/modelSelection";
import { useEdgeEffect } from "@/hooks/useEdgeEffect";
import { useI18n } from "@/hooks/useI18n";
import type { SystemConfig } from "@/lib/api";
import type { ConversationAttachment, ContextBreakdown } from "@/lib/types";

/** Imperative handle for ChatComposer — follows the editorApiRef pattern. */
export interface ChatComposerHandle {
  focus: () => void;
  resize: () => void;
}

interface ChatComposerProps {
  input: string;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onCancel: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  isStreaming: boolean;
  loading: boolean;
  queueFull: boolean;
  attachments: ConversationAttachment[];
  uploadingAttachment: boolean;
  onAttachmentUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onAttachmentDelete: (id: string) => void;
  onAttachmentPillClick: (attachment: ConversationAttachment) => void;
  ragEnabled: boolean;
  onRagToggle: () => void;
  workspaceSidebarOpen: boolean;
  onWorkspaceSidebarToggle: () => void;
  toolToggles: Record<string, boolean>;
  onToolToggle: (toolId: string) => void;
  systemConfig: SystemConfig | null;
  selectedChatModel: string;
  availableChatModels: string[];
  onModelChange: (model: string) => void;
  contextTokens: number;
  maxContextTokens: number | null;
  contextBreakdown?: ContextBreakdown | null;
  userMessageCount: number;
  assistantMessageCount: number;
  isCompacting: boolean;
  activeId: string | null;
  onCompactContext: () => void;
  attachMenuOpen: boolean;
  onAttachMenuToggle: () => void;
  toolsMenuOpen: boolean;
  onToolsMenuToggle: () => void;
  contextMenuOpen: boolean;
  onContextMenuToggle: () => void;
  /** Populated via useLayoutEffect so ChatInterface can call focus()/resize(). */
  composerApiRef?: React.MutableRefObject<ChatComposerHandle | null>;
}

export function ChatComposer({
  input,
  onInputChange,
  onSend,
  onCancel,
  onKeyDown,
  isStreaming,
  loading,
  queueFull,
  attachments,
  uploadingAttachment,
  onAttachmentUpload,
  onAttachmentDelete,
  onAttachmentPillClick,
  ragEnabled,
  onRagToggle,
  workspaceSidebarOpen,
  onWorkspaceSidebarToggle,
  toolToggles,
  onToolToggle,
  systemConfig,
  selectedChatModel,
  availableChatModels,
  onModelChange,
  contextTokens,
  maxContextTokens,
  contextBreakdown,
  userMessageCount,
  assistantMessageCount,
  isCompacting,
  activeId,
  onCompactContext,
  attachMenuOpen,
  onAttachMenuToggle,
  toolsMenuOpen,
  onToolsMenuToggle,
  contextMenuOpen,
  onContextMenuToggle,
  composerApiRef,
}: ChatComposerProps) {
  const { t } = useI18n();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxH = 260; // ~10 lines
    el.style.height = Math.min(el.scrollHeight, maxH) + "px";
    el.style.overflowY = el.scrollHeight > maxH ? "auto" : "hidden";
  }, []);

  // Expose focus/resize to ChatInterface via a mutable ref (same pattern as
  // ActiveDocumentContext's editorApiRef).
  useLayoutEffect(() => {
    if (!composerApiRef) return;
    composerApiRef.current = {
      focus: () => textareaRef.current?.focus(),
      resize: autoResize,
    };
    return () => { composerApiRef.current = null; };
  }, [composerApiRef, autoResize]);

  // Refocus the textarea when a stream finishes so the user can type immediately.
  useEdgeEffect(isStreaming, { onFalling: () => textareaRef.current?.focus() });

  return (
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
                  onClick={() => onAttachmentPillClick(attachment)}
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
                  onClick={() => onAttachmentDelete(attachment.id)}
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
            onChange={(e) => { onInputChange(e.target.value); autoResize(); }}
            onKeyDown={onKeyDown}
            disabled={queueFull}
            className="resize-none rounded-none border-0 bg-transparent px-4 pb-2 pt-3 text-base text-foreground shadow-none focus:ring-0"
            placeholder={queueFull ? t("chat.queuedSlotFull") : isStreaming ? t("chat.queuedHint") : t("chat.typeYourMessage")}
            aria-label={t("chat.typeYourMessage")}
            rows={2}
          />

          {/* Bottom toolbar */}
          <div className="flex items-center gap-1 border-t border-border px-3 py-2">

            {/* Left group: +, tools, RAG, workspace toggle */}
            <div className="flex items-center gap-0.5">
              <AttachMenu
                open={attachMenuOpen}
                onToggle={onAttachMenuToggle}
                onClose={() => onAttachMenuToggle()}
                disabled={loading || uploadingAttachment || isStreaming}
                onFileClick={() => fileInputRef.current?.click()}
                onImageClick={() => imageInputRef.current?.click()}
              />

              <ToolsMenu
                open={toolsMenuOpen}
                onToggle={onToolsMenuToggle}
                onClose={() => onToolsMenuToggle()}
                systemConfig={systemConfig}
                toolToggles={toolToggles}
                onToolToggle={onToolToggle}
              />

              {/* RAG toggle */}
              <Button
                type="button"
                onClick={onRagToggle}
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

              {/* Workspace panel toggle */}
              <Button
                type="button"
                onClick={onWorkspaceSidebarToggle}
                title={t("chat.toggleWorkspaceSidebar")}
                aria-label={t("chat.toggleWorkspaceSidebar")}
                aria-pressed={workspaceSidebarOpen}
                variant="ghost"
                size="sm"
                className={`p-1.5 rounded-lg transition-colors ${
                  workspaceSidebarOpen
                    ? "text-primary hover:bg-primary/10"
                    : "text-muted-foreground hover:bg-surface-muted hover:text-foreground"
                }`}
              >
                {workspaceSidebarOpen ? <PanelRightClose size={20} /> : <PanelRightOpen size={20} />}
              </Button>
            </div>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Right group: context ring, model, mic, send */}
            <div className="flex items-center gap-1.5">
              <ContextWindowMenu
                open={contextMenuOpen}
                onToggle={onContextMenuToggle}
                onClose={() => onContextMenuToggle()}
                contextTokens={contextTokens}
                maxContextTokens={maxContextTokens}
                contextBreakdown={contextBreakdown}
                userMessageCount={userMessageCount}
                assistantMessageCount={assistantMessageCount}
                isStreaming={isStreaming}
                isCompacting={isCompacting}
                activeId={activeId}
                onCompact={onCompactContext}
              />

              {/* Model selector */}
              <div className="relative">
                <Select
                  value={selectedChatModel}
                  onChange={(e) => void onModelChange(e.target.value)}
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

              {/* Send / Cancel */}
              {isStreaming || loading ? (
                <>
                  <Button
                    type="button"
                    onClick={onCancel}
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
                      onClick={onSend}
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
                  onClick={onSend}
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

        {/* Hidden file inputs */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,text/plain,text/markdown"
          className="sr-only"
          onChange={onAttachmentUpload}
        />
        <input
          ref={imageInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.gif,.webp,image/jpeg,image/png,image/gif,image/webp"
          className="sr-only"
          onChange={onAttachmentUpload}
        />
      </div>
    </div>
  );
}
