"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Check, Copy, Pencil } from "lucide-react";
import { ChatMessageShell } from "@/components/product/ChatMessageShell";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/hooks/useI18n";
import type { Message } from "@/lib/types";

interface UserMessageProps {
  message: Message;
  index: number;
  onEdit: (index: number, newContent: string) => void;
  loading: boolean;
}

export function UserMessage({ message, index, onEdit, loading }: UserMessageProps) {
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
    <ChatMessageShell messageRole="user">
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
              <Tooltip>
                <TooltipTrigger
                  render={
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
                    >
                      {copied ? <Check size={14} /> : <Copy size={14} />}
                    </Button>
                  }
                />
                <TooltipContent>
                  {t("chat.copyMessage")}
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={
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
                    >
                      <Pencil size={14} />
                    </Button>
                  }
                />
                <TooltipContent>
                  {t("workspace.edit")}
                </TooltipContent>
              </Tooltip>
            </div>
          </>
        )}
    </ChatMessageShell>
  );
}
