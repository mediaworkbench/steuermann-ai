"use client";

import { useState, useCallback, useEffect } from "react";
import { toast } from "sonner";
import {
  deleteConversationAttachment,
  fetchConversationAttachments,
  uploadConversationAttachment,
} from "@/lib/api";
import type { ConversationAttachment } from "@/lib/types";
import { useI18n } from "@/hooks/useI18n";

interface UseConversationAttachmentsResult {
  attachments: ConversationAttachment[];
  uploadingAttachment: boolean;
  addAttachment: (att: ConversationAttachment) => void;
  handleAttachmentUpload: (
    event: React.ChangeEvent<HTMLInputElement>,
    opts: {
      loading: boolean;
      ensureConversation: (seedText?: string) => Promise<string | null>;
      onUploaded?: (att: ConversationAttachment) => void;
      refresh?: () => void;
    },
  ) => Promise<void>;
  handleAttachmentDelete: (attachmentId: string) => Promise<void>;
}

export function useConversationAttachments(
  activeId: string | null,
): UseConversationAttachmentsResult {
  const { t } = useI18n();
  const [attachments, setAttachments] = useState<ConversationAttachment[]>([]);
  const [uploadingAttachment, setUploadingAttachment] = useState(false);

  // Re-fetch attachments whenever the active conversation changes.
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

  // Dedup-safe setter — silently ignores duplicates by ID.
  const addAttachment = useCallback((att: ConversationAttachment) => {
    setAttachments((prev) => {
      if (prev.some((item) => item.id === att.id)) return prev;
      return [...prev, att];
    });
  }, []);

  const handleAttachmentUpload = useCallback(
    async (
      event: React.ChangeEvent<HTMLInputElement>,
      {
        loading,
        ensureConversation,
        onUploaded,
        refresh,
      }: {
        loading: boolean;
        ensureConversation: (seedText?: string) => Promise<string | null>;
        onUploaded?: (att: ConversationAttachment) => void;
        refresh?: () => void;
      },
    ) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file || uploadingAttachment || loading) return;

      setUploadingAttachment(true);
      try {
        const convId = await ensureConversation(file.name);
        if (!convId) throw new Error(t("chat.couldNotCreateConversationForAttachment"));

        const uploaded = await uploadConversationAttachment(convId, file);
        if (!uploaded) throw new Error(t("chat.attachmentUploadFailed"));

        addAttachment(uploaded);
        onUploaded?.(uploaded);
        refresh?.();
        toast.success(t("chat.attachmentUploaded"), { description: uploaded.original_name });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : t("chat.attachmentUploadFailed");
        toast.error(t("chat.attachmentUploadFailed"), { description: errorMessage });
      } finally {
        setUploadingAttachment(false);
      }
    },
    [uploadingAttachment, addAttachment, t],
  );

  const handleAttachmentDelete = useCallback(
    async (attachmentId: string) => {
      if (!activeId) return;
      const target = attachments.find((a) => a.id === attachmentId);
      const deleted = await deleteConversationAttachment(activeId, attachmentId);
      if (!deleted) {
        toast.error(t("chat.attachmentDeleteFailed"));
        return;
      }
      setAttachments((prev) => prev.filter((a) => a.id !== attachmentId));
      toast.success(t("chat.attachmentRemoved"), { description: target?.original_name || attachmentId });
    },
    [activeId, attachments, t],
  );

  return {
    attachments,
    uploadingAttachment,
    addAttachment,
    handleAttachmentUpload,
    handleAttachmentDelete,
  };
}
