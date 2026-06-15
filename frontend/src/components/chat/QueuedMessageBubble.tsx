"use client";

import { Clock, Send, X } from "lucide-react";
import { ChatMessageShell } from "@/components/product/ChatMessageShell";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { useI18n } from "@/hooks/useI18n";

interface QueuedMessageBubbleProps {
  text: string;
  idle: boolean;
  onDiscard: () => void;
  onSendNow: () => void;
  onEdit: () => void;
}

/**
 * Dimmed pending user bubble for a follow-up typed while the model is still
 * streaming. Not part of the `messages` array (like the live streaming
 * indicator), so it can't disturb message ordering or the persisted-id backfill.
 * Auto-fires on normal completion; on a manual Stop/error it stays put (idle)
 * with explicit Send-now / discard controls.
 */
export function QueuedMessageBubble({
  text,
  idle,
  onDiscard,
  onSendNow,
  onEdit,
}: QueuedMessageBubbleProps) {
  const { t } = useI18n();
  return (
    <ChatMessageShell messageRole="user" className="opacity-60">
        {/* Queued tag */}
        <div className="mr-1 flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock size={13} className="animate-pulse" />
          <span className="font-medium">{t("chat.queued")}</span>
        </div>

        {/* Bubble — click to reclaim into the composer for editing */}
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                type="button"
                onClick={onEdit}
                aria-label={t("chat.editQueued")}
                variant="ghost"
                className="text-left rounded-2xl rounded-tr-sm border border-dashed border-primary/30 bg-primary/10 p-5
                           text-base leading-relaxed text-foreground transition-colors cursor-text hover:border-primary/50"
              >
                <p className="whitespace-pre-wrap m-0">{text}</p>
              </Button>
            }
          />
          <TooltipContent>
            {t("chat.editQueued")}
          </TooltipContent>
        </Tooltip>

        {/* Controls */}
        <div className="flex items-center gap-0.5 mr-1 mt-0.5">
          {idle && (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    type="button"
                    onClick={onSendNow}
                    variant="ghost"
                    size="sm"
                    className="p-1 rounded text-muted-foreground hover:text-primary hover:bg-primary/10
                               transition-colors cursor-pointer"
                    aria-label={t("chat.sendQueuedNow")}
                  >
                    <Send size={14} />
                  </Button>
                }
              />
              <TooltipContent>
                {t("chat.sendQueuedNow")}
              </TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  type="button"
                  onClick={onDiscard}
                  variant="ghost"
                  size="sm"
                  className="p-1 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10
                             transition-colors cursor-pointer"
                  aria-label={t("chat.cancelQueued")}
                >
                  <X size={14} />
                </Button>
              }
            />
            <TooltipContent>
              {t("chat.cancelQueued")}
            </TooltipContent>
          </Tooltip>
        </div>
    </ChatMessageShell>
  );
}
