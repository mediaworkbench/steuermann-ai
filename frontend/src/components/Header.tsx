"use client";

import { useState } from "react";
import Link from "next/link";
import { Download, LogOut, Pin } from "lucide-react";
import { ExportDialog } from "./ExportDialog";
import { Button } from "@/components/ui/button";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { AUTH_ENABLED } from "@/lib/runtime";
import type { Conversation } from "@/lib/types";
import { useI18n } from "@/hooks/useI18n";

interface HeaderProps {
  chatTitle?: string;
  activeConversation?: Conversation | null;
}

export function Header({ chatTitle = "AI Agent", activeConversation }: HeaderProps) {
  const { t, formatRelativeTime } = useI18n();
  const [showExport, setShowExport] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  const hasMeta = activeConversation != null;
  const msgCount = activeConversation?.message_count;
  const createdLabel = activeConversation?.created_at
    ? formatRelativeTime(activeConversation.created_at)
    : null;
  const lang = activeConversation?.language?.toUpperCase();

  async function handleLogout() {
    setLoggingOut(true);
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } finally {
      window.location.assign("/login");
    }
  }

  return (
    <header
      className="h-16 md:h-20 bg-sidebar-muted border-b border-border flex items-center
                 justify-between px-4 md:px-8 shrink-0 sticky top-0 z-20"
    >
      <div className="flex items-center gap-3">
        <SidebarTrigger className="-ml-2" />
        <Link href="/" className="header-title-slot flex flex-col hover:opacity-80 transition-opacity">
          <h2 className="text-foreground font-bold text-base leading-tight truncate max-w-50 md:max-w-none">
            {chatTitle}
          </h2>
          {hasMeta ? (
            <span className="text-muted-foreground text-xs font-mono hidden md:flex items-center gap-2">
              {createdLabel && <span>{createdLabel}</span>}
              {msgCount != null && msgCount > 0 && (
                <>
                  <span className="text-muted-foreground">·</span>
                  <span>
                    {msgCount === 1
                      ? t("header.messageCountOne", { count: msgCount })
                      : t("header.messageCountOther", { count: msgCount })}
                  </span>
                </>
              )}
              {lang && (
                <>
                  <span className="text-muted-foreground">·</span>
                  <span>{lang}</span>
                </>
              )}
              {activeConversation?.pinned && (
                <>
                  <span className="text-muted-foreground">·</span>
                  <Pin size={12} className="text-primary" />
                </>
              )}
            </span>
          ) : (
            <span className="text-primary text-xs font-mono hidden md:block">
              {t("header.activeSession")}
            </span>
          )}
        </Link>
      </div>

      <div className="flex items-center gap-3">
        {/* Export button — only when a conversation is active */}
        {activeConversation && (
          <Button
            type="button"
            onClick={() => setShowExport(true)}
            variant="ghost"
            size="sm"
            className="gap-1.5 text-foreground hover:text-primary text-sm group min-h-11 min-w-11 justify-center"
            title={t("header.exportConversation")}
          >
            <Download
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="hidden lg:inline">{t("header.exportChat")}</span>
          </Button>
        )}

        {AUTH_ENABLED && (
          <Button
            type="button"
            onClick={handleLogout}
            disabled={loggingOut}
            variant="ghost"
            size="sm"
            className="gap-1.5 text-foreground hover:text-destructive text-sm font-medium min-h-11 min-w-11 justify-center md:justify-start disabled:opacity-50"
            title={t("header.signOut")}
          >
            <LogOut size={18} />
            <span className="hidden sm:inline">{loggingOut ? t("header.signingOut") : t("header.signOut")}</span>
          </Button>
        )}
      </div>

      {/* Export dialog */}
      {showExport && activeConversation && (
        <ExportDialog
          conversationId={activeConversation.id}
          conversationTitle={activeConversation.title}
          onClose={() => setShowExport(false)}
        />
      )}
    </header>
  );
}
