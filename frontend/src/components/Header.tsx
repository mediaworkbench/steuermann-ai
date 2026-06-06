"use client";

import { useState } from "react";
import Link from "next/link";
import { Icon } from "./Icon";
import { ExportDialog } from "./ExportDialog";
import { Button } from "@/components/ui/Button";
import { AUTH_ENABLED } from "@/lib/runtime";
import { useRole } from "@/context/RoleContext";
import type { Conversation } from "@/lib/types";
import { useI18n } from "@/hooks/useI18n";

interface HeaderProps {
  chatTitle?: string;
  onOpenSidebar?: () => void;
  activeConversation?: Conversation | null;
  workspaceSidebarOpen?: boolean;
  onToggleWorkspaceSidebar?: () => void;
}

export function Header({ chatTitle = "AI Agent", onOpenSidebar, activeConversation, workspaceSidebarOpen, onToggleWorkspaceSidebar }: HeaderProps) {
  const { t, formatRelativeTime } = useI18n();
  const { isAdmin } = useRole();
  const [showExport, setShowExport] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  const navLinks = [
    ...(isAdmin ? [{ href: "/metrics", label: t("header.metrics"), icon: "bar_chart" }] : []),
    { href: "/chats", label: t("header.chats"), icon: "forum" },
    { href: "/memories", label: t("header.memory"), icon: "psychology" },
    { href: "/settings", label: t("header.settings"), icon: "settings" },
    ...(isAdmin ? [{ href: "/admin/rag", label: t("header.ragExplorer"), icon: "travel_explore" }] : []),
    ...(isAdmin ? [{ href: "/admin", label: t("header.admin"), icon: "admin_panel_settings" }] : []),
  ];
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
      className="h-16 md:h-20 bg-surface border-b border-border flex items-center
                 justify-between px-4 md:px-8 shrink-0 sticky top-0 z-20"
    >
      <div className="flex items-center gap-3">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="md:hidden text-foreground hover:text-primary transition-colors
                     min-h-11 min-w-11 flex items-center justify-center -ml-2"
          onClick={onOpenSidebar}
          aria-label={t("header.openNavigation")}
          aria-expanded="false"
          aria-controls="sidebar"
        >
          <Icon name="menu" size={24} />
        </Button>
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
                  <Icon name="push_pin" size={12} className="text-primary" />
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
            <Icon
              name="download"
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="hidden lg:inline">{t("common.export")}</span>
          </Button>
        )}

        {/* Workspace sidebar toggle — only on chat page */}
        {onToggleWorkspaceSidebar && (
          <Button
            type="button"
            onClick={onToggleWorkspaceSidebar}
            variant="ghost"
            size="sm"
            className="hidden md:flex gap-1.5 text-foreground hover:text-primary text-sm font-medium group min-h-11 min-w-11 justify-center"
            aria-label={t("chat.toggleWorkspaceSidebar")}
            title={t("chat.toggleWorkspaceSidebar")}
          >
            <Icon
              name={workspaceSidebarOpen ? "right_panel_close" : "right_panel_open"}
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="hidden lg:inline">{t("chat.workspace")}</span>
          </Button>
        )}

        <nav className="flex items-center gap-2 md:gap-6" aria-label="Main navigation">
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            prefetch={false}
            className="flex items-center gap-1.5 text-foreground hover:text-primary
                       transition-colors text-sm font-medium group min-h-11 min-w-11
                       justify-center md:justify-start"
          >
            <Icon
              name={link.icon}
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="hidden sm:inline">{link.label}</span>
          </Link>
        ))}
        </nav>

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
            <Icon name="logout" size={18} />
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
