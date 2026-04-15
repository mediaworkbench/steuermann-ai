"use client";

import { useState } from "react";
import Link from "next/link";
import { Icon } from "./Icon";
import { ExportDialog } from "./ExportDialog";
import { AUTH_ENABLED } from "@/lib/runtime";
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
  const [showExport, setShowExport] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);
  const navLinks = [
    { href: "/metrics", label: t("header.metrics"), icon: "bar_chart" },
    { href: "/settings", label: t("header.settings"), icon: "settings" },
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
      className="h-16 md:h-20 bg-light-cyan/20 border-b border-light-cyan flex items-center
                 justify-between px-4 md:px-8 shrink-0 sticky top-0 z-20"
    >
      <div className="flex items-center gap-3">
        <button
          className="md:hidden text-evergreen hover:text-pacific-blue transition-colors
                     min-h-11 min-w-11 flex items-center justify-center -ml-2"
          onClick={onOpenSidebar}
          aria-label={t("header.openNavigation")}
          aria-expanded="false"
          aria-controls="sidebar"
        >
          <Icon name="menu" size={24} />
        </button>
        <Link href="/" className="header-title-slot flex flex-col hover:opacity-80 transition-opacity">
          <h2 className="text-evergreen font-bold text-base leading-tight truncate max-w-50 md:max-w-none">
            {chatTitle}
          </h2>
          {hasMeta ? (
            <span className="text-pacific-blue/70 text-xs font-mono hidden md:flex items-center gap-2">
              {createdLabel && <span>{createdLabel}</span>}
              {msgCount != null && msgCount > 0 && (
                <>
                  <span className="text-pacific-blue/30">·</span>
                  <span>
                    {msgCount === 1
                      ? t("header.messageCountOne", { count: msgCount })
                      : t("header.messageCountOther", { count: msgCount })}
                  </span>
                </>
              )}
              {lang && (
                <>
                  <span className="text-pacific-blue/30">·</span>
                  <span>{lang}</span>
                </>
              )}
              {activeConversation?.pinned && (
                <>
                  <span className="text-pacific-blue/30">·</span>
                  <Icon name="push_pin" size={12} className="text-pacific-blue/60" />
                </>
              )}
            </span>
          ) : (
            <span className="text-pacific-blue text-xs font-mono hidden md:block">
              {t("header.activeSession")}
            </span>
          )}
        </Link>
      </div>

      <div className="flex items-center gap-3">
        {/* Export button — only when a conversation is active */}
        {activeConversation && (
          <button
            onClick={() => setShowExport(true)}
            className="flex items-center gap-1.5 text-evergreen hover:text-pacific-blue
                       transition-colors text-sm group min-h-11 min-w-11 justify-center"
            title={t("header.exportConversation")}
          >
            <Icon
              name="download"
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="hidden lg:inline">{t("common.export")}</span>
          </button>
        )}

        {/* Workspace sidebar toggle — only on chat page */}
        {onToggleWorkspaceSidebar && (
          <button
            type="button"
            onClick={onToggleWorkspaceSidebar}
            className="hidden md:flex items-center gap-1.5 text-evergreen hover:text-pacific-blue
                       transition-colors text-sm font-medium group min-h-11 min-w-11 justify-center"
            aria-label={t("chat.toggleWorkspaceSidebar")}
            title={t("chat.toggleWorkspaceSidebar")}
          >
            <Icon
              name={workspaceSidebarOpen ? "right_panel_close" : "right_panel_open"}
              size={18}
              className="group-hover:scale-110 transition-transform"
            />
            <span className="hidden lg:inline">{t("chat.workspace")}</span>
          </button>
        )}

        <nav className="flex items-center gap-2 md:gap-6" aria-label="Main navigation">
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            prefetch={false}
            className="flex items-center gap-1.5 text-evergreen hover:text-pacific-blue
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
          <button
            onClick={handleLogout}
            disabled={loggingOut}
            className="flex items-center gap-1.5 text-evergreen hover:text-burnt-tangerine transition-colors text-sm font-medium min-h-11 min-w-11 justify-center md:justify-start disabled:opacity-50"
            title={t("header.signOut")}
          >
            <Icon name="logout" size={18} />
            <span className="hidden sm:inline">{loggingOut ? t("header.signingOut") : t("header.signOut")}</span>
          </button>
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
