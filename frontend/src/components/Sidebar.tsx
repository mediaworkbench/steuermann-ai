"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";
import { Icon } from "./Icon";
import { ConfirmDialog } from "./ConfirmDialog";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { SINGLE_USER_DISPLAY_NAME } from "@/lib/runtime";
import { useRole } from "@/context/RoleContext";
import type { Conversation } from "@/lib/types";

/* ── Props ────────────────────────────────────────────────────────────── */

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  conversations?: Conversation[];
  activeId?: string | null;
  onSelect?: (id: string) => void;
  onNewChat?: () => void;
  onDelete?: (id: string) => Promise<boolean>;
  onPin?: (id: string, pinned: boolean) => void;
  onRename?: (id: string, title: string) => Promise<Conversation | null>;
  onExport?: (id: string, format: "json" | "markdown") => void;
}

// The sidebar is a lean quick-access list: all pinned conversations plus the
// 5 most-recent unpinned ones. Search and multi-select bulk management live on
// the dedicated /chats page (reachable via the "See all chats" link below).
const RECENT_LIMIT = 5;

export function Sidebar({
  isOpen,
  onClose,
  conversations = [],
  activeId,
  onSelect,
  onNewChat,
  onDelete,
  onPin,
  onRename,
  onExport,
}: SidebarProps) {
  const { t } = useI18n();
  const profile = useProfile();
  const { role } = useRole();
  const profileDisplayName = profile.appName || profile.displayName || "Steuermann";
  const profileAppName = profile.appName || "Steuermann";
  const frameworkVersion = profile.frameworkVersion || "unknown";

  // Close sidebar on Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Backend returns pinned first, then updated_at desc — so filtering keeps order.
  const pinned = conversations.filter((c) => c.pinned);
  const recent = conversations.filter((c) => !c.pinned).slice(0, RECENT_LIMIT);

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        id="sidebar"
        className={`
          fixed md:static inset-y-0 left-0 z-40
          w-70 lg:w-80 bg-sidebar-background flex flex-col h-full
          border-r border-sidebar-border shrink-0
          transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
        `}
      >
        {/* Logo / Title */}
        <div className="p-6 pb-2">
          <div className="flex items-center justify-between mb-6">
            <div className="flex flex-col">
              <div className="flex items-baseline gap-2">
                <h1 className="text-sidebar-foreground text-xl font-bold tracking-tight leading-tight">
                  {profileAppName}
                </h1>
                <span className="text-sidebar-muted text-xs font-mono">v{frameworkVersion}</span>
              </div>
              <span className="text-primary text-xs font-mono mt-1">
                {t("sidebar.platformSubtitle")}
              </span>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="md:hidden text-sidebar-muted hover:text-sidebar-foreground transition-colors -mr-1
                         min-h-11 min-w-11 flex items-center justify-center"
              onClick={onClose}
              aria-label={t("sidebar.closeNavigation")}
            >
              <Icon name="close" size={24} />
            </Button>
          </div>

          {/* New Chat button */}
          <Button
            type="button"
            onClick={() => { onNewChat?.(); }}
            variant="primary"
            className="w-full gap-2 rounded-lg font-bold min-h-11 shadow-lg shadow-primary/20 mb-2 group cursor-pointer"
            aria-label={t("sidebar.startNewChat")}
          >
            <Icon
              name="add"
              size={20}
              className="transition-transform group-hover:rotate-90"
            />
            <span>{t("sidebar.newChat")}</span>
          </Button>
        </div>

        {/* Conversation history */}
        <nav
          className="flex-1 overflow-y-auto px-4 pb-4 space-y-4"
          aria-label={t("sidebar.chatHistory")}
        >
          {/* Pinned conversations */}
          {pinned.length > 0 && (
            <div>
              <h3 className="text-sidebar-foreground text-xs font-bold uppercase tracking-wider mb-2 pl-3">
                {t("sidebar.pinned")}
              </h3>
              <div className="space-y-0.5">
                {pinned.map((c) => (
                  <ConversationRow
                    key={c.id}
                    conversation={c}
                    isActive={c.id === activeId}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onPin={onPin}
                    onRename={onRename}
                    onExport={onExport}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Recent conversations */}
          {recent.length > 0 && (
            <div>
              <h3 className="text-sidebar-foreground text-xs font-bold uppercase tracking-wider mb-2 pl-3">
                {t("sidebar.recentChats")}
              </h3>
              <div className="space-y-0.5">
                {recent.map((c) => (
                  <ConversationRow
                    key={c.id}
                    conversation={c}
                    isActive={c.id === activeId}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onPin={onPin}
                    onRename={onRename}
                    onExport={onExport}
                  />
                ))}
              </div>
            </div>
          )}

          {conversations.length === 0 && (
            <p className="text-sidebar-muted text-xs text-center mt-4">
              {t("sidebar.noConversations")}
            </p>
          )}

          {/* See all chats → /chats page (always available) */}
          <Link
            href="/chats"
            prefetch={false}
            onClick={onClose}
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                       text-sidebar-muted hover:text-sidebar-foreground hover:bg-sidebar-hover transition-colors
                       border border-sidebar-border"
          >
            <Icon name="forum" size={16} />
            <span>{t("sidebar.seeAllChats")}</span>
          </Link>
        </nav>

        {/* User profile */}
        <div className="p-4 border-t border-sidebar-border bg-sidebar-background space-y-1">
          <Link
            href="/settings"
            prefetch={false}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-sidebar-hover
                       text-sidebar-foreground transition-colors"
            aria-label={t("sidebar.settingsForUser", { name: profileDisplayName })}
          >
            <span className="flex flex-col min-w-0">
              <span className="text-sm font-bold truncate text-sidebar-foreground">
                {SINGLE_USER_DISPLAY_NAME}
              </span>
              <span className="text-xs text-sidebar-muted capitalize">
                {role}
              </span>
            </span>
            <Icon name="settings" className="ml-auto shrink-0 text-sidebar-muted" />
          </Link>
        </div>
      </aside>
    </>
  );
}

/* ── Single conversation row ────────────────────────────────────────── */

function ConversationRow({
  conversation: c,
  isActive,
  onSelect,
  onDelete,
  onPin,
  onRename,
  onExport,
}: {
  conversation: Conversation;
  isActive: boolean;
  onSelect?: (id: string) => void;
  onDelete?: (id: string) => Promise<boolean>;
  onPin?: (id: string, pinned: boolean) => void;
  onRename?: (id: string, title: string) => Promise<Conversation | null>;
  onExport?: (id: string, format: "json" | "markdown") => void;
}) {
  const { t } = useI18n();
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(c.title);
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Focus input when entering edit mode
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [menuOpen]);

  const commitRename = useCallback(async () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== c.title) {
      await onRename?.(c.id, trimmed);
    } else {
      setEditValue(c.title);
    }
    setEditing(false);
  }, [editValue, c.id, c.title, onRename]);

  return (
    <div className="relative">
          <Button
        onClick={() => onSelect?.(c.id)}
        onDoubleClick={(e) => {
          e.preventDefault();
          setEditValue(c.title);
          setEditing(true);
        }}
            variant="ghost"
            className={`
              w-full justify-start gap-2 px-3 py-2.5 rounded-lg
              transition-colors text-left group
              ${isActive
                ? "bg-sidebar-active text-sidebar-foreground hover:bg-sidebar-active"
                : "text-sidebar-muted hover:text-sidebar-foreground hover:bg-sidebar-hover"}
            `}
        aria-current={isActive ? "page" : undefined}
      >
        <Icon
          name={c.pinned ? "push_pin" : "chat_bubble_outline"}
          size={18}
          className={isActive ? "text-primary" : "text-sidebar-muted group-hover:text-sidebar-foreground"}
        />

        {/* Title: either input or text */}
        {editing ? (
          <Input
            ref={inputRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") { setEditValue(c.title); setEditing(false); }
            }}
            onClick={(e) => e.stopPropagation()}
            className="flex-1 min-w-0 text-sm font-medium bg-sidebar-active rounded px-1.5 py-0.5
                       border border-sidebar-border text-sidebar-foreground outline-none focus:border-primary"
          />
        ) : (
          <span className="text-sm font-medium truncate flex-1 min-w-0">
            {c.title}
          </span>
        )}

        {/* Three-dot context menu trigger (hover only) */}
        {!editing && (
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); setMenuOpen((v) => !v); }}
            onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); setMenuOpen((v) => !v); } }}
            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-sidebar-hover
                       text-sidebar-muted hover:text-sidebar-foreground transition-all cursor-pointer shrink-0"
            title={t("sidebar.moreOptions")}
          >
            <Icon name="more_vert" size={16} />
          </span>
        )}
      </Button>

      {/* ── Context dropdown menu ── */}
      {menuOpen && (
        <div
          ref={menuRef}
          className="absolute right-2 top-full mt-1 z-50 w-44 bg-surface border border-border
                     rounded-lg shadow-xl py-1 text-sm text-foreground animate-in fade-in slide-in-from-top-1 duration-150"
        >
          <ContextMenuItem
            icon="edit"
            label={t("sidebar.rename")}
            onClick={() => { setMenuOpen(false); setEditValue(c.title); setEditing(true); }}
          />
          <ContextMenuItem
            icon={c.pinned ? "push_pin" : "keep"}
            label={c.pinned ? t("sidebar.unpin") : t("sidebar.pin")}
            onClick={() => { setMenuOpen(false); onPin?.(c.id, !c.pinned); }}
          />
          <div className="border-t border-border my-1" />
          <ContextMenuItem
            icon="download"
            label={t("sidebar.exportJson")}
            onClick={() => { setMenuOpen(false); onExport?.(c.id, "json"); }}
          />
          <ContextMenuItem
            icon="description"
            label={t("sidebar.exportMarkdown")}
            onClick={() => { setMenuOpen(false); onExport?.(c.id, "markdown"); }}
          />
          <div className="border-t border-border my-1" />
          <ContextMenuItem
            icon="delete_outline"
            label={t("sidebar.delete")}
            danger
            onClick={() => {
              setMenuOpen(false);
              setConfirmDelete(true);
            }}
          />
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmDelete}
        title={t("sidebar.delete")}
        message={t("sidebar.deleteConversationConfirm")}
        variant="danger"
        confirmLabel={t("common.delete")}
        onConfirm={() => {
          setConfirmDelete(false);
          onDelete?.(c.id);
        }}
        onCancel={() => setConfirmDelete(false)}
      />
    </div>
  );
}

/* ── Context menu item ────────────────────────────────────────────────── */

function ContextMenuItem({
  icon,
  label,
  danger = false,
  onClick,
}: {
  icon: string;
  label: string;
  danger?: boolean;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      onClick={onClick}
      variant="ghost"
      size="sm"
      className={`w-full flex items-center gap-2.5 px-3 py-1.5 text-left transition-colors cursor-pointer
        ${danger
          ? "text-destructive/80 hover:bg-destructive/15 hover:text-destructive"
          : "hover:bg-surface-muted hover:text-foreground"
        }`}
    >
      <Icon name={icon} size={16} />
      <span>{label}</span>
    </Button>
  );
}
