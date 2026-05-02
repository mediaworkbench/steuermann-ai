"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";
import { Icon } from "./Icon";
import { searchConversations } from "@/lib/api";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import {
  CURRENT_USER_ID,
} from "@/lib/runtime";
import type { Conversation, SearchResult } from "@/lib/types";

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
  onArchive?: (id: string, archived: boolean) => void;
  onExport?: (id: string, format: "json" | "markdown") => void;
  onBulkDelete?: (ids: string[]) => Promise<void>;
  onBulkArchive?: (ids: string[]) => Promise<void>;
  showArchived?: boolean;
  onToggleArchived?: () => void;
}

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
  onArchive,
  onExport,
  onBulkDelete,
  onBulkArchive,
  showArchived = false,
  onToggleArchived,
}: SidebarProps) {
  const { t, formatDate } = useI18n();
  const profile = useProfile();
  const profileDisplayName = profile.appName || profile.displayName || "Steuermann";
  const profileAppName = profile.appName || "Steuermann";

  const [bulkMode, setBulkMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Close sidebar on Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (searchQuery) { setSearchQuery(""); setSearchResults(null); }
        else if (bulkMode) { setBulkMode(false); setSelected(new Set()); }
        else onClose();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose, bulkMode, searchQuery]);

  // Exit bulk mode when conversations change
  useEffect(() => {
    if (bulkMode) setSelected(new Set());
  }, [conversations, bulkMode]);

  // Debounced deep search (API-backed)
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      setSearching(false);
      return;
    }
    setSearching(true);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(async () => {
      const results = await searchConversations(CURRENT_USER_ID, searchQuery.trim(), 20);
      setSearchResults(results);
      setSearching(false);
    }, 350);
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [searchQuery]);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    setSelected(new Set(conversations.map((c) => c.id)));
  };

  const pinned = conversations.filter((c) => c.pinned && !c.archived);
  const recent = conversations.filter((c) => !c.pinned && !c.archived);
  const archived = conversations.filter((c) => c.archived);

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
          w-70 lg:w-80 bg-evergreen flex flex-col h-full
          border-r border-white/10 shrink-0
          transition-transform duration-300 ease-in-out
          ${isOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0
        `}
      >
        {/* Logo / Title */}
        <div className="p-6 pb-2">
          <div className="flex items-center justify-between mb-6">
            <div className="flex flex-col">
              <h1 className="text-light-cyan text-xl font-bold tracking-tight leading-tight">
                {profileAppName}
              </h1>
              <span className="text-pacific-blue text-xs font-mono mt-1">
                {t("sidebar.platformSubtitle")}
              </span>
            </div>
            <button
              className="md:hidden text-light-cyan/70 hover:text-light-cyan transition-colors -mr-1
                         min-h-11 min-w-11 flex items-center justify-center"
              onClick={onClose}
              aria-label={t("sidebar.closeNavigation")}
            >
              <Icon name="close" size={24} />
            </button>
          </div>

          {/* New Chat button */}
          <button
            onClick={() => { onNewChat?.(); }}
            className="w-full flex items-center justify-center gap-2 bg-atomic-tangerine hover:bg-burnt-tangerine
                       text-white transition-colors py-3 px-4 rounded-lg font-bold min-h-11
                       shadow-lg shadow-atomic-tangerine/20 mb-4 group cursor-pointer"
            aria-label={t("sidebar.startNewChat")}
          >
            <Icon
              name="add"
              size={20}
              className="transition-transform group-hover:rotate-90"
            />
            <span>{t("sidebar.newChat")}</span>
          </button>

          {/* Toolbar: bulk mode toggle + show archived */}
          <div className="flex items-center justify-between mb-2 px-1">
            <button
              onClick={() => { setBulkMode((v) => !v); setSelected(new Set()); }}
              className={`text-xs flex items-center gap-1 px-2 py-1 rounded transition-colors cursor-pointer
                ${bulkMode ? "bg-pacific-blue/30 text-light-cyan" : "text-light-cyan/50 hover:text-light-cyan/80"}`}
              title={bulkMode ? t("sidebar.exitBulkMode") : t("sidebar.selectMultiple")}
            >
              <Icon name="checklist" size={14} />
              <span className="hidden lg:inline">{bulkMode ? t("sidebar.cancelSelection") : t("sidebar.select")}</span>
            </button>
            <button
              onClick={onToggleArchived}
              className={`text-xs flex items-center gap-1 px-2 py-1 rounded transition-colors cursor-pointer
                ${showArchived ? "bg-pacific-blue/30 text-light-cyan" : "text-light-cyan/50 hover:text-light-cyan/80"}`}
              title={showArchived ? t("sidebar.hideArchived") : t("sidebar.showArchived")}
            >
              <Icon name="inventory_2" size={14} />
              <span className="hidden lg:inline">{t("sidebar.archived")}</span>
            </button>
          </div>
        </div>

        {/* ── Search bar ── */}
        <div className="px-4 mb-2">
          <div className="relative">
            <Icon
              name="search"
              size={16}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-light-cyan/40 pointer-events-none"
            />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={t("sidebar.searchConversations")}
              className="w-full bg-white/10 text-light-cyan text-sm rounded-lg pl-8 pr-8 py-2
                         border border-transparent placeholder-light-cyan/30
                         focus:border-pacific-blue/50 focus:outline-none transition-colors"
              aria-label={t("sidebar.searchConversations")}
            />
            {searchQuery && (
              <button
                onClick={() => { setSearchQuery(""); setSearchResults(null); }}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-light-cyan/40 hover:text-light-cyan
                           transition-colors cursor-pointer"
                aria-label={t("sidebar.clearSearch")}
              >
                <Icon name="close" size={14} />
              </button>
            )}
          </div>
        </div>

        {/* ── Search results ── */}
        {searchQuery.trim() && (
          <div className="flex-1 overflow-y-auto px-4 pb-4">
            {searching ? (
              <p className="text-light-cyan/40 text-xs text-center mt-4">{t("sidebar.searching")}</p>
            ) : searchResults && searchResults.length > 0 ? (
              <div className="space-y-1">
                <h3 className="text-light-cyan text-xs font-bold uppercase tracking-wider mb-2 pl-3">
                  {t("sidebar.results", { count: searchResults.length })}
                </h3>
                {searchResults.map((r) => (
                  <button
                    key={r.message_id}
                    onClick={() => {
                      onSelect?.(r.conversation_id);
                      setSearchQuery("");
                      setSearchResults(null);
                    }}
                    className="w-full flex flex-col gap-0.5 px-3 py-2 rounded-lg text-left
                               text-light-cyan/80 hover:text-white hover:bg-white/10 transition-colors"
                  >
                    <span className="text-xs font-bold text-pacific-blue truncate">
                      {r.conversation_title}
                    </span>
                    <span className="text-xs text-light-cyan/60 line-clamp-2">
                      {r.content}
                    </span>
                    <span className="text-[10px] text-light-cyan/30 font-mono">
                      {r.role} · {formatDate(r.created_at)}
                    </span>
                  </button>
                ))}
              </div>
            ) : searchResults ? (
              <p className="text-light-cyan/40 text-xs text-center mt-4">
                {t("sidebar.noResultsFor", { query: searchQuery })}
              </p>
            ) : null}
          </div>
        )}

        {/* ── Bulk toolbar ── */}
        {!searchQuery.trim() && bulkMode && selected.size > 0 && (
          <div className="mx-4 mb-2 flex items-center gap-2 bg-white/10 rounded-lg px-3 py-2">
            <span className="text-light-cyan text-xs font-bold flex-1">
              {t("sidebar.selectedCount", { count: selected.size })}
            </span>
            <button
              onClick={selectAll}
              className="text-xs text-pacific-blue hover:text-light-cyan transition-colors cursor-pointer"
            >
              {t("sidebar.selectAll")}
            </button>
            <button
              onClick={async () => { if (onBulkArchive) { await onBulkArchive(Array.from(selected)); setSelected(new Set()); setBulkMode(false); } }}
              className="p-1 rounded hover:bg-white/20 text-light-cyan/70 hover:text-light-cyan transition-colors cursor-pointer"
              title={t("sidebar.archiveSelected")}
            >
              <Icon name="inventory_2" size={16} />
            </button>
            <button
              onClick={async () => {
                if (onBulkDelete && window.confirm(t("sidebar.deleteSelectedConfirm", { count: selected.size }))) {
                  await onBulkDelete(Array.from(selected));
                  setSelected(new Set());
                  setBulkMode(false);
                }
              }}
              className="p-1 rounded hover:bg-red-500/30 text-red-300/70 hover:text-red-300 transition-colors cursor-pointer"
              title={t("sidebar.deleteSelected")}
            >
              <Icon name="delete_outline" size={16} />
            </button>
          </div>
        )}

        {/* Conversation history (hidden when search is active) */}
        {!searchQuery.trim() && (
        <nav
          className="flex-1 overflow-y-auto px-4 pb-4 space-y-4"
          aria-label={t("sidebar.chatHistory")}
        >
          {/* Pinned conversations */}
          {pinned.length > 0 && (
            <div>
              <h3 className="text-light-cyan text-xs font-bold uppercase tracking-wider mb-2 pl-3">
                {t("sidebar.pinned")}
              </h3>
              <div className="space-y-0.5">
                {pinned.map((c) => (
                  <ConversationRow
                    key={c.id}
                    conversation={c}
                    isActive={c.id === activeId}
                    bulkMode={bulkMode}
                    isSelected={selected.has(c.id)}
                    onToggleSelect={toggleSelect}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onPin={onPin}
                    onRename={onRename}
                    onArchive={onArchive}
                    onExport={onExport}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Recent conversations */}
          {recent.length > 0 && (
            <div>
              <h3 className="text-light-cyan text-xs font-bold uppercase tracking-wider mb-2 pl-3">
                {t("sidebar.recentChats")}
              </h3>
              <div className="space-y-0.5">
                {recent.map((c) => (
                  <ConversationRow
                    key={c.id}
                    conversation={c}
                    isActive={c.id === activeId}
                    bulkMode={bulkMode}
                    isSelected={selected.has(c.id)}
                    onToggleSelect={toggleSelect}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onPin={onPin}
                    onRename={onRename}
                    onArchive={onArchive}
                    onExport={onExport}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Archived conversations */}
          {showArchived && archived.length > 0 && (
            <div>
              <h3 className="text-light-cyan text-xs font-bold uppercase tracking-wider mb-2 pl-3">
                {t("sidebar.archived")}
              </h3>
              <div className="space-y-0.5">
                {archived.map((c) => (
                  <ConversationRow
                    key={c.id}
                    conversation={c}
                    isActive={c.id === activeId}
                    bulkMode={bulkMode}
                    isSelected={selected.has(c.id)}
                    onToggleSelect={toggleSelect}
                    onSelect={onSelect}
                    onDelete={onDelete}
                    onPin={onPin}
                    onRename={onRename}
                    onArchive={onArchive}
                    onExport={onExport}
                  />
                ))}
              </div>
            </div>
          )}

          {conversations.length === 0 && (
            <p className="text-light-cyan/40 text-xs text-center mt-4">
              {t("sidebar.noConversations")}
            </p>
          )}
        </nav>
        )}

        {/* User profile */}
        <div className="p-4 border-t border-white/10 bg-evergreen space-y-1">
          <Link
            href="/memories"
            prefetch={false}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/10
                       text-light-cyan transition-colors"
            aria-label="Memory"
          >
            <Icon name="psychology" className="text-light-cyan/70" />
            <span className="text-sm font-medium text-light-cyan">Memory</span>
          </Link>
          <Link
            href="/settings"
            prefetch={false}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/10
                       text-light-cyan transition-colors"
            aria-label={t("sidebar.settingsForUser", { name: profileDisplayName })}
          >
            <span className="text-sm font-bold truncate block w-full min-w-0 text-light-cyan">
              {profileDisplayName}
            </span>
            <Icon name="settings" className="ml-auto text-light-cyan/70" />
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
  bulkMode,
  isSelected,
  onToggleSelect,
  onSelect,
  onDelete,
  onPin,
  onRename,
  onArchive,
  onExport,
}: {
  conversation: Conversation;
  isActive: boolean;
  bulkMode: boolean;
  isSelected: boolean;
  onToggleSelect: (id: string) => void;
  onSelect?: (id: string) => void;
  onDelete?: (id: string) => Promise<boolean>;
  onPin?: (id: string, pinned: boolean) => void;
  onRename?: (id: string, title: string) => Promise<Conversation | null>;
  onArchive?: (id: string, archived: boolean) => void;
  onExport?: (id: string, format: "json" | "markdown") => void;
}) {
  const { t } = useI18n();
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(c.title);
  const [menuOpen, setMenuOpen] = useState(false);
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

  const handleClick = () => {
    if (bulkMode) { onToggleSelect(c.id); return; }
    onSelect?.(c.id);
  };

  return (
    <div className="relative">
      <button
        onClick={handleClick}
        onDoubleClick={(e) => {
          if (bulkMode) return;
          e.preventDefault();
          setEditValue(c.title);
          setEditing(true);
        }}
        className={`
          w-full flex items-center gap-2 px-3 py-2.5 rounded-lg
          transition-colors text-left group
          ${isActive
            ? "bg-white/10 text-white hover:bg-white/20"
            : "text-light-cyan/80 hover:text-white hover:bg-white/10"}
          ${isSelected ? "ring-1 ring-pacific-blue/60" : ""}
        `}
        aria-current={isActive ? "page" : undefined}
      >
        {/* Checkbox for bulk mode */}
        {bulkMode && (
          <span className={`w-4 h-4 rounded border flex items-center justify-center shrink-0
            ${isSelected ? "bg-pacific-blue border-pacific-blue" : "border-light-cyan/40"}`}>
            {isSelected && <Icon name="check" size={12} className="text-white" />}
          </span>
        )}

        <Icon
          name={c.archived ? "inventory_2" : c.pinned ? "push_pin" : "chat_bubble_outline"}
          size={18}
          className={isActive ? "text-light-cyan" : "text-light-cyan/60 group-hover:text-light-cyan"}
        />

        {/* Title: either input or text */}
        {editing ? (
          <input
            ref={inputRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") { setEditValue(c.title); setEditing(false); }
            }}
            onClick={(e) => e.stopPropagation()}
            className="flex-1 min-w-0 text-sm font-medium bg-white/15 text-white rounded px-1.5 py-0.5
                       border border-pacific-blue/50 outline-none focus:border-pacific-blue"
          />
        ) : (
          <span className="text-sm font-medium truncate flex-1 min-w-0">
            {c.title}
          </span>
        )}

        {/* Three-dot context menu trigger (hover only, not in bulk) */}
        {!bulkMode && !editing && (
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); setMenuOpen((v) => !v); }}
            onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); setMenuOpen((v) => !v); } }}
            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-white/20
                       text-light-cyan/70 hover:text-light-cyan transition-all cursor-pointer shrink-0"
            title={t("sidebar.moreOptions")}
          >
            <Icon name="more_vert" size={16} />
          </span>
        )}
      </button>

      {/* ── Context dropdown menu ── */}
      {menuOpen && (
        <div
          ref={menuRef}
          className="absolute right-2 top-full mt-1 z-50 w-44 bg-[#0a3839] border border-white/15
                     rounded-lg shadow-xl py-1 text-sm text-light-cyan/90 animate-in fade-in slide-in-from-top-1 duration-150"
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
          <ContextMenuItem
            icon={c.archived ? "unarchive" : "inventory_2"}
            label={c.archived ? t("sidebar.unarchive") : t("sidebar.archive")}
            onClick={() => { setMenuOpen(false); onArchive?.(c.id, !c.archived); }}
          />
          <div className="border-t border-white/10 my-1" />
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
          <div className="border-t border-white/10 my-1" />
          <ContextMenuItem
            icon="delete_outline"
            label={t("sidebar.delete")}
            danger
            onClick={() => {
              setMenuOpen(false);
              if (window.confirm(t("sidebar.deleteConversationConfirm"))) onDelete?.(c.id);
            }}
          />
        </div>
      )}
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
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2.5 px-3 py-1.5 text-left transition-colors cursor-pointer
        ${danger
          ? "text-red-300/80 hover:bg-red-500/20 hover:text-red-300"
          : "hover:bg-white/10 hover:text-white"
        }`}
    >
      <Icon name={icon} size={16} />
      <span>{label}</span>
    </button>
  );
}
