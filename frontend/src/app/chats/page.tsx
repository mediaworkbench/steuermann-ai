"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { Icon } from "@/components/Icon";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useConversationContext } from "@/components/LayoutShell";
import { useI18n } from "@/hooks/useI18n";
import { searchConversations } from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type { Conversation } from "@/lib/types";

const PAGE_SIZE = 50;

/** Pinned first, then most-recently updated — mirrors the backend list order so a
 *  local pin/rename reflects immediately without a refetch. */
function sortConversations(list: Conversation[]): Conversation[] {
  return [...list].sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
    const ta = a.updated_at ? Date.parse(a.updated_at) : 0;
    const tb = b.updated_at ? Date.parse(b.updated_at) : 0;
    return tb - ta;
  });
}

export default function ChatsPage() {
  const { t, formatDate } = useI18n();
  const router = useRouter();
  const {
    conversations,
    loading,
    remove,
    rename,
    update,
    bulkDelete,
    bulkPin,
    doExport,
    setActiveId,
    refresh,
  } = useConversationContext();

  const [search, setSearch] = useState("");
  const [searching, setSearching] = useState(false);
  // Deduped set of conversation ids that matched the full-text search + best snippet per id.
  const [match, setMatch] = useState<{ ids: Set<string>; snippets: Map<string, string> } | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [offset, setOffset] = useState(0);
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchSeqRef = useRef(0);

  // ── Full-text search (debounced) — narrows the list to matching chats ──
  useEffect(() => {
    const q = search.trim();
    // Bump the sequence so any in-flight request for a previous (or cleared)
    // query is ignored when it finally resolves.
    const seq = ++searchSeqRef.current;
    if (!q) {
      setMatch(null);
      setSearching(false);
      return;
    }
    setSearching(true);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(async () => {
      const results = await searchConversations(CURRENT_USER_ID, q, 200);
      if (seq !== searchSeqRef.current) return; // superseded by a newer query
      const ids = new Set<string>();
      const snippets = new Map<string, string>();
      for (const r of results) {
        ids.add(r.conversation_id);
        if (!snippets.has(r.conversation_id)) snippets.set(r.conversation_id, r.content);
      }
      setMatch({ ids, snippets });
      setSearching(false);
    }, 350);
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, [search]);

  // Reset pagination whenever the active filter changes.
  useEffect(() => {
    setOffset(0);
  }, [search]);

  // Drop selections that no longer exist (e.g. after a delete).
  useEffect(() => {
    setSelected((prev) => {
      const ids = new Set(conversations.map((c) => c.id));
      const next = new Set([...prev].filter((id) => ids.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [conversations]);

  const isSearching = search.trim().length > 0;

  const filtered = useMemo(() => {
    const base = match ? conversations.filter((c) => match.ids.has(c.id)) : conversations;
    return sortConversations(base);
  }, [conversations, match]);

  const pageItems = isSearching ? filtered : filtered.slice(offset, offset + PAGE_SIZE);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  // ── Selection ──────────────────────────────────────────────────────
  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const allVisibleSelected = pageItems.length > 0 && pageItems.every((c) => selected.has(c.id));
  const toggleSelectAllVisible = useCallback(() => {
    setSelected((prev) => {
      const next = new Set(prev);
      const everySelected = pageItems.length > 0 && pageItems.every((c) => next.has(c.id));
      if (everySelected) pageItems.forEach((c) => next.delete(c.id));
      else pageItems.forEach((c) => next.add(c.id));
      return next;
    });
  }, [pageItems]);

  const clearSelection = useCallback(() => setSelected(new Set()), []);

  const openChat = useCallback((id: string) => {
    setActiveId(id);
    router.push("/");
  }, [setActiveId, router]);

  const handleBulkDelete = useCallback(async () => {
    setConfirmBulkDelete(false);
    await bulkDelete(Array.from(selected));
    clearSelection();
  }, [bulkDelete, selected, clearSelection]);

  const handleBulkPin = useCallback(async (pinned: boolean) => {
    await bulkPin(Array.from(selected), pinned);
    clearSelection();
  }, [bulkPin, selected, clearSelection]);

  return (
    <main className="flex-1 overflow-y-auto bg-white">
      <div className="max-w-5xl mx-auto px-4 py-6 md:px-8 md:py-8 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <Icon name="forum" size={28} className="text-evergreen" />
            <div>
              <h1 className="text-2xl font-bold text-evergreen">{t("chats.title")}</h1>
              <p className="text-sm text-evergreen/60">{t("chats.subtitle")}</p>
            </div>
          </div>
          <button
            onClick={() => refresh()}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                       bg-evergreen text-light-cyan hover:bg-evergreen/80
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <Icon name="refresh" size={14} className={loading ? "animate-spin" : ""} />
            {t("chats.refresh")}
          </button>
        </div>

        {/* Search */}
        <div className="relative max-w-xl">
          <Icon
            name="search"
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-evergreen/40 pointer-events-none"
          />
          <input
            type="search"
            placeholder={t("chats.searchPlaceholder")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg border border-evergreen/20
                       text-evergreen placeholder-evergreen/30 text-sm
                       focus:outline-none focus:ring-2 focus:ring-evergreen/30"
          />
        </div>

        {/* Bulk action bar */}
        {selected.size > 0 && (
          <div className="flex items-center gap-3 flex-wrap rounded-xl border border-evergreen/15 bg-evergreen/5 px-4 py-2.5">
            <span className="text-sm font-semibold text-evergreen">
              {t("chats.selectedCount", { count: selected.size })}
            </span>
            <button
              onClick={toggleSelectAllVisible}
              className="text-xs text-pacific-blue hover:text-evergreen transition-colors cursor-pointer"
            >
              {allVisibleSelected ? t("chats.clearSelection") : t("chats.selectAllVisible")}
            </button>
            <div className="flex items-center gap-2 ml-auto">
              <button
                onClick={() => handleBulkPin(true)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                           border border-evergreen/20 text-evergreen hover:bg-evergreen/10 transition-colors cursor-pointer"
              >
                <Icon name="push_pin" size={14} />
                {t("chats.pinSelected")}
              </button>
              <button
                onClick={() => handleBulkPin(false)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                           border border-evergreen/20 text-evergreen hover:bg-evergreen/10 transition-colors cursor-pointer"
              >
                <Icon name="keep_off" size={14} />
                {t("chats.unpinSelected")}
              </button>
              <button
                onClick={() => setConfirmBulkDelete(true)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                           border border-red-300 text-red-600 hover:bg-red-50 transition-colors cursor-pointer"
              >
                <Icon name="delete_outline" size={14} />
                {t("chats.deleteSelected")}
              </button>
            </div>
          </div>
        )}

        {/* Table */}
        <div className="rounded-xl border border-evergreen/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-evergreen text-light-cyan text-xs uppercase tracking-wider">
                <th className="px-4 py-3 w-10">
                  <input
                    type="checkbox"
                    checked={allVisibleSelected}
                    onChange={toggleSelectAllVisible}
                    aria-label={t("chats.selectAllVisible")}
                    className="cursor-pointer accent-pacific-blue align-middle"
                  />
                </th>
                <th className="px-4 py-3 text-left font-semibold">{t("chats.colTitle")}</th>
                <th className="px-4 py-3 text-left font-semibold hidden sm:table-cell w-28">{t("chats.colMessages")}</th>
                <th className="px-4 py-3 text-left font-semibold hidden md:table-cell w-40">{t("chats.colUpdated")}</th>
                <th className="px-4 py-3 w-12" />
              </tr>
            </thead>
            <tbody>
              {(loading || searching) && pageItems.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-evergreen/40">
                    {searching ? t("chats.searching") : t("chats.loading")}
                  </td>
                </tr>
              )}
              {!loading && !searching && pageItems.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-14 text-center">
                    <div className="flex flex-col items-center gap-2 text-evergreen/40">
                      <Icon name="forum" size={28} className="opacity-30" />
                      <span className="text-sm">
                        {isSearching ? t("chats.noMatch") : t("chats.noChatsYet")}
                      </span>
                    </div>
                  </td>
                </tr>
              )}
              {pageItems.map((c) => (
                <ChatRow
                  key={c.id}
                  conversation={c}
                  snippet={match?.snippets.get(c.id) ?? null}
                  selected={selected.has(c.id)}
                  onToggleSelect={toggleSelect}
                  onOpen={openChat}
                  onRename={rename}
                  onPin={(id, pinned) => update(id, { pinned })}
                  onRequestDelete={setConfirmDeleteId}
                  onExport={doExport}
                  formatDate={formatDate}
                />
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination (hidden while searching) */}
        {!isSearching && totalPages > 1 && (
          <div className="flex items-center justify-between text-sm text-evergreen/50">
            <span>{t("chats.pageOfTotal", { page: currentPage, pages: totalPages, total: filtered.length })}</span>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                disabled={offset === 0}
                className="px-3 py-1.5 rounded border border-evergreen/20 hover:bg-evergreen/5
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                {t("chats.previous")}
              </button>
              <button
                onClick={() => setOffset((o) => o + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= filtered.length}
                className="px-3 py-1.5 rounded border border-evergreen/20 hover:bg-evergreen/5
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                {t("chats.next")}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Bulk delete confirmation */}
      <ConfirmDialog
        isOpen={confirmBulkDelete}
        title={t("chats.deleteSelected")}
        message={t("chats.deleteSelectedConfirm", { count: selected.size })}
        variant="danger"
        confirmLabel={t("common.delete")}
        onConfirm={handleBulkDelete}
        onCancel={() => setConfirmBulkDelete(false)}
      />

      {/* Single delete confirmation */}
      <ConfirmDialog
        isOpen={confirmDeleteId !== null}
        title={t("sidebar.delete")}
        message={t("sidebar.deleteConversationConfirm")}
        variant="danger"
        confirmLabel={t("common.delete")}
        onConfirm={() => {
          const id = confirmDeleteId;
          setConfirmDeleteId(null);
          if (id) remove(id);
        }}
        onCancel={() => setConfirmDeleteId(null)}
      />
    </main>
  );
}

/* ── Single chat row ──────────────────────────────────────────────────── */

function ChatRow({
  conversation: c,
  snippet,
  selected,
  onToggleSelect,
  onOpen,
  onRename,
  onPin,
  onRequestDelete,
  onExport,
  formatDate,
}: {
  conversation: Conversation;
  snippet: string | null;
  selected: boolean;
  onToggleSelect: (id: string) => void;
  onOpen: (id: string) => void;
  onRename: (id: string, title: string) => Promise<Conversation | null>;
  onPin: (id: string, pinned: boolean) => void;
  onRequestDelete: (id: string) => void;
  onExport: (id: string, format: "json" | "markdown") => void;
  formatDate: (value: string) => string;
}) {
  const { t } = useI18n();
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(c.title);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<{ top: number; right: number } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  // Position the portal menu under the trigger; close on outside click / scroll / resize.
  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    const onClick = (e: MouseEvent) => {
      if (
        menuRef.current && !menuRef.current.contains(e.target as Node) &&
        triggerRef.current && !triggerRef.current.contains(e.target as Node)
      ) close();
    };
    document.addEventListener("mousedown", onClick);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("mousedown", onClick);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [menuOpen]);

  const openMenu = () => {
    const r = triggerRef.current?.getBoundingClientRect();
    if (r) setMenuPos({ top: r.bottom + 4, right: window.innerWidth - r.right });
    setMenuOpen(true);
  };

  const commitRename = useCallback(async () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== c.title) await onRename(c.id, trimmed);
    else setEditValue(c.title);
    setEditing(false);
  }, [editValue, c.id, c.title, onRename]);

  const secondary = snippet ?? c.last_message ?? null;

  return (
    <tr className="border-t border-evergreen/8 hover:bg-evergreen/3 transition-colors">
      <td className="px-4 py-3 align-top">
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onToggleSelect(c.id)}
          aria-label={t("chats.selectRow")}
          className="mt-0.5 cursor-pointer accent-pacific-blue"
        />
      </td>
      <td className="px-4 py-3 max-w-xs lg:max-w-lg">
        <div className="flex items-start gap-2">
          {c.pinned && <Icon name="push_pin" size={14} className="text-pacific-blue/70 mt-0.5 shrink-0" />}
          <div className="min-w-0 flex-1">
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
                className="w-full text-sm font-medium text-evergreen bg-white rounded px-1.5 py-0.5
                           border border-pacific-blue/50 outline-none focus:border-pacific-blue"
              />
            ) : (
              <button
                onClick={() => onOpen(c.id)}
                onDoubleClick={(e) => { e.preventDefault(); setEditValue(c.title); setEditing(true); }}
                className="text-left text-sm font-medium text-evergreen hover:text-pacific-blue transition-colors truncate w-full cursor-pointer"
                title={t("chats.openChat")}
              >
                {c.title}
              </button>
            )}
            {secondary && (
              <p className="text-xs text-evergreen/40 line-clamp-1 mt-0.5">{secondary}</p>
            )}
          </div>
        </div>
      </td>
      <td className="px-4 py-3 text-evergreen/50 text-xs hidden sm:table-cell">
        {c.message_count ?? 0}
      </td>
      <td className="px-4 py-3 text-evergreen/40 text-xs hidden md:table-cell">
        {c.updated_at ? formatDate(c.updated_at) : "—"}
      </td>
      <td className="px-4 py-3 text-right">
        <button
          ref={triggerRef}
          onClick={() => (menuOpen ? setMenuOpen(false) : openMenu())}
          className="p-1.5 rounded text-evergreen/40 hover:text-evergreen hover:bg-evergreen/10 transition-colors cursor-pointer"
          aria-label={t("sidebar.moreOptions")}
        >
          <Icon name="more_vert" size={16} />
        </button>
        {menuOpen && menuPos && createPortal(
          <div
            ref={menuRef}
            style={{ position: "fixed", top: menuPos.top, right: menuPos.right }}
            className="z-50 w-44 bg-white border border-evergreen/15 rounded-lg shadow-xl py-1 text-sm text-evergreen/90 text-left"
          >
            <RowMenuItem icon="edit" label={t("sidebar.rename")}
              onClick={() => { setMenuOpen(false); setEditValue(c.title); setEditing(true); }} />
            <RowMenuItem icon={c.pinned ? "keep_off" : "push_pin"} label={c.pinned ? t("sidebar.unpin") : t("sidebar.pin")}
              onClick={() => { setMenuOpen(false); onPin(c.id, !c.pinned); }} />
            <div className="border-t border-evergreen/10 my-1" />
            <RowMenuItem icon="download" label={t("sidebar.exportJson")}
              onClick={() => { setMenuOpen(false); onExport(c.id, "json"); }} />
            <RowMenuItem icon="description" label={t("sidebar.exportMarkdown")}
              onClick={() => { setMenuOpen(false); onExport(c.id, "markdown"); }} />
            <div className="border-t border-evergreen/10 my-1" />
            <RowMenuItem icon="delete_outline" label={t("sidebar.delete")} danger
              onClick={() => { setMenuOpen(false); onRequestDelete(c.id); }} />
          </div>,
          document.body,
        )}
      </td>
    </tr>
  );
}

function RowMenuItem({
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
        ${danger ? "text-red-600 hover:bg-red-50" : "hover:bg-evergreen/5"}`}
    >
      <Icon name={icon} size={16} />
      <span>{label}</span>
    </button>
  );
}
