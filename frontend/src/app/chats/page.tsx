"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { Icon } from "@/components/Icon";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useConversationContext } from "@/components/LayoutShell";
import { useConversationBrowser } from "@/hooks/useConversationBrowser";
import { useI18n } from "@/hooks/useI18n";
import type { Conversation } from "@/lib/types";

export default function ChatsPage() {
  const { t, formatDate } = useI18n();
  const router = useRouter();
  const { revision, remove, rename, update, bulkDelete, bulkPin, doExport, setActiveId } =
    useConversationContext();
  const browser = useConversationBrowser(revision);
  const { items, total, offset, query, loading, pageSize } = browser;

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Drop selections that no longer exist on the current page (after a delete/refetch).
  useEffect(() => {
    setSelected((prev) => {
      const ids = new Set(items.map((c) => c.id));
      const next = new Set([...prev].filter((id) => ids.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [items]);

  const totalPages = Math.ceil(total / pageSize);
  const currentPage = Math.floor(offset / pageSize) + 1;

  // ── Search ─────────────────────────────────────────────────────────
  const onSearchChange = useCallback((v: string) => {
    browser.setQuery(v);
    setSelected(new Set()); // a new result set invalidates the old selection
  }, [browser]);

  // ── Selection ──────────────────────────────────────────────────────
  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const allVisibleSelected = items.length > 0 && items.every((c) => selected.has(c.id));
  const toggleSelectAllVisible = useCallback(() => {
    setSelected((prev) => {
      const next = new Set(prev);
      const everySelected = items.length > 0 && items.every((c) => next.has(c.id));
      if (everySelected) items.forEach((c) => next.delete(c.id));
      else items.forEach((c) => next.add(c.id));
      return next;
    });
  }, [items]);

  const clearSelection = useCallback(() => setSelected(new Set()), []);

  // ── Actions (context mutators bump revision → browser refetches; optimistic
  //    patches keep single-row edits instant in the meantime) ───────────
  const openChat = useCallback((id: string) => {
    setActiveId(id, items.find((c) => c.id === id) ?? null);
    router.push("/");
  }, [setActiveId, items, router]);

  const handleRename = useCallback(async (id: string, title: string) => {
    browser.patchItem(id, { title });
    return rename(id, title);
  }, [browser, rename]);

  const handlePin = useCallback((id: string, pinned: boolean) => {
    browser.patchItem(id, { pinned });
    update(id, { pinned });
  }, [browser, update]);

  const handleBulkDelete = useCallback(async () => {
    setConfirmBulkDelete(false);
    const ids = Array.from(selected);
    browser.removeItems(ids);
    clearSelection();
    await bulkDelete(ids);
  }, [browser, bulkDelete, selected, clearSelection]);

  const handleBulkPin = useCallback(async (pinned: boolean) => {
    const ids = Array.from(selected);
    ids.forEach((id) => browser.patchItem(id, { pinned }));
    clearSelection();
    await bulkPin(ids, pinned);
  }, [browser, bulkPin, selected, clearSelection]);

  const handleConfirmedDelete = useCallback(() => {
    const id = confirmDeleteId;
    setConfirmDeleteId(null);
    if (id) {
      browser.removeItems([id]);
      remove(id);
    }
  }, [confirmDeleteId, browser, remove]);

  return (
    <main className="flex-1 overflow-y-auto bg-background">
      <div className="max-w-5xl mx-auto px-4 py-6 md:px-8 md:py-8 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <Icon name="forum" size={28} className="text-foreground" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">{t("chats.title")}</h1>
              <p className="text-sm text-muted-foreground">{t("chats.subtitle")}</p>
            </div>
          </div>
          <button
            onClick={() => browser.refresh()}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                       bg-primary text-primary-foreground hover:bg-primary/90
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
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none"
          />
          <input
            type="search"
            placeholder={t("chats.searchPlaceholder")}
            value={query}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg border border-border
                       bg-surface text-foreground placeholder:text-muted-foreground text-sm
                       focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          />
        </div>

        {/* Bulk action bar */}
        {selected.size > 0 && (
          <div className="flex items-center gap-3 flex-wrap rounded-xl border border-border bg-surface-muted px-4 py-2.5">
            <span className="text-sm font-semibold text-foreground">
              {t("chats.selectedCount", { count: selected.size })}
            </span>
            <button
              onClick={toggleSelectAllVisible}
              className="text-xs text-primary hover:text-foreground transition-colors cursor-pointer"
            >
              {allVisibleSelected ? t("chats.clearSelection") : t("chats.selectAllVisible")}
            </button>
            <div className="flex items-center gap-2 ml-auto">
              <button
                onClick={() => handleBulkPin(true)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                           border border-border text-foreground hover:bg-surface-muted transition-colors cursor-pointer"
              >
                <Icon name="push_pin" size={14} />
                {t("chats.pinSelected")}
              </button>
              <button
                onClick={() => handleBulkPin(false)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                           border border-border text-foreground hover:bg-surface-muted transition-colors cursor-pointer"
              >
                <Icon name="keep_off" size={14} />
                {t("chats.unpinSelected")}
              </button>
              <button
                onClick={() => setConfirmBulkDelete(true)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                           border border-destructive/30 text-destructive hover:bg-destructive/10 transition-colors cursor-pointer"
              >
                <Icon name="delete_outline" size={14} />
                {t("chats.deleteSelected")}
              </button>
            </div>
          </div>
        )}

        {/* Table */}
        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-surface-muted text-foreground text-xs uppercase tracking-wider">
                <th className="px-4 py-3 w-10">
                  <input
                    type="checkbox"
                    checked={allVisibleSelected}
                    onChange={toggleSelectAllVisible}
                    aria-label={t("chats.selectAllVisible")}
                    className="cursor-pointer accent-primary align-middle"
                  />
                </th>
                <th className="px-4 py-3 text-left font-semibold">{t("chats.colTitle")}</th>
                <th className="px-4 py-3 text-left font-semibold hidden sm:table-cell w-28">{t("chats.colMessages")}</th>
                <th className="px-4 py-3 text-left font-semibold hidden md:table-cell w-40">{t("chats.colUpdated")}</th>
                <th className="px-4 py-3 w-12" />
              </tr>
            </thead>
            <tbody>
              {loading && items.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">
                    {query ? t("chats.searching") : t("chats.loading")}
                  </td>
                </tr>
              )}
              {!loading && items.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-14 text-center">
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                      <Icon name="forum" size={28} className="opacity-30" />
                      <span className="text-sm">
                        {query ? t("chats.noMatch") : t("chats.noChatsYet")}
                      </span>
                    </div>
                  </td>
                </tr>
              )}
              {items.map((c) => (
                <ChatRow
                  key={c.id}
                  conversation={c}
                  snippet={c.match_snippet ?? null}
                  selected={selected.has(c.id)}
                  onToggleSelect={toggleSelect}
                  onOpen={openChat}
                  onRename={handleRename}
                  onPin={handlePin}
                  onRequestDelete={setConfirmDeleteId}
                  onExport={doExport}
                  formatDate={formatDate}
                />
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination — applies to search results too */}
        {total > pageSize && (
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{t("chats.pageOfTotal", { page: currentPage, pages: totalPages, total })}</span>
            <div className="flex gap-2">
              <button
                onClick={browser.prevPage}
                disabled={offset === 0 || loading}
                className="px-3 py-1.5 rounded border border-border hover:bg-surface-muted
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                {t("chats.previous")}
              </button>
              <button
                onClick={browser.nextPage}
                disabled={offset + pageSize >= total || loading}
                className="px-3 py-1.5 rounded border border-border hover:bg-surface-muted
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
        onConfirm={handleConfirmedDelete}
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
    <tr className="border-t border-border/60 hover:bg-surface-muted/60 transition-colors">
      <td className="px-4 py-3 align-top">
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onToggleSelect(c.id)}
          aria-label={t("chats.selectRow")}
          className="mt-0.5 cursor-pointer accent-primary"
        />
      </td>
      <td className="px-4 py-3 max-w-xs lg:max-w-lg">
        <div className="flex items-start gap-2">
          {c.pinned && <Icon name="push_pin" size={14} className="text-primary/70 mt-0.5 shrink-0" />}
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
                className="w-full text-sm font-medium text-foreground bg-surface rounded px-1.5 py-0.5
                           border border-border outline-none focus:border-primary"
              />
            ) : (
              <button
                onClick={() => onOpen(c.id)}
                onDoubleClick={(e) => { e.preventDefault(); setEditValue(c.title); setEditing(true); }}
                className="text-left text-sm font-medium text-foreground hover:text-primary transition-colors truncate w-full cursor-pointer"
                title={t("chats.openChat")}
              >
                {c.title}
              </button>
            )}
            {secondary && (
              <p className="text-xs text-muted-foreground line-clamp-1 mt-0.5">{secondary}</p>
            )}
          </div>
        </div>
      </td>
      <td className="px-4 py-3 text-muted-foreground text-xs hidden sm:table-cell">
        {c.message_count ?? 0}
      </td>
      <td className="px-4 py-3 text-muted-foreground text-xs hidden md:table-cell">
        {c.updated_at ? formatDate(c.updated_at) : "—"}
      </td>
      <td className="px-4 py-3 text-right">
        <button
          ref={triggerRef}
          onClick={() => (menuOpen ? setMenuOpen(false) : openMenu())}
          className="p-1.5 rounded text-muted-foreground hover:text-foreground hover:bg-surface-muted transition-colors cursor-pointer"
          aria-label={t("sidebar.moreOptions")}
        >
          <Icon name="more_vert" size={16} />
        </button>
        {menuOpen && menuPos && createPortal(
          <div
            ref={menuRef}
            style={{ position: "fixed", top: menuPos.top, right: menuPos.right }}
            className="z-50 w-44 bg-surface border border-border rounded-lg shadow-xl py-1 text-sm text-foreground text-left"
          >
            <RowMenuItem icon="edit" label={t("sidebar.rename")}
              onClick={() => { setMenuOpen(false); setEditValue(c.title); setEditing(true); }} />
            <RowMenuItem icon={c.pinned ? "keep_off" : "push_pin"} label={c.pinned ? t("sidebar.unpin") : t("sidebar.pin")}
              onClick={() => { setMenuOpen(false); onPin(c.id, !c.pinned); }} />
            <div className="border-t border-border my-1" />
            <RowMenuItem icon="download" label={t("sidebar.exportJson")}
              onClick={() => { setMenuOpen(false); onExport(c.id, "json"); }} />
            <RowMenuItem icon="description" label={t("sidebar.exportMarkdown")}
              onClick={() => { setMenuOpen(false); onExport(c.id, "markdown"); }} />
            <div className="border-t border-border my-1" />
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
        ${danger ? "text-destructive hover:bg-destructive/10" : "hover:bg-surface-muted"}`}
    >
      <Icon name={icon} size={16} />
      <span>{label}</span>
    </button>
  );
}
