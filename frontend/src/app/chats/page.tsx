"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  BookmarkMinus,
  Download,
  MessageSquare,
  MoreVertical,
  PanelRightOpen,
  Pencil,
  Pin,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { ExportDialog } from "@/components/ExportDialog";
import { ConversationEvidenceDrawer } from "@/components/workspace/ConversationEvidenceDrawer";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useConversationContext } from "@/components/LayoutShell";
import { useConversationBrowser } from "@/hooks/useConversationBrowser";
import { useI18n } from "@/hooks/useI18n";
import type { Conversation } from "@/lib/types";

export default function ChatsPage() {
  const { t, formatDate } = useI18n();
  const router = useRouter();
  const { revision, remove, rename, update, bulkDelete, bulkPin, setActiveId } =
    useConversationContext();
  const browser = useConversationBrowser(revision);
  const { items, total, offset, query, loading, pageSize } = browser;

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [evidenceTarget, setEvidenceTarget] = useState<{ id: string; title: string } | null>(null);

  useEffect(() => {
    setSelected((prev) => {
      const ids = new Set(items.map((c) => c.id));
      const next = new Set([...prev].filter((id) => ids.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [items]);

  const totalPages = Math.ceil(total / pageSize);
  const currentPage = Math.floor(offset / pageSize) + 1;

  const onSearchChange = useCallback((v: string) => {
    browser.setQuery(v);
    setSelected(new Set());
  }, [browser]);

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

  const openChat = useCallback((id: string) => {
    setActiveId(id, items.find((c) => c.id === id) ?? null);
    router.push("/");
  }, [setActiveId, items, router]);

  const openEvidence = useCallback((id: string, title: string) => {
    setEvidenceTarget({ id, title });
  }, []);

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
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-6 lg:px-12">

        {/* Header */}
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-3xl font-bold text-foreground">{t("chats.title")}</h1>
            <p className="mt-1 text-muted-foreground">{t("chats.subtitle")}</p>
          </div>
          <Button
            variant="default"
            size="sm"
            onClick={() => browser.refresh()}
            disabled={loading}
            className="gap-1.5 shrink-0"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            {t("chats.refresh")}
          </Button>
        </div>

        {/* Search */}
        <div className="relative max-w-xl">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none z-10"
          />
          <Input
            type="search"
            placeholder={t("chats.searchPlaceholder")}
            value={query}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-9"
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
              <Button variant="outline" size="sm" onClick={() => handleBulkPin(true)} className="gap-1.5">
                <Pin size={14} />
                {t("chats.pinSelected")}
              </Button>
              <Button variant="outline" size="sm" onClick={() => handleBulkPin(false)} className="gap-1.5">
                <BookmarkMinus size={14} />
                {t("chats.unpinSelected")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setConfirmBulkDelete(true)}
                className="gap-1.5 text-destructive border-destructive/30 hover:bg-destructive/10"
              >
                <Trash2 size={14} />
                {t("chats.deleteSelected")}
              </Button>
            </div>
          </div>
        )}

        {/* Table */}
        {loading && items.length === 0 && (
          <div className="text-center py-10 text-muted-foreground">
            {query ? t("chats.searching") : t("chats.loading")}
          </div>
        )}

        {!loading && items.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-14 text-muted-foreground text-center">
            <MessageSquare size={28} className="opacity-30" />
            <span className="text-sm">
              {query ? t("chats.noMatch") : t("chats.noChatsYet")}
            </span>
          </div>
        )}

        {items.length > 0 && (
          <div className="rounded-xl border border-border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-surface-muted text-xs uppercase tracking-wider">
                  <TableHead className="w-10">
                    <Checkbox
                      checked={allVisibleSelected}
                      onCheckedChange={toggleSelectAllVisible}
                      aria-label={t("chats.selectAllVisible")}
                    />
                  </TableHead>
                  <TableHead className="font-semibold">{t("chats.colTitle")}</TableHead>
                  <TableHead className="font-semibold hidden sm:table-cell w-28">{t("chats.colMessages")}</TableHead>
                  <TableHead className="font-semibold hidden md:table-cell w-40">{t("chats.colUpdated")}</TableHead>
                  <TableHead className="w-20" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((c) => (
                  <ChatRow
                    key={c.id}
                    conversation={c}
                    snippet={c.match_snippet ?? null}
                    selected={selected.has(c.id)}
                    onToggleSelect={toggleSelect}
                    onOpen={openChat}
                    onOpenEvidence={openEvidence}
                    onRename={handleRename}
                    onPin={handlePin}
                    onRequestDelete={setConfirmDeleteId}
                    formatDate={formatDate}
                  />
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        {/* Pagination */}
        {total > pageSize && (
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{t("chats.pageOfTotal", { page: currentPage, pages: totalPages, total })}</span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={browser.prevPage}
                disabled={offset === 0 || loading}
              >
                {t("chats.previous")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={browser.nextPage}
                disabled={offset + pageSize >= total || loading}
              >
                {t("chats.next")}
              </Button>
            </div>
          </div>
        )}

        <ConfirmDialog
          isOpen={confirmBulkDelete}
          title={t("chats.deleteSelected")}
          message={t("chats.deleteSelectedConfirm", { count: selected.size })}
          variant="danger"
          confirmLabel={t("common.delete")}
          onConfirm={handleBulkDelete}
          onCancel={() => setConfirmBulkDelete(false)}
        />

        <ConfirmDialog
          isOpen={confirmDeleteId !== null}
          title={t("sidebar.delete")}
          message={t("sidebar.deleteConversationConfirm")}
          variant="danger"
          confirmLabel={t("common.delete")}
          onConfirm={handleConfirmedDelete}
          onCancel={() => setConfirmDeleteId(null)}
        />

        {evidenceTarget && (
          <ConversationEvidenceDrawer
            conversationId={evidenceTarget.id}
            title={evidenceTarget.title}
            onClose={() => setEvidenceTarget(null)}
          />
        )}
      </div>
    </div>
  );
}

/* ── Single chat row ──────────────────────────────────────────────────── */

function ChatRow({
  conversation: c,
  snippet,
  selected,
  onToggleSelect,
  onOpen,
  onOpenEvidence,
  onRename,
  onPin,
  onRequestDelete,
  formatDate,
}: {
  conversation: Conversation;
  snippet: string | null;
  selected: boolean;
  onToggleSelect: (id: string) => void;
  onOpen: (id: string) => void;
  onOpenEvidence: (id: string, title: string) => void;
  onRename: (id: string, title: string) => Promise<Conversation | null>;
  onPin: (id: string, pinned: boolean) => void;
  onRequestDelete: (id: string) => void;
  formatDate: (value: string) => string;
}) {
  const { t } = useI18n();
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(c.title);
  const [showExport, setShowExport] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const commitRename = useCallback(async () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== c.title) await onRename(c.id, trimmed);
    else setEditValue(c.title);
    setEditing(false);
  }, [editValue, c.id, c.title, onRename]);

  const secondary = snippet ?? c.last_message ?? null;

  return (
    <TableRow>
      <TableCell className="align-top">
        <Checkbox
          checked={selected}
          onCheckedChange={() => onToggleSelect(c.id)}
          aria-label={t("chats.selectRow")}
        />
      </TableCell>
      <TableCell className="max-w-xs lg:max-w-lg">
        <div className="flex items-start gap-2">
          {c.pinned && <Pin size={14} className="text-primary/70 mt-0.5 shrink-0" />}
          <div className="min-w-0 flex-1">
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
                className="h-7 text-sm font-medium"
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
      </TableCell>
      <TableCell className="text-muted-foreground text-xs hidden sm:table-cell">
        {c.message_count ?? 0}
      </TableCell>
      <TableCell className="text-muted-foreground text-xs hidden md:table-cell">
        {c.updated_at ? formatDate(c.updated_at) : "—"}
      </TableCell>
      <TableCell className="text-right">
        <div className="flex items-center justify-end gap-0.5">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onOpenEvidence(c.id, c.title)}
            aria-label={t("chats.viewEvidence")}
            title={t("chats.viewEvidence")}
            className="text-muted-foreground hover:text-foreground"
          >
            <PanelRightOpen size={16} />
          </Button>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={t("sidebar.moreOptions")}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    <MoreVertical size={16} />
                  </Button>
                }
              />
            <DropdownMenuContent align="end" sideOffset={4} className="w-44">
              <DropdownMenuItem onClick={() => { setEditValue(c.title); setEditing(true); }}>
                <Pencil size={16} />
                <span>{t("sidebar.rename")}</span>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onPin(c.id, !c.pinned)}>
                <Pin size={16} />
                <span>{c.pinned ? t("sidebar.unpin") : t("sidebar.pin")}</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setShowExport(true)}>
                <Download size={16} />
                <span>{t("common.export")}</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" onClick={() => onRequestDelete(c.id)}>
                <Trash2 size={16} />
                <span>{t("sidebar.delete")}</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </TableCell>
      {showExport && (
        <ExportDialog
          conversationId={c.id}
          conversationTitle={c.title}
          onClose={() => setShowExport(false)}
        />
      )}
    </TableRow>
  );
}
