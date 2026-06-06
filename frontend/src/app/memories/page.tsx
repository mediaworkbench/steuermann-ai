"use client";

import { useState, useEffect, useCallback } from "react";
import { fetchMemories, fetchMemoryStats, deleteMemory, resetMyData } from "@/lib/api";
import { MemoryRating } from "@/components/MemoryRating";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useI18n } from "@/hooks/useI18n";
import { CURRENT_USER_ID } from "@/lib/runtime";
import { toast } from "sonner";
import type { MemoryItem, MemoryStats } from "@/lib/types";
import { Icon } from "@/components/Icon";
import { PageShell } from "@/components/product/PageShell";

const PAGE_SIZE = 50;

function ImportanceBar({ score }: { score: number | null }) {
  if (score === null) return <span className="text-xs text-muted-foreground">—</span>;
  const pct = Math.round(score * 100);
  const color =
    pct >= 80 ? "bg-success" : pct >= 50 ? "bg-warning" : "bg-destructive";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-surface-muted overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-muted-foreground">{pct}%</span>
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface-muted px-5 py-4">
      <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
        {label}
      </p>
      <p className="text-2xl font-bold text-foreground">{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  );
}

export default function MemoriesPage() {
  const { t, formatDate } = useI18n();
  const userId = CURRENT_USER_ID;
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [clearing, setClearing] = useState(false);

  const load = useCallback(
    async (off: number) => {
      setLoading(true);
      const [list, st] = await Promise.all([
        fetchMemories(userId, PAGE_SIZE, off),
        stats === null ? fetchMemoryStats(userId) : Promise.resolve(stats),
      ]);
      if (list) {
        setItems(list.items);
        setTotal(list.count);
        setOffset(off);
      }
      if (st && stats === null) setStats(st);
      setLoading(false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [userId],
  );

  const refreshStats = useCallback(async () => {
    const st = await fetchMemoryStats(userId);
    if (st) setStats(st);
  }, [userId]);

  useEffect(() => {
    load(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    const ok = await deleteMemory(id);
    if (ok) {
      setItems((prev) => prev.filter((m) => m.memory_id !== id));
      setTotal((t) => t - 1);
      await refreshStats();
    }
    setDeletingId(null);
    setConfirmDelete(null);
  };

  const handleRateChange = useCallback((id: string, rating: number) => {
    setItems((prev) =>
      prev.map((m) => (m.memory_id === id ? { ...m, user_rating: rating } : m)),
    );
  }, []);

  const handleClearAll = useCallback(async () => {
    setClearConfirmOpen(false);
    setClearing(true);
    try {
      const result = await resetMyData({ conversations: false, workspace: false, memories: true });
      if (result.status === "ok") {
        toast.success(t("memories.clearAllSuccess"));
        setItems([]);
        setTotal(0);
        setStats(null);
        setSearch("");
        await Promise.all([load(0), refreshStats()]);
      } else {
        toast.warning(t("memories.clearAllFailed"), { description: result.errors.join(", ") });
      }
    } catch (err) {
      console.error("clear all memories failed", err);
      toast.error(t("memories.clearAllFailed"));
    } finally {
      setClearing(false);
    }
  }, [t, load, refreshStats]);

  const filtered = search.trim()
    ? items.filter((m) => m.text.toLowerCase().includes(search.toLowerCase()))
    : items;

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <PageShell contentClassName="max-w-5xl space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <Icon name="psychology" size={28} className="text-foreground" />
          <div>
            <h1 className="text-2xl font-bold text-foreground">{t("memories.title")}</h1>
              <p className="text-sm text-muted-foreground">
                {t("memories.subtitle")}
              </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setClearConfirmOpen(true)}
            disabled={clearing || total === 0}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                       border border-destructive/30 text-destructive hover:bg-destructive/10
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <Icon name="delete" size={14} className={clearing ? "animate-pulse" : ""} />
            {t("memories.clearAll")}
          </button>
          <button
            onClick={() => load(offset)}
            disabled={clearing}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                       bg-primary text-primary-foreground hover:bg-primary/90
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <Icon name="refresh" size={14} className={loading ? "animate-spin" : ""} />
            {t("memories.refresh")}
          </button>
        </div>
      </div>

        {/* Stats strip */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <StatCard label={t("memories.total")} value={stats.totals.memories} />
            <StatCard label={t("memories.recent7d")} value={stats.totals.recent_7d} />
            <StatCard
              label={t("memories.rated")}
              value={stats.totals.rated}
              sub={t("memories.coverage", { count: Math.round(stats.ratios.rated_coverage * 100) })}
            />
            <StatCard label={t("memories.unrated")} value={stats.totals.unrated} />
            <StatCard
              label={t("memories.avgImportance")}
              value={`${Math.round((stats.quality.average_importance || 0) * 100)}%`}
            />
          </div>
        )}

        {/* Search */}
        <div>
          <input
            type="search"
            placeholder={t("memories.filterPlaceholder")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full max-w-md px-4 py-2 rounded-lg border border-border
                       bg-surface text-foreground placeholder:text-muted-foreground text-sm
                       focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
          />
        </div>

        {/* Table */}
        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-surface-muted text-foreground text-xs uppercase tracking-wider">
                <th className="px-4 py-3 text-left font-semibold">{t("memories.memory")}</th>
                <th className="px-4 py-3 text-left font-semibold hidden md:table-cell w-36">{t("memories.importance")}</th>
                <th className="px-4 py-3 text-left font-semibold hidden sm:table-cell w-36">
                  <span className="flex items-center gap-1">
                    {t("memories.rating")}
                    <span
                      className="group relative cursor-default"
                      aria-label={t("memories.ratingHelp")}
                    >
                      <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-border text-[9px] leading-none text-muted-foreground select-none">
                        i
                      </span>
                      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg bg-surface px-3 py-2 text-[11px] text-foreground leading-snug shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 normal-case tracking-normal font-normal border border-border">
                        {t("memories.ratingHelp")}
                      </span>
                    </span>
                  </span>
                </th>
                <th className="px-4 py-3 text-left font-semibold hidden lg:table-cell w-40">{t("memories.saved")}</th>
                <th className="px-4 py-3 w-12" />
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">
                    {t("memories.loading")}
                  </td>
                </tr>
              )}
              {!loading && filtered.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-14 text-center">
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                      <Icon name="psychology" size={28} className="opacity-30" />
                      <span className="text-sm">
                        {search ? t("memories.noMemoriesMatchFilter") : t("memories.noMemoriesYet")}
                      </span>
                      {!search && (
                        <span className="text-xs max-w-xs leading-relaxed">
                          {t("memories.emptyHint")}
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              )}
              {!loading &&
                filtered.map((mem) => (
                  <tr
                    key={mem.memory_id}
                    className="border-t border-border/60 hover:bg-surface-muted/60 transition-colors"
                  >
                    <td className="px-4 py-3 text-foreground leading-snug max-w-xs lg:max-w-lg">
                      <span className="line-clamp-3">{mem.text}</span>
                      {mem.is_related && (
                        <span className="ml-2 text-xs text-foreground bg-surface-muted px-1.5 py-0.5 rounded font-medium border border-border">
                          {t("memories.related")}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 hidden md:table-cell">
                      <ImportanceBar score={mem.importance_score} />
                    </td>
                    <td className="px-4 py-3 hidden sm:table-cell">
                      <MemoryRating
                        memoryId={mem.memory_id}
                        initialRating={typeof mem.user_rating === "number" ? mem.user_rating : 0}
                        onRatingChange={(rating) => handleRateChange(mem.memory_id, rating)}
                        compact
                        showStatus
                        ariaLabel={t("memories.ratingAriaLabel")}
                        getRateLabel={(count) => t("memories.rateStars", { count })}
                        statusLabels={{
                          saving: t("memories.ratingStatusSaving"),
                          saved: t("memories.ratingStatusSaved"),
                          retry: t("memories.ratingStatusRetry"),
                        }}
                      />
                    </td>
                    <td className="px-4 py-3 text-muted-foreground text-xs hidden lg:table-cell">
                      {mem.created_at
                        ? formatDate(mem.created_at)
                        : "—"}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {confirmDelete === mem.memory_id ? (
                        <div className="flex items-center gap-1 justify-end">
                          <button
                            onClick={() => handleDelete(mem.memory_id)}
                            disabled={deletingId === mem.memory_id}
                            className="text-xs px-2 py-1 rounded bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors cursor-pointer"
                          >
                            {deletingId === mem.memory_id ? "…" : t("memories.confirmYes")}
                          </button>
                          <button
                            onClick={() => setConfirmDelete(null)}
                            className="text-xs px-2 py-1 rounded bg-surface-muted text-foreground hover:bg-surface-muted/80 transition-colors cursor-pointer"
                          >
                            {t("memories.confirmNo")}
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setConfirmDelete(mem.memory_id)}
                          className="p-1.5 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors cursor-pointer"
                          aria-label={t("memories.deleteMemory")}
                        >
                          <Icon name="delete" size={14} />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>
              {t("memories.pageOfTotal", { page: currentPage, pages: totalPages, total })}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => load(offset - PAGE_SIZE)}
                disabled={offset === 0 || loading}
                className="px-3 py-1.5 rounded border border-border hover:bg-surface-muted
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                {t("memories.previous")}
              </button>
              <button
                onClick={() => load(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total || loading}
                className="px-3 py-1.5 rounded border border-border hover:bg-surface-muted
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                {t("memories.next")}
              </button>
            </div>
          </div>
        )}

      <ConfirmDialog
        isOpen={clearConfirmOpen}
        title={t("memories.clearAll")}
        message={t("memories.clearAllConfirmMessage")}
        confirmLabel={t("memories.clearAll")}
        requireChecked
        checkboxLabel={t("confirmDialog.resetCheckboxLabel")}
        onConfirm={handleClearAll}
        onCancel={() => setClearConfirmOpen(false)}
        variant="danger"
      />
    </PageShell>
  );
}
