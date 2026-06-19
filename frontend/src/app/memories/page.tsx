"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { fetchMemories, fetchMemoryStats, deleteMemory, resetMyData } from "@/lib/api";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardDescription, CardContent } from "@/components/ui/card";
import { DataTable } from "@/components/ui/data-table";
import { createColumns } from "./columns";
import { useI18n } from "@/hooks/useI18n";
import { toast } from "sonner";
import type { MemoryItem, MemoryStats } from "@/lib/types";
import { Brain, Trash2, RefreshCw } from "lucide-react";

const PAGE_SIZE = 50;

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
    <Card size="sm">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-1">
        <p className="text-3xl font-bold text-foreground">{value}</p>
        {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      </CardContent>
    </Card>
  );
}

export default function MemoriesPage() {
  const { t, formatDate } = useI18n();
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [clearing, setClearing] = useState(false);

  const load = useCallback(
    async () => {
      setLoading(true);
      const [list, st] = await Promise.all([
        fetchMemories(PAGE_SIZE, 0),
        stats === null ? fetchMemoryStats() : Promise.resolve(stats),
      ]);
      if (list) {
        setItems(list.items);
        setTotal(list.count);
      }
      if (st && stats === null) setStats(st);
      setLoading(false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const refreshStats = useCallback(async () => {
    const st = await fetchMemoryStats();
    if (st) setStats(st);
  }, []);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    setDeletingId(id);
    const ok = await deleteMemory(id);
    if (ok) {
      setItems((prev) => prev.filter((m) => m.memory_id !== id));
      setTotal((t) => t - 1);
      await refreshStats();
    }
    setDeletingId(null);
    setConfirmDelete(null);
  }, [refreshStats]);

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
        await Promise.all([load(), refreshStats()]);
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

  const columns = useMemo(
    () => createColumns({
      deletingId,
      confirmDelete,
      onDelete: handleDelete,
      onCancel: () => setConfirmDelete(null),
      onStartDelete: (id) => setConfirmDelete(id),
      onRateChange: handleRateChange,
      t,
      formatDate,
    }),
    [deletingId, confirmDelete, handleDelete, handleRateChange, t, formatDate],
  );

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">

        {/* Header */}
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-3xl font-bold text-foreground">{t("memories.title")}</h1>
            <p className="mt-1 text-muted-foreground">{t("memories.subtitle")}</p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setClearConfirmOpen(true)}
              disabled={clearing || total === 0}
              className="gap-1.5 text-destructive border-destructive/30 hover:bg-destructive/10"
            >
              <Trash2 size={14} className={clearing ? "animate-pulse" : ""} />
              {t("memories.clearAll")}
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={() => load()}
              disabled={clearing}
              className="gap-1.5"
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              {t("memories.refresh")}
            </Button>
          </div>
        </div>

        {/* Stats strip */}
        {stats && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-5 mb-8">
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

        {/* Table */}
        <DataTable
          columns={columns}
          data={items}
          searchColumn="text"
          searchPlaceholder={t("memories.filterPlaceholder")}
          loading={loading}
          loadingText={t("memories.loading")}
          emptyText={t("memories.noMemoriesYet")}
          emptyNode={
            <div className="flex flex-col items-center gap-2 py-14 text-muted-foreground text-center">
              <Brain size={28} className="opacity-30" />
              <span className="text-sm">{t("memories.noMemoriesYet")}</span>
              <span className="text-xs max-w-xs leading-relaxed">
                {t("memories.emptyHint")}
              </span>
            </div>
          }
        />

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
      </div>
    </div>
  );
}
