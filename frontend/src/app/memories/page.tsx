"use client";

import { useState, useEffect, useCallback } from "react";
import { Trash2, Star, RefreshCw, Brain } from "lucide-react";
import { fetchMemories, fetchMemoryStats, deleteMemory, rateMemory } from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";
import type { MemoryItem, MemoryStats } from "@/lib/types";

const PAGE_SIZE = 50;

function ImportanceBar({ score }: { score: number | null }) {
  if (score === null) return <span className="text-xs text-white/30">—</span>;
  const pct = Math.round(score * 100);
  const color =
    pct >= 80 ? "bg-emerald-400" : pct >= 50 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-white/50">{pct}%</span>
    </div>
  );
}

function StarRating({
  memoryId,
  current,
  onRate,
}: {
  memoryId: string;
  current: number | null;
  onRate: (id: string, r: number) => void;
}) {
  const [hover, setHover] = useState(0);
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          onClick={() => onRate(memoryId, n)}
          onMouseEnter={() => setHover(n)}
          onMouseLeave={() => setHover(0)}
          className="p-0.5 rounded transition-colors cursor-pointer"
          aria-label={`Rate ${n} stars`}
        >
          <Star
            size={13}
            className={
              n <= (hover || current || 0)
                ? "fill-yellow-400 text-yellow-400"
                : "text-white/20"
            }
          />
        </button>
      ))}
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
    <div className="rounded-xl border border-white/10 bg-white/5 px-5 py-4">
      <p className="text-xs font-semibold uppercase tracking-wider text-light-cyan/50 mb-1">
        {label}
      </p>
      <p className="text-2xl font-bold text-light-cyan">{value}</p>
      {sub && <p className="text-xs text-white/40 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function MemoriesPage() {
  const userId = CURRENT_USER_ID;
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

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

  const handleRate = async (id: string, rating: number) => {
    const ok = await rateMemory(id, rating);
    if (ok) {
      setItems((prev) =>
        prev.map((m) => (m.memory_id === id ? { ...m, user_rating: rating } : m)),
      );
    }
  };

  const filtered = search.trim()
    ? items.filter((m) => m.text.toLowerCase().includes(search.toLowerCase()))
    : items;

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <main className="flex-1 overflow-y-auto bg-white">
      <div className="max-w-5xl mx-auto px-4 py-6 md:px-8 md:py-8 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <Brain size={28} className="text-evergreen" />
            <div>
              <h1 className="text-2xl font-bold text-evergreen">Memory</h1>
              <p className="text-sm text-evergreen/60">
                Memories the agent has learned about you
              </p>
            </div>
          </div>
          <button
            onClick={() => load(offset)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                       bg-evergreen text-light-cyan hover:bg-evergreen/80 transition-colors cursor-pointer"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>

        {/* Stats strip */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <StatCard label="Total" value={stats.totals.memories} />
            <StatCard label="Recent 7d" value={stats.totals.recent_7d} />
            <StatCard label="Rated" value={stats.totals.rated} sub={`${Math.round(stats.ratios.rated_coverage * 100)}% coverage`} />
            <StatCard label="Unrated" value={stats.totals.unrated} />
            <StatCard
              label="Avg importance"
              value={`${Math.round((stats.quality.average_importance || 0) * 100)}%`}
            />
          </div>
        )}

        {/* Search */}
        <div>
          <input
            type="search"
            placeholder="Filter memories…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full max-w-md px-4 py-2 rounded-lg border border-evergreen/20
                       text-evergreen placeholder-evergreen/30 text-sm
                       focus:outline-none focus:ring-2 focus:ring-evergreen/30"
          />
        </div>

        {/* Table */}
        <div className="rounded-xl border border-evergreen/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-evergreen text-light-cyan text-xs uppercase tracking-wider">
                <th className="px-4 py-3 text-left font-semibold">Memory</th>
                <th className="px-4 py-3 text-left font-semibold hidden md:table-cell w-36">Importance</th>
                <th className="px-4 py-3 text-left font-semibold hidden sm:table-cell w-36">Rating</th>
                <th className="px-4 py-3 text-left font-semibold hidden lg:table-cell w-40">Saved</th>
                <th className="px-4 py-3 w-12" />
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-evergreen/40">
                    Loading…
                  </td>
                </tr>
              )}
              {!loading && filtered.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-10 text-center text-evergreen/40">
                    {search ? "No memories match your filter." : "No memories yet."}
                  </td>
                </tr>
              )}
              {!loading &&
                filtered.map((mem, i) => (
                  <tr
                    key={mem.memory_id}
                    className={`border-t border-evergreen/5 ${
                      i % 2 === 0 ? "bg-white" : "bg-evergreen/[0.02]"
                    } hover:bg-evergreen/5 transition-colors`}
                  >
                    <td className="px-4 py-3 text-evergreen leading-snug max-w-xs lg:max-w-lg">
                      <span className="line-clamp-3">{mem.text}</span>
                      {mem.is_related && (
                        <span className="ml-2 text-xs text-light-cyan bg-evergreen/20 px-1.5 py-0.5 rounded">
                          related
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 hidden md:table-cell">
                      <ImportanceBar score={mem.importance_score} />
                    </td>
                    <td className="px-4 py-3 hidden sm:table-cell">
                      <StarRating
                        memoryId={mem.memory_id}
                        current={mem.user_rating}
                        onRate={handleRate}
                      />
                    </td>
                    <td className="px-4 py-3 text-evergreen/40 text-xs hidden lg:table-cell">
                      {mem.created_at
                        ? new Date(mem.created_at).toLocaleDateString()
                        : "—"}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {confirmDelete === mem.memory_id ? (
                        <div className="flex items-center gap-1 justify-end">
                          <button
                            onClick={() => handleDelete(mem.memory_id)}
                            disabled={deletingId === mem.memory_id}
                            className="text-xs px-2 py-1 rounded bg-red-500 text-white hover:bg-red-600 transition-colors cursor-pointer"
                          >
                            {deletingId === mem.memory_id ? "…" : "Yes"}
                          </button>
                          <button
                            onClick={() => setConfirmDelete(null)}
                            className="text-xs px-2 py-1 rounded bg-evergreen/10 text-evergreen hover:bg-evergreen/20 transition-colors cursor-pointer"
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setConfirmDelete(mem.memory_id)}
                          className="p-1.5 rounded text-evergreen/30 hover:text-red-500 hover:bg-red-50 transition-colors cursor-pointer"
                          aria-label="Delete memory"
                        >
                          <Trash2 size={14} />
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
          <div className="flex items-center justify-between text-sm text-evergreen/50">
            <span>
              Page {currentPage} of {totalPages} ({total} total)
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => load(offset - PAGE_SIZE)}
                disabled={offset === 0 || loading}
                className="px-3 py-1.5 rounded border border-evergreen/20 hover:bg-evergreen/5
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                Previous
              </button>
              <button
                onClick={() => load(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total || loading}
                className="px-3 py-1.5 rounded border border-evergreen/20 hover:bg-evergreen/5
                           disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
