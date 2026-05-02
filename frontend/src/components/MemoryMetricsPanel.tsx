"use client";

import { Brain, Database, GitMerge, Zap } from "lucide-react";
import type { MetricsData } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

interface Props {
  metrics: MetricsData;
  formatNumber: (n: number) => string;
}

function MiniCard({
  icon,
  label,
  value,
  unit,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  unit?: string;
}) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-evergreen/10 bg-white px-4 py-3 shadow-sm">
      <div className="mt-0.5 text-evergreen/50">{icon}</div>
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-evergreen/40 mb-0.5">
          {label}
        </p>
        <p className="text-xl font-bold text-evergreen">
          {value}
          {unit && <span className="text-sm font-medium text-evergreen/50 ml-1">{unit}</span>}
        </p>
      </div>
    </div>
  );
}

const STATUS_COLORS: Record<string, string> = {
  success: "text-emerald-600 bg-emerald-50",
  error: "text-red-600 bg-red-50",
};

export function MemoryMetricsPanel({ metrics, formatNumber }: Props) {
  const { t } = useI18n();
  const opLabels: Record<string, string> = {
    load: t("metrics.memoryLoads"),
    update: t("metrics.memoryUpdates"),
    query: t("metrics.memoryQueries"),
  };
  const ops = metrics.memory_ops ?? {};
  const byStatus = metrics.memory_ops_by_status ?? {};

  const totalOps = ["load", "update", "query"].reduce(
    (sum, op) => sum + (ops[op] ?? 0),
    0
  );
  const avgQuality = ops.avg_quality_score != null ? ops.avg_quality_score : null;
  const coocNodes = ops.co_occurrence_nodes ?? null;
  const coocEdges = ops.co_occurrence_edges ?? null;
  const rankingMs = ops.avg_importance_ranking_ms ?? null;

  // Build per-operation status breakdown table
  const opBreakdown: Record<string, Record<string, number>> = {};
  for (const [key, val] of Object.entries(byStatus)) {
    const [op, status] = key.split("/");
    if (!opBreakdown[op]) opBreakdown[op] = {};
    opBreakdown[op][status] = val;
  }
  const knownOps = Object.keys(opBreakdown).length > 0 ? Object.keys(opBreakdown) : Object.keys(ops).filter(k => !k.startsWith("avg_") && !k.startsWith("co_occurrence") && k !== "avg_importance_ranking_ms");

  if (totalOps === 0 && avgQuality === null && coocNodes === null) {
    return null;
  }

  return (
    <div
      style={{
        background: "var(--bg-card)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        border: "1px solid var(--border-color)",
        padding: "var(--spacing-lg)",
        marginBottom: "var(--spacing-xl)",
      }}
    >
      <h3
        style={{
          fontSize: "1.125rem",
          fontWeight: 600,
          color: "var(--text-primary)",
          margin: "0 0 var(--spacing-md) 0",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
        }}
      >
        <Brain size={18} />
        {t("metrics.memoryOperations")}
      </h3>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <MiniCard
          icon={<Database size={16} />}
          label={t("metrics.totalOps")}
          value={formatNumber(Math.round(totalOps))}
        />
        {avgQuality !== null && (
          <MiniCard
            icon={<Zap size={16} />}
            label={t("metrics.avgQuality")}
            value={`${Math.round(avgQuality * 100)}%`}
          />
        )}
        {coocNodes !== null && (
          <MiniCard
            icon={<GitMerge size={16} />}
            label={t("metrics.coOccurrenceNodes")}
            value={formatNumber(Math.round(coocNodes))}
          />
        )}
        {coocEdges !== null && (
          <MiniCard
            icon={<GitMerge size={16} />}
            label={t("metrics.coOccurrenceEdges")}
            value={formatNumber(Math.round(coocEdges))}
          />
        )}
        {rankingMs !== null && (
          <MiniCard
            icon={<Zap size={16} />}
            label={t("metrics.rankingLatency")}
            value={rankingMs.toFixed(1)}
            unit="ms"
          />
        )}
      </div>

      {/* Breakdown table */}
      {knownOps.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-evergreen/10">
                <th className="py-2 text-left font-semibold text-evergreen/50 uppercase text-xs tracking-wider pr-4">
                  {t("metrics.operation")}
                </th>
                <th className="py-2 text-right font-semibold text-evergreen/50 uppercase text-xs tracking-wider px-4">
                  {t("metrics.success")}
                </th>
                <th className="py-2 text-right font-semibold text-evergreen/50 uppercase text-xs tracking-wider px-4">
                  {t("metrics.error")}
                </th>
                <th className="py-2 text-right font-semibold text-evergreen/50 uppercase text-xs tracking-wider pl-4">
                  {t("metrics.total")}
                </th>
              </tr>
            </thead>
            <tbody>
              {knownOps.map((op) => {
                const statuses = opBreakdown[op] ?? {};
                const success = statuses["success"] ?? 0;
                const error = statuses["error"] ?? 0;
                // fall back to raw ops count when no status breakdown available
                const opTotal = Object.keys(statuses).length > 0
                  ? success + error
                  : (ops[op] ?? 0);
                return (
                  <tr key={op} className="border-b border-evergreen/5 hover:bg-evergreen/[0.02]">
                    <td className="py-2 pr-4 font-medium text-evergreen">
                      {opLabels[op] ?? op}
                    </td>
                    <td className="py-2 px-4 text-right">
                      {Object.keys(statuses).length > 0 ? (
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${STATUS_COLORS["success"]}`}>
                          {formatNumber(Math.round(success))}
                        </span>
                      ) : (
                        <span className="text-evergreen/30">—</span>
                      )}
                    </td>
                    <td className="py-2 px-4 text-right">
                      {Object.keys(statuses).length > 0 ? (
                        error > 0 ? (
                          <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${STATUS_COLORS["error"]}`}>
                            {formatNumber(Math.round(error))}
                          </span>
                        ) : (
                          <span className="text-evergreen/30">0</span>
                        )
                      ) : (
                        <span className="text-evergreen/30">—</span>
                      )}
                    </td>
                    <td className="py-2 pl-4 text-right font-bold text-evergreen">
                      {formatNumber(Math.round(opTotal))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
