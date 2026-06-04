"use client";

import { AdminOnly } from "@/components/AdminOnly";
import { Icon } from "@/components/Icon";
import { useI18n } from "@/hooks/useI18n";
import { useRagSearch } from "@/hooks/useRagSearch";
import type { RagSearchMode } from "@/lib/api";
import styles from "../../settings/Settings.module.css";
import { RagResultCard } from "./RagResultCard";

export default function RagExplorerPage() {
  const { t } = useI18n();
  const {
    query,
    setQuery,
    mode,
    setMode,
    topK,
    setTopK,
    collection,
    setCollection,
    collections,
    result,
    loading,
    error,
    runSearch,
  } = useRagSearch();

  const items = result?.items ?? [];
  // Results are sorted desc by score, so above-cutoff hits come first. In raw mode we
  // draw a divider before the first below-cutoff hit to mark the production threshold.
  const firstBelowIdx =
    result?.mode === "raw" ? items.findIndex((h) => !h.above_cutoff) : -1;

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={styles.title}>{t("ragExplorer.title")}</h1>
        <p className={styles.subtitle}>{t("ragExplorer.subtitle")}</p>
      </div>

      <AdminOnly
        fallback={
          <div className={styles.error}>
            <p className={styles.errorTitle}>{t("adminPage.accessDenied")}</p>
          </div>
        }
      >
        {/* Search controls */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runSearch();
          }}
          className="flex flex-col gap-4"
        >
          <div className="relative">
            <Icon
              name="travel_explore"
              size={18}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-evergreen/40 pointer-events-none"
            />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("ragExplorer.searchPlaceholder")}
              className="w-full pl-10 pr-28 py-2.5 rounded-lg border border-evergreen/20 text-evergreen placeholder-evergreen/30 text-sm focus:outline-none focus:ring-2 focus:ring-evergreen/30"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-1.5 bg-evergreen text-light-cyan rounded-md px-3 py-1.5 text-sm font-medium hover:bg-pacific-blue transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Icon name="progress_activity" size={16} className="animate-spin" />
              ) : (
                <Icon name="search" size={16} />
              )}
              <span className="hidden sm:inline">
                {loading ? t("ragExplorer.searching") : t("ragExplorer.search")}
              </span>
            </button>
          </div>

          <div className="flex flex-wrap items-end gap-x-6 gap-y-3 text-sm">
            {/* Mode toggle */}
            <div>
              <span className="block text-xs font-medium text-evergreen/60 mb-1">
                {t("ragExplorer.mode")}
              </span>
              <div className="inline-flex rounded-lg border border-evergreen/20 overflow-hidden">
                {(["raw", "production"] as RagSearchMode[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    title={
                      m === "raw"
                        ? t("ragExplorer.modeRawHint")
                        : t("ragExplorer.modeProductionHint")
                    }
                    className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                      mode === m
                        ? "bg-evergreen text-light-cyan"
                        : "bg-white text-evergreen/70 hover:bg-evergreen/5"
                    }`}
                  >
                    {m === "raw" ? t("ragExplorer.modeRaw") : t("ragExplorer.modeProduction")}
                  </button>
                ))}
              </div>
            </div>

            {/* Collection */}
            <label className="flex flex-col">
              <span className="text-xs font-medium text-evergreen/60 mb-1">
                {t("ragExplorer.collection")}
              </span>
              <select
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                className="rounded-lg border border-evergreen/20 px-3 py-1.5 text-sm text-evergreen bg-white focus:outline-none focus:ring-2 focus:ring-evergreen/30"
              >
                {collections.length === 0 && collection && (
                  <option value={collection}>{collection}</option>
                )}
                {collections.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                    {c.points_count != null ? ` (${c.points_count})` : ""}
                  </option>
                ))}
              </select>
            </label>

            {/* top_k */}
            <label className="flex flex-col">
              <span className="text-xs font-medium text-evergreen/60 mb-1">
                {t("ragExplorer.topK")}
              </span>
              <input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) =>
                  setTopK(Math.min(50, Math.max(1, Number(e.target.value) || 1)))
                }
                className="w-20 rounded-lg border border-evergreen/20 px-3 py-1.5 text-sm text-evergreen focus:outline-none focus:ring-2 focus:ring-evergreen/30"
              />
            </label>
          </div>
        </form>

        {/* Error */}
        {error && (
          <div className={styles.error}>
            <p className={styles.errorTitle}>{t("common.error")}</p>
            <p>{error}</p>
          </div>
        )}

        {/* Results */}
        {result && !error && (
          <section className="flex flex-col gap-3">
            <p className="text-sm text-evergreen/60">
              {t("ragExplorer.resultsSummary", {
                count: result.count,
                collection: result.collection,
              })}
            </p>

            {items.length === 0 && (
              <div className="rounded-xl border border-dashed border-evergreen/20 p-8 text-center text-evergreen/50 text-sm">
                {t("ragExplorer.noResults")}
              </div>
            )}

            {items.map((hit, idx) => (
              <div key={`${hit.file_path}-${hit.chunk_index}-${idx}`}>
                {idx === firstBelowIdx && (
                  <div className="flex items-center gap-3 my-2 text-xs text-burnt-tangerine/70">
                    <span className="h-px flex-1 bg-burnt-tangerine/30" />
                    <span className="font-medium whitespace-nowrap">
                      {t("ragExplorer.cutoffDivider", {
                        threshold: result.production_threshold.toFixed(2),
                      })}
                    </span>
                    <span className="h-px flex-1 bg-burnt-tangerine/30" />
                  </div>
                )}
                <RagResultCard hit={hit} query={result.query} />
              </div>
            ))}
          </section>
        )}

        {/* Empty (pre-search) state */}
        {!result && !error && !loading && (
          <div className="rounded-xl border border-dashed border-evergreen/20 p-10 text-center text-evergreen/40 text-sm">
            {t("ragExplorer.emptyState")}
          </div>
        )}
      </AdminOnly>
    </main>
  );
}
