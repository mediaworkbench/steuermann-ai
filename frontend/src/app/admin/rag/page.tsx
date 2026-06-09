"use client";

import { AdminOnly } from "@/components/AdminOnly";
import { Compass, LoaderCircle, Search } from "lucide-react";
import { useI18n } from "@/hooks/useI18n";
import { useRagSearch } from "@/hooks/useRagSearch";
import { RagResultCard } from "./RagResultCard";

export default function RagExplorerPage() {
  const { t } = useI18n();
  const {
    query,
    setQuery,
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
  // Results are sorted desc by score, so above-cutoff hits come first. Draw a divider
  // before the first below-cutoff hit to mark the production threshold.
  const firstBelowIdx = items.findIndex((h) => !h.above_cutoff);

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("ragExplorer.title")}</h1>
          <p className="mt-1 text-muted-foreground">{t("ragExplorer.subtitle")}</p>
        </div>
      </div>

      <AdminOnly
        fallback={
          <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
            <p className="mb-1 font-semibold">{t("adminPage.accessDenied")}</p>
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
            <Compass
              size={18}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none"
            />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("ragExplorer.searchPlaceholder")}
              className="w-full pl-10 pr-28 py-2.5 rounded-lg border border-border bg-surface text-foreground placeholder:text-muted-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-1.5 bg-primary text-primary-foreground rounded-md px-3 py-1.5 text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? (
                <LoaderCircle size={16} className="animate-spin" />
              ) : (
                <Search size={16} />
              )}
              <span className="hidden sm:inline">
                {loading ? t("ragExplorer.searching") : t("ragExplorer.search")}
              </span>
            </button>
          </div>

          <div className="flex flex-wrap items-end gap-x-6 gap-y-3 text-sm">
            {/* Collection */}
            <label className="flex flex-col">
              <span className="text-xs font-medium text-muted-foreground mb-1">
                {t("ragExplorer.collection")}
              </span>
              <select
                value={collection}
                onChange={(e) => setCollection(e.target.value)}
                className="rounded-lg border border-border px-3 py-1.5 text-sm text-foreground bg-surface focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
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
              <span className="text-xs font-medium text-muted-foreground mb-1">
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
                className="w-20 rounded-lg border border-border px-3 py-1.5 text-sm text-foreground bg-surface focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
              />
            </label>
          </div>
        </form>

        {/* Error */}
        {error && (
          <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
            <p className="mb-1 font-semibold">{t("common.error")}</p>
            {error ? <p>{error}</p> : null}
          </div>
        )}

        {/* Results */}
        {result && !error && (
          <section className="flex flex-col gap-3">
            <p className="text-sm text-muted-foreground">
              {t("ragExplorer.resultsSummary", {
                count: result.count,
                collection: result.collection,
              })}
            </p>

            {items.length === 0 && (
              <div className="rounded-xl border border-dashed border-border p-8 text-center text-muted-foreground text-sm">
                {t("ragExplorer.noResults")}
              </div>
            )}

            {items.map((hit, idx) => (
              <div key={`${hit.file_path}-${hit.chunk_index}-${idx}`}>
                {idx === firstBelowIdx && (
                  <div className="flex items-center gap-3 my-2 text-xs text-warning">
                    <span className="h-px flex-1 bg-warning/40" />
                    <span className="font-medium whitespace-nowrap">
                      {t("ragExplorer.cutoffDivider", {
                        threshold: result.production_threshold.toFixed(2),
                      })}
                    </span>
                    <span className="h-px flex-1 bg-warning/40" />
                  </div>
                )}
                <RagResultCard hit={hit} query={result.query} />
              </div>
            ))}
          </section>
        )}

        {/* Empty (pre-search) state */}
        {!result && !error && !loading && (
          <div className="rounded-xl border border-dashed border-border p-10 text-center text-muted-foreground text-sm">
            {t("ragExplorer.emptyState")}
          </div>
        )}
      </AdminOnly>
      </div>
    </div>
  );
}
