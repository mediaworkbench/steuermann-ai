"use client";

import { useMemo, useState } from "react";
import { FileText, Check, Copy } from "lucide-react";
import { useI18n } from "@/hooks/useI18n";
import type { RagSearchHit } from "@/lib/api";

const PREVIEW_CHARS = 320;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Highlight whitespace-separated query tokens (≥2 chars) within the chunk text. */
function useHighlighter(query: string) {
  return useMemo(() => {
    const tokens = query
      .split(/\s+/)
      .map((t) => t.trim())
      .filter((t) => t.length >= 2)
      .map(escapeRegExp);
    if (tokens.length === 0) return null;
    return new RegExp(`(${tokens.join("|")})`, "gi");
  }, [query]);
}

function Highlighted({ text, pattern }: { text: string; pattern: RegExp | null }) {
  if (!pattern) return <>{text}</>;
  // The pattern has a single capture group, so String.split keeps the matched
  // delimiters at odd indices — render those as <mark>. (Avoids the stateful
  // `.test()`/lastIndex pitfall of a /g regex.)
  const parts = text.split(pattern);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="bg-primary/20 text-foreground rounded px-0.5">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

export function RagResultCard({ hit, query }: { hit: RagSearchHit; query: string }) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const pattern = useHighlighter(query);

  const isLong = hit.text.length > PREVIEW_CHARS;
  const shown = expanded || !isLong ? hit.text : hit.text.slice(0, PREVIEW_CHARS) + "…";

  async function copyText() {
    try {
      await navigator.clipboard.writeText(hit.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — no-op */
    }
  }

  const scoreClasses = hit.above_cutoff
    ? "bg-primary/15 text-foreground"
    : "bg-warning/15 text-warning";

  return (
    <div className="rounded-xl border border-border bg-surface p-4 shadow-sm">
      <div className="flex items-center gap-3 flex-wrap">
        <span className={`font-mono text-sm font-semibold rounded-md px-2 py-0.5 ${scoreClasses}`}>
          {hit.score.toFixed(3)}
        </span>
        <span className="flex items-center gap-1.5 text-foreground font-medium text-sm min-w-0">
          <FileText size={16} className="text-muted-foreground shrink-0" />
          <span className="truncate" title={hit.file_path}>
            {hit.file_name}
          </span>
        </span>
        {hit.chunk_index != null && (
          <span className="text-xs text-muted-foreground font-mono">
            {t("ragExplorer.chunk", {
              index: hit.chunk_index,
              count: hit.chunk_count ?? "?",
            })}
          </span>
        )}
        {hit.detected_language && (
          <span className="text-xs uppercase tracking-wide rounded bg-surface-muted text-muted-foreground px-1.5 py-0.5 border border-border">
            {hit.detected_language}
          </span>
        )}
        <span
          className={`text-xs ml-auto ${
            hit.above_cutoff ? "text-primary" : "text-warning"
          }`}
        >
          {hit.above_cutoff ? t("ragExplorer.aboveCutoff") : t("ragExplorer.belowCutoff")}
        </span>
        <button
          type="button"
          onClick={copyText}
          className="text-muted-foreground hover:text-primary transition-colors"
          title={copied ? t("ragExplorer.copied") : t("ragExplorer.copy")}
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
        </button>
      </div>

      <p className="mt-3 text-sm text-foreground whitespace-pre-wrap leading-relaxed">
        <Highlighted text={shown} pattern={pattern} />
      </p>

      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mt-2 text-xs font-medium text-primary hover:text-foreground transition-colors"
        >
          {expanded ? t("ragExplorer.showLess") : t("ragExplorer.showMore")}
        </button>
      )}
    </div>
  );
}
