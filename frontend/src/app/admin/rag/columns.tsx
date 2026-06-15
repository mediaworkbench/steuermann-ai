"use client"

import { useState } from "react"
import type { ColumnDef } from "@tanstack/react-table"
import { Check, Copy, FileText } from "lucide-react"

import { Button } from "@/components/ui/button"
import { SortableHeader } from "@/components/ui/data-table"
import type { RagSearchHit } from "@/lib/api"

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function Highlighted({ text, pattern }: { text: string; pattern: RegExp | null }) {
  if (!pattern) return <>{text}</>
  const parts = text.split(pattern)
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
  )
}

function useTokens(query: string): RegExp | null {
  // useMemo via state setter pattern — no hook needed in callback-based cells
  const tokens = query
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2)
    .map(escapeRegExp)
  return tokens.length === 0 ? null : new RegExp(`(${tokens.join("|")})`, "gi")
}

const PREVIEW_CHARS = 240

function TextCell({ text, query }: { text: string; query: string }) {
  const [expanded, setExpanded] = useState(false)
  const pattern = useTokens(query)
  const isLong = text.length > PREVIEW_CHARS
  const shown = expanded || !isLong ? text : text.slice(0, PREVIEW_CHARS) + "…"

  return (
    <div className="min-w-0 max-w-xl">
      <p className={`text-sm text-foreground whitespace-pre-wrap leading-relaxed ${!expanded ? "line-clamp-3" : ""}`}>
        <Highlighted text={shown} pattern={pattern} />
      </p>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 text-xs font-medium text-primary hover:text-foreground transition-colors"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  )
}

function CopyCell({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  async function copyText() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* no-op */
    }
  }

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={copyText}
      className="text-muted-foreground hover:text-primary"
      aria-label="Copy"
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </Button>
  )
}

export function createColumns(query: string): ColumnDef<RagSearchHit>[] {
  return [
    {
      id: "score",
      accessorKey: "score",
      header: ({ column }) => (
        <SortableHeader
          title="Score"
          canSort={column.getCanSort()}
          isSorted={column.getIsSorted()}
          toggleSorting={column.toggleSorting}
        />
      ),
      cell: ({ row }) => {
        const hit = row.original
        const scoreClasses = hit.above_cutoff
          ? "bg-primary/15 text-foreground"
          : "bg-warning/15 text-warning"
        return (
          <span
            className={`font-mono text-sm font-semibold rounded-md px-2 py-0.5 ${scoreClasses}`}
          >
            {hit.score.toFixed(3)}
          </span>
        )
      },
    },
    {
      id: "file_name",
      accessorKey: "file_name",
      header: ({ column }) => (
        <SortableHeader
          title="File"
          canSort={column.getCanSort()}
          isSorted={column.getIsSorted()}
          toggleSorting={column.toggleSorting}
        />
      ),
      cell: ({ row }) => {
        const hit = row.original
        return (
          <span className="flex items-center gap-1.5 text-foreground font-medium text-sm min-w-0">
            <FileText size={14} className="text-muted-foreground shrink-0" />
            <span className="truncate max-w-48" title={hit.file_path}>
              {hit.file_name}
            </span>
          </span>
        )
      },
    },
    {
      id: "chunk",
      accessorFn: (row) =>
        row.chunk_index != null
          ? `${row.chunk_index}/${row.chunk_count ?? "?"}`
          : "",
      header: "Chunk",
      cell: ({ row }) => {
        const hit = row.original
        return hit.chunk_index != null ? (
          <span className="text-xs text-muted-foreground font-mono">
            {hit.chunk_index}/{hit.chunk_count ?? "?"}
          </span>
        ) : null
      },
    },
    {
      id: "detected_language",
      accessorKey: "detected_language",
      header: "Lang",
      cell: ({ row }) => {
        const lang = row.original.detected_language
        return lang ? (
          <span className="text-xs uppercase tracking-wide rounded bg-surface-muted text-muted-foreground px-1.5 py-0.5 border border-border">
            {lang}
          </span>
        ) : null
      },
    },
    {
      id: "text",
      accessorKey: "text",
      header: "Preview",
      cell: ({ row }) => <TextCell text={row.original.text} query={query} />,
    },
    {
      id: "actions",
      cell: ({ row }) => <CopyCell text={row.original.text} />,
    },
  ]
}
