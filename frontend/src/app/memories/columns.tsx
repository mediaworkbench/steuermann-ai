"use client"

import type { ColumnDef } from "@tanstack/react-table"
import { Trash2 } from "lucide-react"

import { MemoryRating } from "@/components/MemoryRating"
import { Button } from "@/components/ui/button"
import { SortableHeader } from "@/components/ui/data-table"
import type { MemoryItem } from "@/lib/types"

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

function RatingHelpPopover({ label }: { label: string }) {
  return (
    <span className="group relative cursor-default" aria-label={label}>
      <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-border text-[9px] leading-none text-muted-foreground select-none">
        i
      </span>
      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg bg-surface px-3 py-2 text-[11px] text-foreground leading-snug shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-10 normal-case tracking-normal font-normal border border-border">
        {label}
      </span>
    </span>
  );
}

interface DeleteCellProps {
  memoryId: string
  deletingId: string | null
  confirmDelete: string | null
  onDelete: (id: string) => void
  onCancel: () => void
  onStartDelete: () => void
  t: (key: string, vars?: Record<string, string | number>) => string
}

function DeleteCell({
  memoryId,
  deletingId,
  confirmDelete,
  onDelete,
  onCancel,
  onStartDelete,
  t,
}: DeleteCellProps) {
  if (confirmDelete === memoryId) {
    return (
      <div className="flex items-center gap-1 justify-end">
        <Button
          variant="destructive"
          size="sm"
          onClick={() => onDelete(memoryId)}
          disabled={deletingId === memoryId}
        >
          {deletingId === memoryId ? "…" : t("memories.confirmYes")}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={onCancel}
        >
          {t("memories.confirmNo")}
        </Button>
      </div>
    );
  }

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={onStartDelete}
      aria-label={t("memories.deleteMemory")}
      className="text-muted-foreground hover:text-destructive"
    >
      <Trash2 size={14} />
    </Button>
  );
}

export function createColumns(
  deps: {
    deletingId: string | null
    confirmDelete: string | null
    onDelete: (id: string) => void
    onCancel: () => void
    onStartDelete: (id: string) => void
    onRateChange: (id: string, rating: number) => void
    t: (key: string, vars?: Record<string, string | number>) => string
    formatDate: (date: string) => string
  }
): ColumnDef<MemoryItem>[] {
  const { deletingId, confirmDelete, onDelete, onCancel, onStartDelete, onRateChange, t, formatDate } = deps

  return [
    {
      accessorKey: "text",
      header: ({ column }) => (
        <SortableHeader
          title={t("memories.memory")}
          canSort={column.getCanSort()}
          isSorted={column.getIsSorted()}
          toggleSorting={column.toggleSorting}
        />
      ),
      cell: ({ row }) => (
        <div className="max-w-xs lg:max-w-lg">
          <span className="line-clamp-3 text-foreground">{row.original.text}</span>
          {row.original.is_related && (
            <span className="ml-2 text-xs text-foreground bg-surface-muted px-1.5 py-0.5 rounded font-medium border border-border">
              {t("memories.related")}
            </span>
          )}
        </div>
      ),
    },
    {
      accessorKey: "importance_score",
      header: ({ column }) => (
        <SortableHeader
          title={t("memories.importance")}
          canSort={column.getCanSort()}
          isSorted={column.getIsSorted()}
          toggleSorting={column.toggleSorting}
        />
      ),
      cell: ({ row }) => <ImportanceBar score={row.original.importance_score} />,
    },
    {
      accessorKey: "user_rating",
      header: ({ column }) => (
        <div className="flex items-center gap-1">
          <SortableHeader
            title={t("memories.rating")}
            canSort={column.getCanSort()}
            isSorted={column.getIsSorted()}
            toggleSorting={column.toggleSorting}
          />
          <RatingHelpPopover label={t("memories.ratingHelp")} />
        </div>
      ),
      cell: ({ row }) => (
        <MemoryRating
          memoryId={row.original.memory_id}
          initialRating={typeof row.original.user_rating === "number" ? row.original.user_rating : 0}
          onRatingChange={(rating) => onRateChange(row.original.memory_id, rating)}
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
      ),
    },
    {
      accessorKey: "created_at",
      header: ({ column }) => (
        <SortableHeader
          title={t("memories.saved")}
          canSort={column.getCanSort()}
          isSorted={column.getIsSorted()}
          toggleSorting={column.toggleSorting}
        />
      ),
      cell: ({ row }) => {
        const date = row.original.created_at
        return date ? (
          <span className="text-muted-foreground text-xs">{formatDate(date)}</span>
        ) : (
          <span className="text-muted-foreground text-xs">—</span>
        )
      },
    },
    {
      id: "actions",
      cell: ({ row }) => (
        <div className="text-right">
          <DeleteCell
            memoryId={row.original.memory_id}
            deletingId={deletingId}
            confirmDelete={confirmDelete}
            onDelete={onDelete}
            onCancel={onCancel}
            onStartDelete={() => onStartDelete(row.original.memory_id)}
            t={t}
          />
        </div>
      ),
    },
  ]
}
