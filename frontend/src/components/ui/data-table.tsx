"use client"

import * as React from "react"
import {
  type ColumnDef,
  type ColumnFiltersState,
  type RowData,
  type SortingState,
  type ColumnVisibilityState,
  columnFilteringFeature,
  columnVisibilityFeature,
  createFilteredRowModel,
  createPaginatedRowModel,
  createSortedRowModel,
  filterFn_arrIncludes,
  filterFn_equals,
  filterFn_inDateRange,
  filterFn_inNumberRange,
  filterFn_includesString,
  filterFn_weakEquals,
  flexRender,
  rowPaginationFeature,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_datetime,
  sortFn_text,
  tableFeatures,
  useTable,
} from "@tanstack/react-table"
import { ArrowUpDown, ChevronLeft, ChevronRight } from "lucide-react"

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

// TanStack Table v9 only includes what is registered here. Besides the features and row
// models DataTable uses, register every built-in fn the 'auto' filter/sort resolution can
// pick — an unregistered name resolves to nothing (the search box would filter nothing).
const dataTableFeatures = tableFeatures({
  columnFilteringFeature,
  columnVisibilityFeature,
  rowSortingFeature,
  rowPaginationFeature,
  filteredRowModel: createFilteredRowModel(),
  sortedRowModel: createSortedRowModel(),
  paginatedRowModel: createPaginatedRowModel(),
  filterFns: {
    includesString: filterFn_includesString,
    inNumberRange: filterFn_inNumberRange,
    equals: filterFn_equals,
    arrIncludes: filterFn_arrIncludes,
    inDateRange: filterFn_inDateRange,
    weakEquals: filterFn_weakEquals,
  },
  sortFns: {
    alphanumeric: sortFn_alphanumeric,
    datetime: sortFn_datetime,
    text: sortFn_text,
  },
})

export type DataTableFeatures = typeof dataTableFeatures

interface DataTableProps<TData extends RowData> {
  columns: ColumnDef<DataTableFeatures, TData>[]
  data: TData[]
  searchColumn?: string
  searchPlaceholder?: string
  loading?: boolean
  loadingText?: string
  emptyText?: string
  emptyNode?: React.ReactNode
  pageSize?: number
  pageSizeOptions?: number[]
  disablePagination?: boolean
}

export function DataTable<TData extends RowData>({
  columns,
  data,
  searchColumn,
  searchPlaceholder,
  loading,
  loadingText,
  emptyText,
  emptyNode,
  pageSize = 50,
  disablePagination,
}: DataTableProps<TData>) {
  const [sorting, setSorting] = React.useState<SortingState>([])
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = React.useState<ColumnVisibilityState>({})

  const table = useTable({
    features: dataTableFeatures,
    data,
    columns,
    // The paginated row model is always registered; manual pagination shows every row.
    manualPagination: !!disablePagination,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
    },
    ...(!disablePagination ? {
      initialState: {
        pagination: {
          pageIndex: 0,
          pageSize,
        },
      },
    } : {}),
  })

  return (
    <div>
      {searchColumn && (
        <div className="flex items-center py-4">
          <Input
            type="search"
            placeholder={searchPlaceholder ?? "Filter..."}
            value={(table.getColumn(searchColumn)?.getFilterValue() as string) ?? ""}
            onChange={(event) =>
              table.getColumn(searchColumn)?.setFilterValue(event.target.value)
            }
            className="max-w-md"
          />
        </div>
      )}

      {loading ? (
        <div className="text-center py-10 text-muted-foreground">
          {loadingText ?? "Loading..."}
        </div>
      ) : table.getRowModel().rows.length === 0 ? (
        emptyNode ?? (
          <div className="text-center py-10 text-muted-foreground">
            {emptyText ?? "No results."}
          </div>
        )
      ) : (
        <>
          <div className="overflow-hidden rounded-xl border border-border bg-surface shadow-sm">
            <Table>
              <TableHeader>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id} className="bg-surface-muted text-xs uppercase tracking-wider">
                    {headerGroup.headers.map((header) => (
                      <TableHead key={header.id} className="font-semibold">
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                      </TableHead>
                    ))}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows.map((row) => (
                  <TableRow key={row.id}>
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {!disablePagination && (
            <div className="flex items-center justify-between py-4">
              <div className="text-sm text-muted-foreground">
                {table.getFilteredRowModel().rows.length} row(s)
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => table.previousPage()}
                  disabled={!table.getCanPreviousPage()}
                >
                  <ChevronLeft />
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  Page {table.state.pagination.pageIndex + 1} of{" "}
                  {table.getPageCount()}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => table.nextPage()}
                  disabled={!table.getCanNextPage()}
                >
                  Next
                  <ChevronRight />
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export function SortableHeader({
  title,
  canSort,
  isSorted,
  toggleSorting,
}: {
  title: string
  canSort: boolean
  isSorted: false | "asc" | "desc"
  toggleSorting: (desc?: boolean) => void
}) {
  if (!canSort) {
    return <span>{title}</span>
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-3 h-8"
      onClick={() => toggleSorting(isSorted === "asc")}
    >
      <span>{title}</span>
      <ArrowUpDown className="ml-2 h-4 w-4" />
    </Button>
  )
}
