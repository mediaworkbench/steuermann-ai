"use client"

import type { ColumnDef } from "@tanstack/react-table"
import { KeyRound, Loader2, Trash2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { AdminUser } from "@/lib/api"

export function createColumns(
  deps: {
    selfId: string
    roles: string[]
    statuses: readonly string[]
    busyId: string | null
    onUpdate: (user: AdminUser, body: { role?: string; status?: string; reset_password?: boolean }, successMsg: string) => void
    onDelete: (user: AdminUser) => void
  }
): ColumnDef<AdminUser>[] {
  const { selfId, roles, statuses, busyId, onUpdate, onDelete } = deps

  return [
    {
      accessorKey: "username",
      header: "Username",
      cell: ({ row }) => {
        const u = row.original
        const isSelf = u.user_id === selfId
        return (
          <span className="font-medium">
            {u.username}
            {isSelf && <span className="ml-2 text-xs text-muted-foreground">(you)</span>}
          </span>
        )
      },
    },
    {
      accessorKey: "email",
      header: "Email",
      cell: ({ row }) => (
        <span className="text-muted-foreground">{row.original.email}</span>
      ),
    },
    {
      accessorKey: "role_name",
      header: "Role",
      cell: ({ row }) => {
        const u = row.original
        const isSelf = u.user_id === selfId
        const rowBusy = busyId === u.user_id
        return (
          <Select
            value={u.role_name ?? "user"}
            disabled={rowBusy || isSelf}
            onValueChange={(v) => onUpdate(u, { role: v }, "Role updated.")}
          >
            <SelectTrigger aria-label={`Role for ${u.username}`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {roles.map((r) => (
                <SelectItem key={r} value={r}>{r}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )
      },
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => {
        const u = row.original
        const isSelf = u.user_id === selfId
        const rowBusy = busyId === u.user_id
        return (
          <Select
            value={u.status}
            disabled={rowBusy || isSelf}
            onValueChange={(v) => onUpdate(u, { status: v }, "Status updated.")}
          >
            <SelectTrigger aria-label={`Status for ${u.username}`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {statuses.map((s) => (
                <SelectItem key={s} value={s}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )
      },
    },
    {
      id: "flags",
      header: "Flags",
      cell: ({ row }) => {
        const u = row.original
        return u.must_change_password ? (
          <Badge variant="outline">must change pw</Badge>
        ) : null
      },
    },
    {
      id: "actions",
      header: () => <div className="text-right">Actions</div>,
      cell: ({ row }) => {
        const u = row.original
        const isSelf = u.user_id === selfId
        const rowBusy = busyId === u.user_id
        return (
          <div className="flex items-center justify-end gap-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={rowBusy}
              onClick={() => onUpdate(u, { reset_password: true }, "Password reset.")}
            >
              {rowBusy ? <Loader2 className="size-4 animate-spin" /> : <KeyRound className="size-4" />}
              <span>Reset password</span>
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Delete ${u.username}`}
              disabled={rowBusy || isSelf}
              onClick={() => onDelete(u)}
            >
              <Trash2 className="size-4 text-destructive" />
            </Button>
          </div>
        )
      },
    },
  ]
}
