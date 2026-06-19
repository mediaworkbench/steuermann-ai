"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Copy, KeyRound, Loader2, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable } from "@/components/ui/data-table";
import { useCurrentUser } from "@/context/SessionContext";
import { createColumns } from "@/app/admin/users/columns";
import {
  type AdminRole,
  type AdminUser,
  createUser,
  deleteUser,
  fetchRoles,
  fetchUsers,
  updateUser,
} from "@/lib/api";

const STATUSES = ["active", "inactive", "suspended"] as const;
const DEFAULT_ROLES = ["user", "researcher", "administrator"];

interface TempPassword {
  username: string;
  password: string;
}

export function AdminUsersPanel() {
  const { userId: selfId } = useCurrentUser();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [roles, setRoles] = useState<string[]>(DEFAULT_ROLES);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [newUsername, setNewUsername] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newRole, setNewRole] = useState("user");
  const [creating, setCreating] = useState(false);

  const [busyId, setBusyId] = useState<string | null>(null);
  const [tempPassword, setTempPassword] = useState<TempPassword | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    const [data, roleList] = await Promise.all([fetchUsers(500, 0), fetchRoles()]);
    if (data) {
      setUsers(data.users);
    } else {
      setError("Failed to load users.");
    }
    if (roleList.length) {
      setRoles(roleList.map((r: AdminRole) => r.role_name));
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleCreate = useCallback(async () => {
    const username = newUsername.trim();
    const email = newEmail.trim();
    if (!username || !email) {
      toast.error("Username and email are required.");
      return;
    }
    setCreating(true);
    try {
      const result = await createUser({ username, email, role: newRole });
      setTempPassword({ username: result.user.username, password: result.temporary_password });
      setNewUsername("");
      setNewEmail("");
      setNewRole("user");
      await load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create user.");
    } finally {
      setCreating(false);
    }
  }, [newUsername, newEmail, newRole, load]);

  const runUpdate = useCallback(
    async (user: AdminUser, body: { role?: string; status?: string; reset_password?: boolean }, successMsg: string) => {
      setBusyId(user.user_id);
      try {
        const result = await updateUser(user.user_id, body);
        if (result.temporary_password) {
          setTempPassword({ username: result.user.username, password: result.temporary_password });
        }
        toast.success(successMsg);
        await load();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Update failed.");
      } finally {
        setBusyId(null);
      }
    },
    [load],
  );

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    const target = deleteTarget;
    setDeleteTarget(null);
    setBusyId(target.user_id);
    try {
      await deleteUser(target.user_id);
      toast.success(`Deleted ${target.username}.`);
      await load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setBusyId(null);
    }
  }, [deleteTarget, load]);

  const columns = useMemo(
    () => createColumns({
      selfId: selfId ?? "",
      roles,
      statuses: STATUSES,
      busyId,
      onUpdate: runUpdate,
      onDelete: setDeleteTarget,
    }),
    [selfId, roles, busyId, runUpdate],
  );

  return (
    <div className="space-y-8">
      {/* Create user */}
      <Card>
        <CardHeader>
          <CardTitle>Create user</CardTitle>
          <CardDescription>
            A temporary password is generated and shown once. The user must change it on first login.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-[1fr_1fr_auto_auto] sm:items-end">
            <div>
              <Label htmlFor="new-username" className="mb-2 block">Username</Label>
              <Input
                id="new-username"
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                placeholder="e.g., jdoe"
              />
            </div>
            <div>
              <Label htmlFor="new-email" className="mb-2 block">Email</Label>
              <Input
                id="new-email"
                type="email"
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
                placeholder="user@example.com"
              />
            </div>
            <div>
              <Label htmlFor="new-role" className="mb-2 block">Role</Label>
              <Select value={newRole} onValueChange={setNewRole}>
                <SelectTrigger id="new-role" className="w-full">
                  <SelectValue placeholder="Select role" />
                </SelectTrigger>
                <SelectContent>
                  {roles.map((r) => (
                    <SelectItem key={r} value={r}>{r}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button variant="primary" onClick={handleCreate} disabled={creating}>
              {creating ? <Loader2 className="size-4 animate-spin" /> : <UserPlus className="size-4" />}
              <span>Create</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* User list */}
      <Card>
        <CardHeader>
          <CardTitle>Users</CardTitle>
          <CardDescription>Assign roles, change status, reset passwords, or remove users.</CardDescription>
        </CardHeader>
        <CardContent>
          {loading && <p className="text-sm text-muted-foreground">Loading users…</p>}
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
          {!loading && !error && (
            <DataTable
              columns={columns}
              data={users}
              loading={false}
              emptyText="No users found."
              disablePagination
            />
          )}
        </CardContent>
      </Card>

      {/* Temporary password (shown once) */}
      <AlertDialog open={tempPassword !== null} onOpenChange={(open) => { if (!open) setTempPassword(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia>
              <KeyRound className="size-6 text-primary" />
            </AlertDialogMedia>
            <AlertDialogTitle>Temporary password</AlertDialogTitle>
            <AlertDialogDescription>
              Share this with <span className="font-medium">{tempPassword?.username}</span>. It is shown only
              once; they must change it on first login.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="flex items-center gap-2 px-2">
            <code className="flex-1 rounded-md bg-muted px-3 py-2 font-mono text-sm break-all">
              {tempPassword?.password}
            </code>
            <Button
              variant="secondary"
              size="icon"
              aria-label="Copy password"
              onClick={() => {
                if (tempPassword) {
                  navigator.clipboard?.writeText(tempPassword.password);
                  toast.success("Copied to clipboard.");
                }
              }}
            >
              <Copy className="size-4" />
            </Button>
          </div>
          <AlertDialogFooter>
            <Button variant="primary" size="md" onClick={() => setTempPassword(null)}>
              Done
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete confirmation */}
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        variant="danger"
        title="Delete user"
        message={`Permanently delete ${deleteTarget?.username ?? "this user"}? This cannot be undone.`}
        confirmLabel="Delete"
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
