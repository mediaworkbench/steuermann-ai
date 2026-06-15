"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Copy, KeyRound, Loader2, Trash2, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
import { useCurrentUser } from "@/context/SessionContext";
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
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Username</TableHead>
                  <TableHead>Email</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Flags</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.map((u) => {
                  const isSelf = u.user_id === selfId;
                  const rowBusy = busyId === u.user_id;
                  return (
                    <TableRow key={u.user_id}>
                      <TableCell className="font-medium">
                        {u.username}
                        {isSelf && <span className="ml-2 text-xs text-muted-foreground">(you)</span>}
                      </TableCell>
                      <TableCell className="text-muted-foreground">{u.email}</TableCell>
                      <TableCell>
                        <Select
                          value={u.role_name ?? "user"}
                          disabled={rowBusy || isSelf}
                          onValueChange={(v) => runUpdate(u, { role: v }, "Role updated.")}
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
                      </TableCell>
                      <TableCell>
                        <Select
                          value={u.status}
                          disabled={rowBusy || isSelf}
                          onValueChange={(v) => runUpdate(u, { status: v }, "Status updated.")}
                        >
                          <SelectTrigger aria-label={`Status for ${u.username}`} className="w-full">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {STATUSES.map((s) => (
                              <SelectItem key={s} value={s}>{s}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell>
                        {u.must_change_password && <Badge variant="outline">must change pw</Badge>}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            variant="secondary"
                            size="sm"
                            disabled={rowBusy}
                            onClick={() => runUpdate(u, { reset_password: true }, "Password reset.")}
                          >
                            <KeyRound className="size-4" />
                            <span>Reset password</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label={`Delete ${u.username}`}
                            disabled={rowBusy || isSelf}
                            onClick={() => setDeleteTarget(u)}
                          >
                            <Trash2 className="size-4 text-destructive" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
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
