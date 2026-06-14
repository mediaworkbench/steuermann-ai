"use client";

import { createContext, useContext, useEffect, useState } from "react";
import type { UserRole } from "@/lib/auth/session";

const VALID_ROLES: readonly UserRole[] = ["user", "researcher", "administrator"];

function normalizeRole(value: unknown): UserRole {
  return VALID_ROLES.includes(value as UserRole) ? (value as UserRole) : "user";
}

export interface CurrentUser {
  userId: string;
  username: string;
  displayName: string;
  email: string;
  role: UserRole;
  mustChangePassword: boolean;
}

interface SessionContextValue extends CurrentUser {
  loading: boolean;
  isAdmin: boolean;
  canAccessRag: boolean;
}

const EMPTY: CurrentUser = {
  userId: "",
  username: "",
  displayName: "",
  email: "",
  role: "user",
  mustChangePassword: false,
};

const SessionContext = createContext<SessionContextValue>({
  ...EMPTY,
  loading: true,
  isAdmin: false,
  canAccessRag: false,
});

/**
 * Single source of client-side identity + role, fed by `GET /api/auth/session`
 * (which returns the dev-bypass identity when auth is disabled, so this works in both
 * modes). Render is gated until the session resolves so every child sees a stable
 * identity and role — no flash of the wrong nav and no API calls with an undefined id.
 */
export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<CurrentUser>(EMPTY);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/auth/session")
      .then((res) => res.json())
      .then((data: { user?: Partial<CurrentUser> | null }) => {
        if (cancelled) return;
        const u = data?.user;
        if (u && u.userId) {
          setUser({
            userId: u.userId,
            username: u.username ?? u.userId,
            displayName: u.displayName ?? u.username ?? u.userId,
            email: u.email ?? "",
            role: normalizeRole(u.role),
            mustChangePassword: Boolean(u.mustChangePassword),
          });
        }
      })
      .catch(() => {
        /* leave EMPTY; middleware redirects unauthenticated page loads to /login */
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div
        className="flex min-h-screen items-center justify-center bg-background"
        role="status"
        aria-live="polite"
      >
        <span className="text-sm text-muted-foreground">Loading…</span>
      </div>
    );
  }

  const value: SessionContextValue = {
    ...user,
    loading,
    isAdmin: user.role === "administrator",
    canAccessRag: user.role === "researcher" || user.role === "administrator",
  };

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  return useContext(SessionContext);
}

export function useCurrentUser(): CurrentUser {
  const { userId, username, displayName, email, role, mustChangePassword } =
    useContext(SessionContext);
  return { userId, username, displayName, email, role, mustChangePassword };
}
