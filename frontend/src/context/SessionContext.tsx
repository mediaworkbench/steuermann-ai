"use client";

import { createContext, useContext, useEffect, useState } from "react";
import type { UserRole } from "@/lib/auth/session";
import { hardNavigate } from "@/lib/navigation";

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

    // `initial` gates the redirect-on-logout: on first load an unauthenticated result is
    // expected (the middleware sends those to /login), but a *later* refetch that comes back
    // empty means the session was revoked (suspended/deleted) → log the user out.
    async function refresh(initial: boolean) {
      try {
        const res = await fetch("/api/auth/session", { cache: "no-store" });
        const data = (await res.json()) as {
          enabled?: boolean;
          user?: Partial<CurrentUser> | null;
        };
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
        } else if (!initial && data?.enabled) {
          // Session was revoked mid-use — the route cleared the cookie; bounce to login.
          hardNavigate("/login");
        }
      } catch {
        /* leave state as-is; middleware handles unauthenticated page loads */
      } finally {
        if (!cancelled && initial) setLoading(false);
      }
    }

    refresh(true);

    // Re-validate when the tab regains focus so role changes / revocations propagate
    // without a full reload.
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh(false);
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisible);
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
