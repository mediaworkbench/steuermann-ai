"use client";

import type { UserRole } from "@/lib/auth/session";
import { useSession } from "@/context/SessionContext";

interface RoleContextValue {
  role: UserRole;
  isAdmin: boolean;
  canAccessRag: boolean;
  roleLoading: boolean;
}

/**
 * Thin role view over {@link useSession}. Identity/role now come from a single
 * source (`SessionProvider`), so role state is always resolved by the time children
 * render (the provider gates on load) — `roleLoading` is kept for API compatibility.
 */
export function useRole(): RoleContextValue {
  const { role, isAdmin, canAccessRag, loading } = useSession();
  return { role, isAdmin, canAccessRag, roleLoading: loading };
}
