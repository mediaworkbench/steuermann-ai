"use client";

import { createContext, useContext, useEffect, useState } from "react";
import type { UserRole } from "@/lib/auth/session";
import { AUTH_ENABLED, DEV_ROLE } from "@/lib/runtime";

interface RoleContextValue {
  role: UserRole;
  isAdmin: boolean;
  roleLoading: boolean;
}

const RoleContext = createContext<RoleContextValue>({
  role: "user",
  isAdmin: false,
  roleLoading: true,
});

export function RoleProvider({ children }: { children: React.ReactNode }) {
  // Lazy init: when auth is disabled the role is known synchronously from the env var,
  // so we start resolved to avoid a flash where admin nav links are absent then appear.
  const [role, setRole] = useState<UserRole>(() => (!AUTH_ENABLED ? DEV_ROLE : "user"));
  const [roleLoading, setRoleLoading] = useState(() => AUTH_ENABLED);

  useEffect(() => {
    if (!AUTH_ENABLED) return;

    fetch("/api/auth/session")
      .then((res) => res.json())
      .then((data: { role?: string }) => {
        setRole(data.role === "administrator" ? "administrator" : "user");
      })
      .catch(() => {
        setRole("user");
      })
      .finally(() => {
        setRoleLoading(false);
      });
  }, []);

  return (
    <RoleContext.Provider value={{ role, isAdmin: role === "administrator", roleLoading }}>
      {children}
    </RoleContext.Provider>
  );
}

export function useRole(): RoleContextValue {
  return useContext(RoleContext);
}
