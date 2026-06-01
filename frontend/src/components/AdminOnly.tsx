"use client";

import { useRole } from "@/context/RoleContext";

interface AdminOnlyProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function AdminOnly({ children, fallback = null }: AdminOnlyProps) {
  const { isAdmin, roleLoading } = useRole();

  if (roleLoading) return null;
  if (!isAdmin) return <>{fallback}</>;

  return <>{children}</>;
}
