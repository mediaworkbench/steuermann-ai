"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { GroupedToolChecklist } from "@/components/product/GroupedToolChecklist";
import { fetchRoleTools, updateRoleTools, type ToolCatalogItem } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

// Administrators are always unrestricted, so only these roles are configurable.
const CONFIGURABLE_ROLES = ["user", "researcher"] as const;
type ConfigurableRole = (typeof CONFIGURABLE_ROLES)[number];

/**
 * Admin-only editor for per-role tool access. For each configurable role the
 * full tool catalog is shown as a grouped 3-column checklist; checking a tool
 * allows that role to use it (users still toggle within their allowed set).
 */
export function RoleToolPermissionsSection() {
  const { t } = useI18n();
  const [tools, setTools] = useState<ToolCatalogItem[]>([]);
  const [roleAllowed, setRoleAllowed] = useState<Record<string, Set<string>>>({});
  const [loading, setLoading] = useState(true);
  const [savingRole, setSavingRole] = useState<string | null>(null);

  const applyRoles = useCallback((roles: Record<string, string[]>) => {
    const next: Record<string, Set<string>> = {};
    for (const role of CONFIGURABLE_ROLES) next[role] = new Set(roles[role] ?? []);
    setRoleAllowed(next);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchRoleTools().then((data) => {
      if (cancelled) return;
      if (!data) {
        toast.error(t("roleTools.loadFailed"));
        setLoading(false);
        return;
      }
      setTools(data.tools);
      applyRoles(data.roles);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [t, applyRoles]);

  const handleToggle = useCallback((role: ConfigurableRole, toolId: string) => {
    setRoleAllowed((prev) => {
      const set = new Set(prev[role] ?? []);
      if (set.has(toolId)) set.delete(toolId);
      else set.add(toolId);
      return { ...prev, [role]: set };
    });
  }, []);

  const handleSave = useCallback(
    async (role: ConfigurableRole) => {
      setSavingRole(role);
      try {
        const result = await updateRoleTools(role, Array.from(roleAllowed[role] ?? []));
        if (result) {
          applyRoles(result.roles);
          toast.success(t("roleTools.saved"));
        } else {
          toast.error(t("roleTools.saveFailed"));
        }
      } finally {
        setSavingRole(null);
      }
    },
    [roleAllowed, applyRoles, t]
  );

  const roleLabel = (role: ConfigurableRole) =>
    role === "user" ? t("roleTools.roleUser") : t("roleTools.roleResearcher");

  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle>{t("roleTools.title")}</CardTitle>
        <CardDescription>{t("roleTools.description")}</CardDescription>
      </CardHeader>
      <div className="space-y-8 px-6 pb-6">
        {loading ? (
          <p className="text-sm text-muted-foreground">{t("roleTools.loading")}</p>
        ) : (
          CONFIGURABLE_ROLES.map((role) => (
            <div key={role} className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-foreground">{roleLabel(role)}</h3>
                <Button
                  type="button"
                  size="sm"
                  onClick={() => handleSave(role)}
                  disabled={savingRole === role}
                >
                  {savingRole === role ? t("common.saving") : t("roleTools.save")}
                </Button>
              </div>
              <GroupedToolChecklist
                tools={tools}
                isChecked={(toolId) => roleAllowed[role]?.has(toolId) ?? false}
                onToggle={(toolId) => handleToggle(role, toolId)}
              />
            </div>
          ))
        )}
      </div>
    </Card>
  );
}
