"use client";

import { AdminOnly } from "@/components/AdminOnly";
import { DreamingMetricsSection } from "@/components/product/DreamingMetricsSection";
import { useI18n } from "@/hooks/useI18n";

/**
 * Admin-only Dreaming Engine dashboard (plan-memory.md Phase 6) — aggregate,
 * anonymized metrics only. Admin-gated by the proxy `/admin/*` rule + AdminOnly.
 */
export default function AdminDreamingMetricsPage() {
  const { t } = useI18n();

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("dreaming.adminTitle")}</h1>
          <p className="mt-1 text-muted-foreground">{t("dreaming.adminDescription")}</p>
        </div>

        <AdminOnly
          fallback={
            <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
              <p className="mb-1 font-semibold">{t("adminPage.accessDenied")}</p>
            </div>
          }
        >
          <DreamingMetricsSection />
        </AdminOnly>
      </div>
    </div>
  );
}
