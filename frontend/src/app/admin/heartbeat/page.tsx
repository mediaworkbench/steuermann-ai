"use client";

import { AdminOnly } from "@/components/AdminOnly";
import { HeartbeatSettingsSection } from "@/components/product/HeartbeatSettingsSection";
import { useI18n } from "@/hooks/useI18n";

export default function AdminHeartbeatPage() {
  const { t } = useI18n();

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("adminPage.heartbeatSection")}</h1>
          <p className="mt-1 text-muted-foreground">{t("adminPage.heartbeatDescription")}</p>
        </div>

        <AdminOnly
          fallback={
            <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
              <p className="mb-1 font-semibold">{t("adminPage.accessDenied")}</p>
            </div>
          }
        >
          <HeartbeatSettingsSection />
        </AdminOnly>
      </div>
    </div>
  );
}
