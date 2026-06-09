"use client";

import { AdminPanel } from "@/components/AdminPanel";
import { AdminOnly } from "@/components/AdminOnly";
import { useSettings } from "@/hooks/useSettings";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { ProfileMetaCard } from "@/components/product/ProfileMetaCard";
import { CURRENT_USER_ID } from "@/lib/runtime";

export default function AdminPage() {
  const { t } = useI18n();
  const profile = useProfile();
  const { settings, loading, error, saveSettings } = useSettings(CURRENT_USER_ID);

  return (
    <main className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("adminPage.title")}</h1>
          <p className="mt-1 text-muted-foreground">{t("adminPage.subtitle")}</p>
        </div>
      </div>

      <ProfileMetaCard
        heading={t("adminPage.profileLabel", { profile: profile.displayName })}
        detail={t("adminPage.frameworkVersionLabel", { version: profile.frameworkVersion })}
      />

      {error && (
        <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
          <p className="mb-1 font-semibold">{t("common.error")}</p>
          <p>{error}</p>
        </div>
      )}

      <AdminOnly fallback={
        <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
          <p className="mb-1 font-semibold">{t("adminPage.accessDenied")}</p>
        </div>
      }>
        <AdminPanel
          settings={settings}
          loading={loading}
          onSave={saveSettings}
        />
      </AdminOnly>
      </div>
    </main>
  );
}
