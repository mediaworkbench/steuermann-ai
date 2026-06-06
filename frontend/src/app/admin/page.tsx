"use client";

import { AdminPanel } from "@/components/AdminPanel";
import { AdminOnly } from "@/components/AdminOnly";
import { useSettings } from "@/hooks/useSettings";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { PageShell } from "@/components/product/PageShell";
import { PageHeader } from "@/components/product/PageHeader";
import { PageErrorAlert } from "@/components/product/PageErrorAlert";
import { ProfileMetaCard } from "@/components/product/ProfileMetaCard";
import { CURRENT_USER_ID } from "@/lib/runtime";

export default function AdminPage() {
  const { t } = useI18n();
  const profile = useProfile();
  const { settings, loading, error, saveSettings } = useSettings(CURRENT_USER_ID);

  return (
    <PageShell contentClassName="space-y-8 lg:px-12">
      <PageHeader title={t("adminPage.title")} subtitle={t("adminPage.subtitle")} />

      <ProfileMetaCard
        heading={t("adminPage.profileLabel", { profile: profile.displayName })}
        detail={t("adminPage.frameworkVersionLabel", { version: profile.frameworkVersion })}
      />

      {error && (
        <PageErrorAlert title={t("common.error")} message={error} />
      )}

      <AdminOnly fallback={
        <PageErrorAlert title={t("adminPage.accessDenied")} />
      }>
        <AdminPanel
          settings={settings}
          loading={loading}
          onSave={saveSettings}
        />
      </AdminOnly>
    </PageShell>
  );
}
