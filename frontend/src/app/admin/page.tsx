"use client";

import { AdminPanel } from "@/components/AdminPanel";
import { AdminOnly } from "@/components/AdminOnly";
import { useSettings } from "@/hooks/useSettings";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { PageShell } from "@/components/product/PageShell";
import { PageHeader } from "@/components/product/PageHeader";
import { PageErrorAlert } from "@/components/product/PageErrorAlert";
import { CURRENT_USER_ID } from "@/lib/runtime";
import styles from "../settings/Settings.module.css";

export default function AdminPage() {
  const { t } = useI18n();
  const profile = useProfile();
  const { settings, loading, error, saveSettings } = useSettings(CURRENT_USER_ID);

  return (
    <PageShell contentClassName="space-y-8 lg:px-12">
      <PageHeader title={t("adminPage.title")} subtitle={t("adminPage.subtitle")} />

      <section className={styles.accountCard}>
        <div className={styles.accountInfo}>
          <h2 className={styles.accountName}>{t("adminPage.profileLabel", { profile: profile.displayName })}</h2>
          <p className={styles.versionText}>{t("adminPage.frameworkVersionLabel", { version: profile.frameworkVersion })}</p>
        </div>
      </section>

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
