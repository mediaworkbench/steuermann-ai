"use client";

import { AdminPanel } from "@/components/AdminPanel";
import { AdminOnly } from "@/components/AdminOnly";
import { useSettings } from "@/hooks/useSettings";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import { PageShell } from "@/components/product/PageShell";
import { PageHeader } from "@/components/product/PageHeader";
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
          <h2 className={styles.accountName}>Profile: {profile.displayName}</h2>
          <p className={styles.versionText}>Framework version: {profile.frameworkVersion}</p>
        </div>
      </section>

      {error && (
        <div className={styles.error}>
          <p className={styles.errorTitle}>{t("common.error")}</p>
          <p>{error}</p>
        </div>
      )}

      <AdminOnly fallback={
        <div className={styles.error}>
          <p className={styles.errorTitle}>{t("adminPage.accessDenied")}</p>
        </div>
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
