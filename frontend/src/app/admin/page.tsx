"use client";

import { AdminPanel } from "@/components/AdminPanel";
import { AdminOnly } from "@/components/AdminOnly";
import { useSettings } from "@/hooks/useSettings";
import { useI18n } from "@/hooks/useI18n";
import { CURRENT_USER_ID } from "@/lib/runtime";
import styles from "../settings/Settings.module.css";

export default function AdminPage() {
  const { t } = useI18n();
  const { settings, loading, error, saveSettings } = useSettings(CURRENT_USER_ID);

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={styles.title}>{t("adminPage.title")}</h1>
        <p className={styles.subtitle}>{t("adminPage.subtitle")}</p>
      </div>

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
    </main>
  );
}
