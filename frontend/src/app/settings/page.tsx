"use client";

import { useState } from "react";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useSettings } from "@/hooks/useSettings";
import { useProfile } from "@/hooks/useProfile";
import { useI18n } from "@/hooks/useI18n";
import {
  CURRENT_USER_ID,
} from "@/lib/runtime";
import styles from "./Settings.module.css";

export default function SettingsPage() {
  const [userId] = useState(CURRENT_USER_ID);
  const profile = useProfile();
  const profileDisplayName = profile.displayName;
  const frameworkVersion = profile.frameworkVersion;
  const { t } = useI18n();
  const { settings, loading, error, saveSettings } = useSettings(userId);

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={styles.title}>{t("settingsPage.title")}</h1>
      </div>

      <section className={styles.accountCard}>
        <div className={styles.accountInfo}>
          <h2 className={styles.accountName}>Profile: {profileDisplayName}</h2>
          <p className={styles.versionText}>Framework version: {frameworkVersion}</p>
        </div>
      </section>

      {error && (
        <div className={styles.error}>
          <p className={styles.errorTitle}>{t("common.error")}</p>
          <p>{error}</p>
        </div>
      )}

      <SettingsPanel
        settings={settings}
        loading={loading}
        onSave={saveSettings}
      />

      {process.env.NODE_ENV === "development" && settings && (
        <details className={styles.debug}>
          <summary>{t("settingsPage.debugCurrentSettings")}</summary>
          <pre>{JSON.stringify(settings, null, 2)}</pre>
        </details>
      )}
    </main>
  );
}
