"use client";

import { useState } from "react";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useSettings } from "@/hooks/useSettings";
import { useI18n } from "@/hooks/useI18n";
import { useRole } from "@/context/RoleContext";
import {
  CURRENT_USER_ID,
  SINGLE_USER_DISPLAY_NAME,
} from "@/lib/runtime";
import styles from "./Settings.module.css";

export default function SettingsPage() {
  const [userId] = useState(CURRENT_USER_ID);
  const { t } = useI18n();
  const { role } = useRole();
  const { settings, loading, error, saveSettings } = useSettings(userId);

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <h1 className={styles.title}>{t("settingsPage.title")}</h1>
        <p className={styles.subtitle}>
          {SINGLE_USER_DISPLAY_NAME}
          <span className="ml-2 text-sm font-normal capitalize opacity-60">· {role}</span>
        </p>
      </div>

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
