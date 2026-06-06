"use client";

import { useState } from "react";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useSettings } from "@/hooks/useSettings";
import { useI18n } from "@/hooks/useI18n";
import { useRole } from "@/context/RoleContext";
import { PageShell } from "@/components/product/PageShell";
import { PageHeader } from "@/components/product/PageHeader";
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
    <PageShell contentClassName="space-y-8 lg:px-12">
      <PageHeader
        title={t("settingsPage.title")}
        subtitle={
          <>
            {SINGLE_USER_DISPLAY_NAME}
            <span className="ml-2 text-sm font-normal capitalize opacity-60">· {role}</span>
          </>
        }
      />

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
    </PageShell>
  );
}
