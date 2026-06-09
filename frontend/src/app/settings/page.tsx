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


export default function SettingsPage() {
  const [userId] = useState(CURRENT_USER_ID);
  const { t } = useI18n();
  const { role } = useRole();
  const { settings, loading, error, saveSettings } = useSettings(userId);

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="mx-auto w-full px-4 py-6 md:px-8 md:py-8 space-y-8 lg:px-12">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t("settingsPage.title")}</h1>
          <p className="mt-1 text-muted-foreground">
            {SINGLE_USER_DISPLAY_NAME}
            <span className="ml-2 text-sm font-normal capitalize opacity-60">· {role}</span>
          </p>
        </div>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive">
          <p className="mb-1 font-semibold">{t("common.error")}</p>
          <p>{error}</p>
        </div>
      )}

      <SettingsPanel
        settings={settings}
        loading={loading}
        onSave={saveSettings}
      />

      {process.env.NODE_ENV === "development" && settings && (
        <details className="mt-8 bg-card p-4 rounded-lg text-sm">
          <summary className="cursor-pointer font-semibold text-accent">{t("settingsPage.debugCurrentSettings")}</summary>
          <pre className="mt-2 overflow-auto max-h-[400px] text-xs">{JSON.stringify(settings, null, 2)}</pre>
        </details>
      )}
      </div>
    </div>
  );
}
