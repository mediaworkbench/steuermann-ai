"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  UserSettings,
  UserResetOptions,
  fetchSystemConfig,
  resetMyData,
  type SystemConfig,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { DangerConfirmDialog } from "@/components/product/DangerConfirmDialog";
import { DangerOptionsList } from "@/components/product/DangerOptionsList";
import { DangerSelectionActions } from "@/components/product/DangerSelectionActions";
import { OptionChecklist } from "@/components/product/OptionChecklist";
import { OptionCheckboxRow } from "@/components/product/OptionCheckboxRow";
import { PrimarySaveBar } from "@/components/product/PrimarySaveBar";
import { RoleModelSelectionSection } from "@/components/product/RoleModelSelectionSection";
import { SectionStateText } from "@/components/product/SectionStateText";
import { updatePreferredModelSelection } from "@/components/product/modelSelection";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select } from "@/components/ui/select";

export interface SettingsPanelProps {
  settings: UserSettings | null;
  loading: boolean;
  onSave: (updates: Partial<Omit<UserSettings, "user_id" | "updated_at">>) => Promise<boolean>;
}

const LANGUAGE_LABELS: Record<string, string> = {
  en: "English",
  de: "Deutsch",
  fr: "Français",
  es: "Español",
};

const FALLBACK_TOOLS = [
  { id: "web_search_mcp", label: "Web Search" },
  { id: "extract_webpage_mcp", label: "Extract Webpage" },
  { id: "analyze_image_tool", label: "Analyze Image" },
  { id: "ocr_tool", label: "OCR" },
  { id: "analyze_document_tool", label: "Analyze Document" },
  { id: "analyze_chart_tool", label: "Analyze Chart" },
  { id: "image_metadata_tool", label: "Image Metadata" },
  { id: "read_barcodes_tool", label: "Read Barcodes" },
  { id: "datetime_tool", label: "Datetime" },
  { id: "calculator_tool", label: "Calculator" },
  { id: "map_tool", label: "Map" },
  { id: "file_ops_tool", label: "File Ops" },
];

const USER_MODEL_ROLES = ["chat"];

export function SettingsPanel({ settings, loading, onSave }: SettingsPanelProps) {
  const { t, formatDateTime } = useI18n();
  const [toolToggles, setToolToggles] = useState<Record<string, boolean>>(
    settings?.tool_toggles || {}
  );
  const [ragConfig, setRagConfig] = useState<Record<string, unknown>>(
    settings?.rag_config || { collection: "", top_k: 5, enabled: true }
  );
  const [preferredModels, setPreferredModels] = useState<Record<string, string | null>>(
    settings?.preferred_models || {}
  );
  const [language, setLanguage] = useState(settings?.language || "en");
  const [analyticsPreferences, setAnalyticsPreferences] = useState<Record<string, unknown>>(
    settings?.analytics_preferences || {}
  );
  const [saving, setSaving] = useState(false);
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [myDataOptions, setMyDataOptions] = useState<UserResetOptions>({
    conversations: true,
    workspace: true,
    memories: true,
  });
  const [myDataConfirmOpen, setMyDataConfirmOpen] = useState(false);
  const [myDataResetting, setMyDataResetting] = useState(false);

  // Read-modify-write: sync all fields from server so saves don't wipe admin-owned keys
  useEffect(() => {
    if (settings) {
      setToolToggles(settings.tool_toggles || {});
      const ragDefault = { collection: systemConfig?.rag_defaults.collection_name || "framework", top_k: systemConfig?.rag_defaults.top_k || 5 };
      setRagConfig(settings.rag_config && settings.rag_config.collection ? settings.rag_config : ragDefault);
      setPreferredModels(settings.preferred_models || {});
      setLanguage(settings.language || "en");
      setAnalyticsPreferences(settings.analytics_preferences || {});
    }
  }, [settings, systemConfig]);

  useEffect(() => {
    let cancelled = false;
    async function loadConfig() {
      setConfigLoading(true);
      const config = await fetchSystemConfig();
      if (!cancelled) {
        setSystemConfig(config);
        setConfigLoading(false);
      }
    }
    loadConfig();
    return () => { cancelled = true; };
  }, []);

  const handleToolToggle = useCallback((toolId: string) => {
    setToolToggles((prev) => ({ ...prev, [toolId]: !prev[toolId] }));
  }, []);

  const handleRagConfigChange = useCallback((key: string, value: unknown) => {
    setRagConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Save: send the full settings object (read-modify-write — preserves admin-owned fields)
  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const success = await onSave({
        tool_toggles: toolToggles,
        rag_config: ragConfig,
        preferred_model: preferredModels.chat || null,
        preferred_models: preferredModels,
        language,
        analytics_preferences: analyticsPreferences,
      });
      if (success) {
        toast.success(t("settingsPanel.settingsSaved"));
      } else {
        toast.error(t("settingsPanel.failedToSaveSettings"));
      }
    } finally {
      setSaving(false);
    }
  }, [toolToggles, ragConfig, preferredModels, language, analyticsPreferences, onSave, t]);

  const handleMyDataReset = useCallback(async () => {
    setMyDataConfirmOpen(false);
    setMyDataResetting(true);
    try {
      const result = await resetMyData(myDataOptions);
      if (result.status === "ok") {
        toast.success(t("settingsPanel.myDataResetSuccess"));
      } else {
        toast.warning(t("settingsPanel.myDataResetFailed"), { description: result.errors.join(", ") });
      }
    } catch (err) {
      console.error("reset_my_data failed", err);
      toast.error(t("settingsPanel.myDataResetFailed"));
    } finally {
      setMyDataResetting(false);
    }
  }, [myDataOptions, t]);

  if (loading) {
    return <div className="space-y-3 py-4"><Skeleton className="h-4 w-3/4" /><Skeleton className="h-4 w-1/2" /><p className="text-sm text-muted-foreground">{t("common.loading")}</p></div>;
  }

  const chatModelOptions = (systemConfig?.model_roles || []).filter((r) =>
    USER_MODEL_ROLES.includes(r.role)
  );

  return (
    <div className="space-y-6">

      {/* Language */}
      <Card>
        <CardHeader>
          <CardTitle>{t("settingsPanel.language")}</CardTitle>
        </CardHeader>
        <div className="px-6 pb-6">
        <Select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
        >
          {(systemConfig?.supported_languages || ["en"]).map((code) => (
            <option key={code} value={code}>
              {LANGUAGE_LABELS[code] || code.toUpperCase()}
            </option>
          ))}
        </Select>
        </div>
      </Card>

      {/* Sound */}
      <Card>
        <CardHeader>
          <CardTitle>{t("settingsPanel.soundSection")}</CardTitle>
        </CardHeader>
        <div className="px-6 pb-6">
        <OptionCheckboxRow
          checked={(analyticsPreferences.sound_enabled as boolean) ?? true}
          onToggle={() =>
            setAnalyticsPreferences((prev) => ({
              ...prev,
              sound_enabled: !((prev.sound_enabled as boolean) ?? true),
            }))
          }
          label={t("settingsPanel.soundEnabled")}
          description={t("settingsPanel.soundEnabledDescription")}
        />
        </div>
      </Card>

      {/* Tool Toggles */}
      <Card>
        <CardHeader>
          <CardTitle>{t("settingsPanel.toolSettings")}</CardTitle>
        </CardHeader>
        <div className="px-6 pb-6">
        {configLoading ? (
          <SectionStateText>{t("settingsPanel.loadingTools")}</SectionStateText>
        ) : (
          <OptionChecklist
            items={(systemConfig?.available_tools || FALLBACK_TOOLS).map((tool) => ({
              key: tool.id,
              checked: toolToggles[tool.id] ?? true,
              onToggle: () => handleToolToggle(tool.id),
              label: tool.label,
              alignment: "center" as const,
              checkboxClassName: "w-5 h-5",
            }))}
          />
        )}
        </div>
      </Card>

      {/* RAG — user-scoped controls: enabled toggle + top_k */}
      <Card>
        <CardHeader>
          <CardTitle>{t("settingsPanel.ragConfiguration")}</CardTitle>
        </CardHeader>
        <div className="px-6 pb-6">
        {configLoading && <SectionStateText>{t("settingsPanel.loadingDefaults")}</SectionStateText>}
        <div className="space-y-4">
          <div>
            <OptionCheckboxRow
              checked={(ragConfig.enabled as boolean) !== false}
              onToggle={() =>
                handleRagConfigChange("enabled", !((ragConfig.enabled as boolean) !== false))
              }
              label={t("settingsPanel.ragEnabled")}
              alignment="center"
              checkboxClassName="w-4 h-4"
            />
          </div>
          <div>
            <Label className="mb-2 block">
              {t("settingsPanel.topKResults", {
                default: systemConfig?.rag_defaults.top_k || 5,
                value: (ragConfig.top_k as number) || systemConfig?.rag_defaults.top_k || 5,
              })}
            </Label>
            <Slider
              min="1"
              max="20"
              value={(ragConfig.top_k as number) || systemConfig?.rag_defaults.top_k || 5}
              onChange={(e) => handleRagConfigChange("top_k", parseInt(e.target.value))}
            />
          </div>
        </div>
        </div>
      </Card>

      {/* Chat Model Selection */}
      <RoleModelSelectionSection
        title={t("settingsPanel.modelSelection")}
        loading={configLoading}
        loadingLabel={t("settingsPanel.loadingModels")}
        emptyLabel={t("settingsPanel.noRoleModelsAvailable")}
        roleConfigs={chatModelOptions}
        preferredModels={preferredModels}
        onModelChange={(roleName, value, roleDefaultModel) =>
          setPreferredModels((prev) =>
            updatePreferredModelSelection(prev, roleName, value, roleDefaultModel)
          )
        }
        t={t}
      />

      {/* Save */}
      <PrimarySaveBar
        onSave={handleSave}
        saving={saving}
        label={t("settingsPanel.saveSettings")}
        loadingLabel={t("common.saving")}
      />

      {/* Last updated */}
      {settings?.updated_at && (
        <div className="rounded-xl border border-border bg-surface-muted p-4 text-sm text-muted-foreground">
          <p>{t("settingsPage.lastUpdated", { value: formatDateTime(settings.updated_at) })}</p>
        </div>
      )}

      {/* My Data — danger zone */}
      <Card className="!ring-destructive/30">
        <CardHeader>
          <CardTitle className="text-destructive">{t("settingsPanel.myDataSection")}</CardTitle>
          <CardDescription>{t("settingsPanel.myDataDescription")}</CardDescription>
        </CardHeader>
        <div className="px-6 pb-6">
        <DangerOptionsList
          options={(
            [
              ["conversations", "myDataResetConversationsLabel", "myDataResetConversationsDescription"],
              ["workspace",    "myDataResetWorkspaceLabel",     "myDataResetWorkspaceDescription"],
              ["memories",     "myDataResetMemoriesLabel",      "myDataResetMemoriesDescription"],
            ] as const
          ).map(([key, labelKey, descKey]) => ({
            key,
            checked: myDataOptions[key],
            onToggle: () => setMyDataOptions((prev) => ({ ...prev, [key]: !prev[key] })),
            label: t(`settingsPanel.${labelKey}`),
            description: t(`settingsPanel.${descKey}`),
          }))}
        />
        <DangerSelectionActions
          hasSelection={Object.values(myDataOptions).some(Boolean)}
          hintText={t("settingsPanel.myDataResetNoneSelected")}
          onAction={() => setMyDataConfirmOpen(true)}
          loading={myDataResetting}
          loadingLabel={t("settingsPanel.myDataResetting")}
          actionLabel={t("settingsPanel.myDataResetButton")}
          disabled={myDataResetting}
        />
        </div>
      </Card>

      <DangerConfirmDialog
        isOpen={myDataConfirmOpen}
        title={t("settingsPanel.myDataSection")}
        message={t("settingsPanel.myDataResetConfirmMessage")}
        confirmLabel={t("settingsPanel.myDataResetButton")}
        checkboxLabel={t("confirmDialog.resetCheckboxLabel")}
        onConfirm={handleMyDataReset}
        onCancel={() => setMyDataConfirmOpen(false)}
      />
    </div>
  );
}
