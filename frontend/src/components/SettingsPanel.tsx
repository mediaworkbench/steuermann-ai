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
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DangerActionButton } from "@/components/product/DangerActionButton";
import { DangerHintText } from "@/components/product/DangerHintText";
import { FormFieldLabel } from "@/components/product/FormFieldLabel";
import { OptionCheckboxRow } from "@/components/product/OptionCheckboxRow";
import { PanelLoadingState } from "@/components/product/PanelLoadingState";
import { PrimarySaveBar } from "@/components/product/PrimarySaveBar";
import { RoleModelSelectionSection } from "@/components/product/RoleModelSelectionSection";
import { SectionStateText } from "@/components/product/SectionStateText";
import { TitledSectionCard } from "@/components/product/TitledSectionCard";
import { Checkbox } from "@/components/ui/Checkbox";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";

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
    return <PanelLoadingState label={t("common.loading")} />;
  }

  const chatModelOptions = (systemConfig?.model_roles || []).filter((r) =>
    USER_MODEL_ROLES.includes(r.role)
  );

  return (
    <div className="space-y-6">

      {/* Language */}
      <TitledSectionCard title={t("settingsPanel.language")}>
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
      </TitledSectionCard>

      {/* Sound */}
      <TitledSectionCard title={t("settingsPanel.soundSection")}>
        <label className="flex items-start gap-3 cursor-pointer">
          <Checkbox
            type="checkbox"
            checked={(analyticsPreferences.sound_enabled as boolean) ?? true}
            onChange={(e) =>
              setAnalyticsPreferences((prev) => ({ ...prev, sound_enabled: e.target.checked }))
            }
            className="mt-0.5"
          />
          <span className="flex flex-col">
            <span className="text-sm font-medium text-foreground">{t("settingsPanel.soundEnabled")}</span>
            <span className="mt-0.5 text-xs text-foreground/60">{t("settingsPanel.soundEnabledDescription")}</span>
          </span>
        </label>
      </TitledSectionCard>

      {/* Tool Toggles */}
      <TitledSectionCard title={t("settingsPanel.toolSettings")}>
        {configLoading ? (
          <SectionStateText>{t("settingsPanel.loadingTools")}</SectionStateText>
        ) : (
          <div className="space-y-3">
            {(systemConfig?.available_tools || FALLBACK_TOOLS).map((tool) => (
              <label key={tool.id} className="flex items-center gap-3 cursor-pointer">
                <Checkbox
                  type="checkbox"
                  checked={toolToggles[tool.id] ?? true}
                  onChange={() => handleToolToggle(tool.id)}
                  className="w-5 h-5"
                />
                <span className="font-medium text-foreground">{tool.label}</span>
              </label>
            ))}
          </div>
        )}
      </TitledSectionCard>

      {/* RAG — user-scoped controls: enabled toggle + top_k */}
      <TitledSectionCard title={t("settingsPanel.ragConfiguration")}>
        {configLoading && <SectionStateText>{t("settingsPanel.loadingDefaults")}</SectionStateText>}
        <div className="space-y-4">
          <div>
            <label className="flex items-center gap-3 cursor-pointer">
              <Checkbox
                type="checkbox"
                checked={(ragConfig.enabled as boolean) !== false}
                onChange={(e) => handleRagConfigChange("enabled", e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm font-medium text-foreground">{t("settingsPanel.ragEnabled")}</span>
            </label>
          </div>
          <div>
            <FormFieldLabel>
              {t("settingsPanel.topKResults", {
                default: systemConfig?.rag_defaults.top_k || 5,
                value: (ragConfig.top_k as number) || systemConfig?.rag_defaults.top_k || 5,
              })}
            </FormFieldLabel>
            <Slider
              min="1"
              max="20"
              value={(ragConfig.top_k as number) || systemConfig?.rag_defaults.top_k || 5}
              onChange={(e) => handleRagConfigChange("top_k", parseInt(e.target.value))}
            />
          </div>
        </div>
      </TitledSectionCard>

      {/* Chat Model Selection */}
      <RoleModelSelectionSection
        title={t("settingsPanel.modelSelection")}
        loading={configLoading}
        loadingLabel={t("settingsPanel.loadingModels")}
        emptyLabel={t("settingsPanel.noRoleModelsAvailable")}
        roleConfigs={chatModelOptions}
        preferredModels={preferredModels}
        onModelChange={(roleName, value, roleDefaultModel) =>
          setPreferredModels((prev) => ({
            ...prev,
            [roleName]: value === roleDefaultModel ? "" : value,
          }))
        }
        getRoleLabel={(roleName) => t("settingsPanel.roleModelLabel", { role: roleName })}
        getProviderLabel={(providerId) =>
          t("settingsPanel.roleProviderLocked", { provider: providerId })
        }
        getSystemDefaultLabel={(defaultModel) =>
          t("settingsPanel.systemDefault", { value: defaultModel })
        }
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
      <TitledSectionCard
        title={t("settingsPanel.myDataSection")}
        description={t("settingsPanel.myDataDescription")}
        tone="danger"
      >
        <div className="space-y-3 mb-5">
          {(
            [
              ["conversations", "myDataResetConversationsLabel", "myDataResetConversationsDescription"],
              ["workspace",    "myDataResetWorkspaceLabel",     "myDataResetWorkspaceDescription"],
              ["memories",     "myDataResetMemoriesLabel",      "myDataResetMemoriesDescription"],
            ] as const
          ).map(([key, labelKey, descKey]) => (
            <OptionCheckboxRow
              key={key}
              checked={myDataOptions[key]}
              onToggle={() =>
                setMyDataOptions((prev) => ({ ...prev, [key]: !prev[key] }))
              }
              label={t(`settingsPanel.${labelKey}`)}
              description={t(`settingsPanel.${descKey}`)}
            />
          ))}
        </div>
        {!Object.values(myDataOptions).some(Boolean) && (
          <DangerHintText>{t("settingsPanel.myDataResetNoneSelected")}</DangerHintText>
        )}
        <DangerActionButton
          onClick={() => {
            if (!Object.values(myDataOptions).some(Boolean)) return;
            setMyDataConfirmOpen(true);
          }}
          disabled={myDataResetting || !Object.values(myDataOptions).some(Boolean)}
          loading={myDataResetting}
          loadingLabel={t("settingsPanel.myDataResetting")}
          label={t("settingsPanel.myDataResetButton")}
        />
      </TitledSectionCard>

      <ConfirmDialog
        isOpen={myDataConfirmOpen}
        title={t("settingsPanel.myDataSection")}
        message={t("settingsPanel.myDataResetConfirmMessage")}
        confirmLabel={t("settingsPanel.myDataResetButton")}
        requireChecked
        checkboxLabel={t("confirmDialog.resetCheckboxLabel")}
        onConfirm={handleMyDataReset}
        onCancel={() => setMyDataConfirmOpen(false)}
        variant="danger"
      />
    </div>
  );
}
