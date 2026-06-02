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
    return <div className="text-center py-8 text-gray-500">{t("common.loading")}</div>;
  }

  const chatModelOptions = (systemConfig?.model_roles || []).filter((r) =>
    USER_MODEL_ROLES.includes(r.role)
  );

  return (
    <div className="space-y-6">

      {/* Language */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.language")}</h3>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {(systemConfig?.supported_languages || ["en"]).map((code) => (
            <option key={code} value={code}>
              {LANGUAGE_LABELS[code] || code.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      {/* Sound */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.soundSection")}</h3>
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={(analyticsPreferences.sound_enabled as boolean) ?? true}
            onChange={(e) =>
              setAnalyticsPreferences((prev) => ({ ...prev, sound_enabled: e.target.checked }))
            }
            className="mt-0.5 w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="flex flex-col">
            <span className="text-sm font-medium text-gray-700">{t("settingsPanel.soundEnabled")}</span>
            <span className="text-xs text-gray-500 mt-0.5">{t("settingsPanel.soundEnabledDescription")}</span>
          </span>
        </label>
      </div>

      {/* Tool Toggles */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.toolSettings")}</h3>
        {configLoading ? (
          <div className="text-gray-500 text-sm">{t("settingsPanel.loadingTools")}</div>
        ) : (
          <div className="space-y-3">
            {(systemConfig?.available_tools || FALLBACK_TOOLS).map((tool) => (
              <label key={tool.id} className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={toolToggles[tool.id] ?? true}
                  onChange={() => handleToolToggle(tool.id)}
                  className="w-5 h-5 text-blue-600 rounded"
                />
                <span className="text-gray-700 font-medium">{tool.label}</span>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* RAG — user-scoped controls: enabled toggle + top_k */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.ragConfiguration")}</h3>
        {configLoading && <div className="text-gray-500 text-sm">{t("settingsPanel.loadingDefaults")}</div>}
        <div className="space-y-4">
          <div>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={(ragConfig.enabled as boolean) !== false}
                onChange={(e) => handleRagConfigChange("enabled", e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-700">{t("settingsPanel.ragEnabled")}</span>
            </label>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t("settingsPanel.topKResults", {
                default: systemConfig?.rag_defaults.top_k || 5,
                value: (ragConfig.top_k as number) || systemConfig?.rag_defaults.top_k || 5,
              })}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value={(ragConfig.top_k as number) || systemConfig?.rag_defaults.top_k || 5}
              onChange={(e) => handleRagConfigChange("top_k", parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Chat Model Selection */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.modelSelection")}</h3>
        {configLoading ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.loadingModels")}</p>
        ) : chatModelOptions.length === 0 ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.noRoleModelsAvailable")}</p>
        ) : (
          <div className="space-y-4">
            {chatModelOptions.map((roleConfig) => {
              const roleName = roleConfig.role;
              const roleDefaultModel = roleConfig.default_model || "";
              const selectedModel = preferredModels[roleName] || roleDefaultModel;
              const roleModels = roleConfig.available_models || [];
              const mergedRoleModels = roleModels.includes(roleDefaultModel)
                ? roleModels
                : [roleDefaultModel, ...roleModels].filter(Boolean);
              return (
                <div key={roleName} className="border border-gray-200 rounded-lg p-4">
                  <div className="mb-2">
                    <p className="text-sm font-semibold text-gray-800">
                      {t("settingsPanel.roleModelLabel", { role: roleName })}
                    </p>
                    <p className="text-xs text-gray-600">
                      {t("settingsPanel.roleProviderLocked", { provider: roleConfig.provider_id })}
                    </p>
                    <p className="text-xs text-gray-600">
                      {t("settingsPanel.systemDefault", { value: roleDefaultModel })}
                    </p>
                    {roleConfig.model_load_error && (
                      <p className="text-xs text-amber-700">{roleConfig.model_load_error}</p>
                    )}
                  </div>
                  <select
                    value={selectedModel}
                    onChange={(e) =>
                      setPreferredModels((prev) => ({
                        ...prev,
                        [roleName]: e.target.value === roleDefaultModel ? "" : e.target.value,
                      }))
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {mergedRoleModels.map((model) => (
                      <option key={`${roleName}:${model}`} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Save */}
      <div className="flex gap-3">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold px-6 py-3 rounded-lg transition-colors"
        >
          {saving ? t("common.saving") : t("settingsPanel.saveSettings")}
        </button>
      </div>

      {/* Last updated */}
      {settings?.updated_at && (
        <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600">
          <p>{t("settingsPage.lastUpdated", { value: formatDateTime(settings.updated_at) })}</p>
        </div>
      )}

      {/* My Data — danger zone */}
      <div className="bg-white rounded-lg shadow border border-red-200 p-6">
        <h3 className="text-lg font-semibold text-red-700 mb-1">{t("settingsPanel.myDataSection")}</h3>
        <p className="text-sm text-gray-500 mb-4">{t("settingsPanel.myDataDescription")}</p>
        <div className="space-y-3 mb-5">
          {(
            [
              ["conversations", "myDataResetConversationsLabel", "myDataResetConversationsDescription"],
              ["workspace",    "myDataResetWorkspaceLabel",     "myDataResetWorkspaceDescription"],
              ["memories",     "myDataResetMemoriesLabel",      "myDataResetMemoriesDescription"],
            ] as const
          ).map(([key, labelKey, descKey]) => (
            <label key={key} className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={myDataOptions[key]}
                onChange={() =>
                  setMyDataOptions((prev) => ({ ...prev, [key]: !prev[key] }))
                }
                className="mt-0.5 w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
              />
              <span className="flex flex-col">
                <span className="text-sm font-medium text-gray-700">{t(`settingsPanel.${labelKey}`)}</span>
                <span className="text-xs text-gray-500 mt-0.5">{t(`settingsPanel.${descKey}`)}</span>
              </span>
            </label>
          ))}
        </div>
        {!Object.values(myDataOptions).some(Boolean) && (
          <p className="text-xs text-amber-600 mb-3">{t("settingsPanel.myDataResetNoneSelected")}</p>
        )}
        <button
          onClick={() => {
            if (!Object.values(myDataOptions).some(Boolean)) return;
            setMyDataConfirmOpen(true);
          }}
          disabled={myDataResetting || !Object.values(myDataOptions).some(Boolean)}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-300 text-white font-semibold px-4 py-2 rounded-lg text-sm transition-colors"
        >
          {myDataResetting ? t("settingsPanel.myDataResetting") : t("settingsPanel.myDataResetButton")}
        </button>
      </div>

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
