"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  UserSettings,
  fetchAvailableModels,
  fetchSystemConfig,
  triggerReingestAllDocuments,
  type SystemConfig,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

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
  { id: "datetime_tool", label: "Datetime" },
  { id: "calculator_tool", label: "Calculator" },
  { id: "file_ops_tool", label: "File Ops" },
];

export function SettingsPanel({ settings, loading, onSave }: SettingsPanelProps) {
  const { t, formatDateTime } = useI18n();
  const [toolToggles, setToolToggles] = useState<Record<string, boolean>>(
    settings?.tool_toggles || {}
  );
  const [ragConfig, setRagConfig] = useState<Record<string, unknown>>(
    settings?.rag_config || { collection: "", top_k: 5 }
  );
  const [preferredModel, setPreferredModel] = useState(settings?.preferred_model || "");
  const [language, setLanguage] = useState(settings?.language || "en");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [reingesting, setReingesting] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);

  // Sync local state when settings arrive from the server
  useEffect(() => {
    if (settings) {
      setToolToggles(settings.tool_toggles || {});
      // Use server settings, but prefill missing fields with system defaults
      const ragDefault = { collection: systemConfig?.rag_defaults.collection_name || "framework", top_k: systemConfig?.rag_defaults.top_k || 5 };
      setRagConfig(settings.rag_config && settings.rag_config.collection ? settings.rag_config : ragDefault);
      setPreferredModel(settings.preferred_model || "");
      setLanguage(settings.language || "en");
    }
  }, [settings, systemConfig]);

  // Fetch system config (tools, rag defaults, default model)
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

  // Fetch available models from Ollama via backend
  useEffect(() => {
    let cancelled = false;
    async function loadModels() {
      setModelsLoading(true);
      const models = await fetchAvailableModels();
      if (!cancelled) {
        setAvailableModels(models);
        setModelsLoading(false);
      }
    }
    loadModels();
    return () => { cancelled = true; };
  }, []);

  const handleToolToggle = useCallback((toolId: string) => {
    setToolToggles((prev) => ({
      ...prev,
      [toolId]: !prev[toolId],
    }));
  }, []);

  const handleRagConfigChange = useCallback((key: string, value: unknown) => {
    setRagConfig((prev) => ({
      ...prev,
      [key]: value,
    }));
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveMessage(null);
    try {
      const success = await onSave({
        tool_toggles: toolToggles,
        rag_config: ragConfig,
        preferred_model: preferredModel || null,
        language,
      });
      if (success) {
        setSaveMessage(`✓ ${t("settingsPanel.settingsSaved")}`);
        toast.success(t("settingsPanel.settingsSaved"));
        setTimeout(() => setSaveMessage(null), 3000);
      } else {
        setSaveMessage(`✗ ${t("settingsPanel.failedToSaveSettings")}`);
        toast.error(t("settingsPanel.failedToSaveSettings"));
      }
    } finally {
      setSaving(false);
    }
  }, [toolToggles, ragConfig, preferredModel, language, onSave, t]);

  const handleReingestAll = useCallback(async () => {
    const confirmed = window.confirm(t("settingsPanel.confirmReingestAll"));
    if (!confirmed) {
      return;
    }

    setReingesting(true);
    try {
      const result = await triggerReingestAllDocuments();
      toast.success(
        t("settingsPanel.reingestSuccess", {
          processed: result.processed,
          chunks: result.total_chunks,
        })
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : t("settingsPanel.reingestFailed");
      toast.error(`${t("settingsPanel.reingestFailed")}: ${message}`);
    } finally {
      setReingesting(false);
    }
  }, [t]);

  if (loading) {
    return <div className="text-center py-8 text-gray-500">{t("common.loading")}</div>;
  }

  const systemDefaultModel = systemConfig?.default_model || "gemma3:4b";
  const selectedModel = preferredModel || systemDefaultModel;

  return (
    <div className="space-y-6">
      {/* Language Section */}
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

        {/* Tool Toggles Section */}
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

      {/* RAG Configuration Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.ragConfiguration")}</h3>
        {configLoading && <div className="text-gray-500 text-sm">{t("settingsPanel.loadingDefaults")}</div>}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t("settingsPanel.knowledgeCollection", { value: systemConfig?.rag_defaults.collection_name || "framework" })}
            </label>
            <input
              type="text"
              value={(ragConfig.collection as string) || systemConfig?.rag_defaults.collection_name || "framework"}
              onChange={(e) => handleRagConfigChange("collection", e.target.value)}
              placeholder={systemConfig?.rag_defaults.collection_name || "e.g., framework"}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
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

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t("settingsPanel.similarityThreshold")}
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={(ragConfig.score_threshold as number) || 0.5}
              onChange={(e) =>
                handleRagConfigChange("score_threshold", parseFloat(e.target.value))
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="mt-6 border-t border-gray-200 pt-4">
          <h4 className="text-md font-semibold text-gray-800 mb-2">{t("settingsPanel.reingestSectionTitle")}</h4>
          <p className="text-sm text-gray-600 mb-3">{t("settingsPanel.reingestDescription")}</p>
          <button
            type="button"
            onClick={handleReingestAll}
            disabled={reingesting}
            className="bg-amber-600 hover:bg-amber-700 disabled:bg-gray-400 text-white font-semibold px-4 py-2 rounded-lg transition-colors"
          >
            {reingesting ? t("settingsPanel.reingesting") : t("settingsPanel.reingestAllDocuments")}
          </button>
        </div>
      </div>

      {/* Model Selection Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("settingsPanel.modelSelection")}</h3>
        <div className="mb-2 text-xs text-gray-600">
          <span className="font-semibold">{t("settingsPanel.systemDefault", { value: systemConfig?.default_model || "gemma3:4b" })}</span>
        </div>
        {modelsLoading ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.loadingModels")}</p>
        ) : availableModels.length === 0 ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.noModelsAvailable")}</p>
        ) : (
          <select
            value={selectedModel}
            onChange={(e) =>
              setPreferredModel(e.target.value === systemDefaultModel ? "" : e.target.value)
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {!availableModels.includes(systemDefaultModel) && (
              <option value={systemDefaultModel}>{systemDefaultModel}</option>
            )}
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Save Section */}
      <div className="flex gap-3">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold px-6 py-3 rounded-lg transition-colors"
        >
          {saving ? t("common.saving") : t("settingsPanel.saveSettings")}
        </button>
      </div>

      {saveMessage && (
        <div
          className={`p-4 rounded-lg text-center font-medium ${
            saveMessage.startsWith("✓")
              ? "bg-green-100 text-green-800"
              : "bg-red-100 text-red-800"
          }`}
        >
          {saveMessage}
        </div>
      )}

      {/* Metadata Section */}
      {settings?.updated_at && (
        <div className="bg-gray-50 rounded-lg p-4 text-sm text-gray-600">
          <p>{t("settingsPage.lastUpdated", { value: formatDateTime(settings.updated_at) })}</p>
        </div>
      )}
    </div>
  );
}

