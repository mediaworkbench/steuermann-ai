"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  LLMCapabilityItem,
  fetchLLMCapabilities,
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
  const [capabilities, setCapabilities] = useState<LLMCapabilityItem[]>([]);
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(true);
  const [capabilitiesError, setCapabilitiesError] = useState<string | null>(null);
  const [probeTtlSeconds, setProbeTtlSeconds] = useState<number | null>(null);
  const [copyingDiagnostics, setCopyingDiagnostics] = useState(false);
  const [expandedCapabilityRows, setExpandedCapabilityRows] = useState<Record<string, boolean>>({});

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

  const loadCapabilities = useCallback(async () => {
    setCapabilitiesLoading(true);
    setCapabilitiesError(null);
    const data = await fetchLLMCapabilities();
    if (!data) {
      setCapabilities([]);
      setCapabilitiesError(t("settingsPanel.capabilitiesLoadFailed"));
      setCapabilitiesLoading(false);
      return;
    }
    setCapabilities(data.items || []);
    setProbeTtlSeconds(data.probe_ttl_seconds);
    setCapabilitiesLoading(false);
  }, [t]);

  useEffect(() => {
    void loadCapabilities();
  }, [loadCapabilities]);

  const getEffectiveModeBadgeClass = useCallback((mode: string) => {
    switch (mode) {
      case "native":
        return "bg-green-100 text-green-800";
      case "structured":
        return "bg-amber-100 text-amber-800";
      case "react":
        return "bg-blue-100 text-blue-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  }, []);

  const handleCopyDiagnostics = useCallback(async () => {
    if (capabilities.length === 0 || typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      toast.error(t("settingsPanel.copyDiagnosticsFailed"));
      return;
    }

    setCopyingDiagnostics(true);
    try {
      const header = [
        "provider_id",
        "model_name",
        "desired_mode",
        "effective_mode",
        "configured_tool_calling_mode",
        "probe_status",
        "effective_mode_reason",
        "api_base",
        "error_message",
        "probed_at",
      ].join("\t");
      const rows = capabilities.map((item) =>
        [
          item.provider_id,
          item.model_name,
          item.desired_mode,
          item.effective_mode,
          item.configured_tool_calling_mode || "",
          item.probe_status,
          item.effective_mode_reason,
          item.api_base || "",
          item.error_message || "",
          item.probed_at || "",
        ].join("\t")
      );
      const ttlLine = `probe_ttl_seconds\t${probeTtlSeconds ?? ""}`;
      await navigator.clipboard.writeText([ttlLine, header, ...rows].join("\n"));
      toast.success(t("settingsPanel.copyDiagnosticsSuccess"));
    } catch {
      toast.error(t("settingsPanel.copyDiagnosticsFailed"));
    } finally {
      setCopyingDiagnostics(false);
    }
  }, [capabilities, probeTtlSeconds, t]);

  const toggleCapabilityRow = useCallback((key: string) => {
    setExpandedCapabilityRows((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
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

      {/* Model Capability Status */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">{t("settingsPanel.capabilitiesTitle")}</h3>
            <p className="text-sm text-gray-600">
              {t("settingsPanel.capabilitiesSubtitle")}
              {probeTtlSeconds !== null && (
                <span>{t("settingsPanel.capabilitiesTtl", { value: probeTtlSeconds })}</span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopyDiagnostics}
              disabled={capabilitiesLoading || copyingDiagnostics || capabilities.length === 0}
              className="px-3 py-2 text-sm bg-indigo-100 hover:bg-indigo-200 disabled:bg-gray-100 text-indigo-800 disabled:text-gray-500 rounded-lg transition-colors"
            >
              {copyingDiagnostics ? t("common.loading") : t("settingsPanel.copyDiagnostics")}
            </button>
            <button
              type="button"
              onClick={() => void loadCapabilities()}
              disabled={capabilitiesLoading}
              className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 disabled:bg-gray-100 text-gray-700 rounded-lg transition-colors"
            >
              {t("common.refresh")}
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-xs font-semibold text-gray-600">{t("settingsPanel.legendTitle")}</span>
          <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-800">{t("settingsPanel.legendNative")}</span>
          <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-amber-100 text-amber-800">{t("settingsPanel.legendStructured")}</span>
          <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-blue-100 text-blue-800">{t("settingsPanel.legendReact")}</span>
        </div>

        {capabilitiesLoading ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.capabilitiesLoading")}</p>
        ) : capabilitiesError ? (
          <p className="text-sm text-red-600">{capabilitiesError}</p>
        ) : capabilities.length === 0 ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.capabilitiesEmpty")}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full border border-gray-200 rounded-lg text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityModel")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityDesired")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityEffective")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityProbeStatus")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityReason")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityProbedAt")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityDetails")}</th>
                </tr>
              </thead>
              <tbody>
                {capabilities.map((item) => {
                  const rowKey = `${item.provider_id}:${item.model_name}`;
                  const expanded = !!expandedCapabilityRows[rowKey];
                  return (
                    <Fragment key={rowKey}>
                      <tr className="border-t border-gray-200">
                        <td className="px-3 py-2 text-gray-800">
                          <div className="font-medium">{item.model_name}</div>
                          <div className="text-xs text-gray-500">{item.provider_id}</div>
                        </td>
                        <td className="px-3 py-2 text-gray-700">{item.desired_mode}</td>
                        <td className="px-3 py-2">
                          <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${getEffectiveModeBadgeClass(item.effective_mode)}`}>
                            {item.effective_mode}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-gray-700">{item.probe_status}</td>
                        <td className="px-3 py-2 text-gray-700">{item.effective_mode_reason}</td>
                        <td className="px-3 py-2 text-gray-700">{item.probed_at ? formatDateTime(item.probed_at) : t("metrics.na")}</td>
                        <td className="px-3 py-2 text-gray-700">
                          <button
                            type="button"
                            onClick={() => toggleCapabilityRow(rowKey)}
                            className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded"
                          >
                            {expanded ? t("settingsPanel.hideDetails") : t("settingsPanel.showDetails")}
                          </button>
                        </td>
                      </tr>
                      {expanded && (
                        <tr className="border-t border-gray-100 bg-gray-50">
                          <td colSpan={8} className="px-3 py-3 text-xs text-gray-700">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              <div>
                                <span className="font-semibold">{t("settingsPanel.detailConfiguredMode")}: </span>
                                <span>{item.configured_tool_calling_mode || t("metrics.na")}</span>
                              </div>
                              <div>
                                <span className="font-semibold">{t("settingsPanel.detailApiBase")}: </span>
                                <span>{item.api_base || t("metrics.na")}</span>
                              </div>
                              <div>
                                <span className="font-semibold">{t("settingsPanel.detailError")}: </span>
                                <span>{item.error_message || t("metrics.na")}</span>
                              </div>
                              <div>
                                <span className="font-semibold">{t("settingsPanel.detailBindTools")}: </span>
                                <span>{item.supports_bind_tools === null ? t("metrics.na") : String(item.supports_bind_tools)}</span>
                              </div>
                              <div>
                                <span className="font-semibold">{t("settingsPanel.detailToolSchema")}: </span>
                                <span>{item.supports_tool_schema === null ? t("metrics.na") : String(item.supports_tool_schema)}</span>
                              </div>
                              <div>
                                <span className="font-semibold">{t("settingsPanel.detailMismatch")}: </span>
                                <span>{String(item.capability_mismatch)}</span>
                              </div>
                            </div>
                            <div className="mt-2">
                              <div className="font-semibold mb-1">{t("settingsPanel.detailMetadata")}</div>
                              <pre className="overflow-x-auto p-2 bg-white border border-gray-200 rounded">
                                {JSON.stringify(item.metadata || {}, null, 2)}
                              </pre>
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
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

