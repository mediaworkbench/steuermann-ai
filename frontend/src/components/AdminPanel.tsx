"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { ConfirmDialog } from "./ConfirmDialog";
import {
  LLMCapabilityItem,
  fetchLLMCapabilities,
  UserSettings,
  fetchSystemConfig,
  triggerReingestAllDocuments,
  resetAllDatabases,
  type SystemConfig,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";

export interface AdminPanelProps {
  settings: UserSettings | null;
  loading: boolean;
  onSave: (updates: Partial<Omit<UserSettings, "user_id" | "updated_at">>) => Promise<boolean>;
}

const ADMIN_MODEL_ROLES = ["vision", "auxiliary"];

export function AdminPanel({ settings, loading, onSave }: AdminPanelProps) {
  const { t, formatDateTime } = useI18n();

  // Full rag_config read from server — we only expose collection + threshold here
  const [ragConfig, setRagConfig] = useState<Record<string, unknown>>(
    settings?.rag_config || { collection: "", top_k: 5, enabled: true }
  );
  // Full preferred_models — we only expose vision + auxiliary here
  const [preferredModels, setPreferredModels] = useState<Record<string, string | null>>(
    settings?.preferred_models || {}
  );

  const [saving, setSaving] = useState(false);
  const [reingesting, setReingesting] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [confirmReingest, setConfirmReingest] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);

  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [capabilities, setCapabilities] = useState<LLMCapabilityItem[]>([]);
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(true);
  const [capabilitiesError, setCapabilitiesError] = useState<string | null>(null);
  const [probeTtlSeconds, setProbeTtlSeconds] = useState<number | null>(null);
  const [copyingDiagnostics, setCopyingDiagnostics] = useState(false);
  const [expandedCapabilityRows, setExpandedCapabilityRows] = useState<Record<string, boolean>>({});

  // Read-modify-write: sync all fields from server on load so saves don't wipe user-owned keys
  useEffect(() => {
    if (settings) {
      const ragDefault = {
        collection: systemConfig?.rag_defaults.collection_name || "framework",
        top_k: systemConfig?.rag_defaults.top_k || 5,
      };
      setRagConfig(
        settings.rag_config && settings.rag_config.collection ? settings.rag_config : ragDefault
      );
      setPreferredModels(settings.preferred_models || {});
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
      case "native": return "bg-green-100 text-green-800";
      case "structured": return "bg-amber-100 text-amber-800";
      case "react": return "bg-blue-100 text-blue-800";
      default: return "bg-gray-100 text-gray-800";
    }
  }, []);

  const getRoleBadgeClass = useCallback((role?: string) => {
    switch (role) {
      case "chat": return "bg-indigo-100 text-indigo-800";
      case "vision": return "bg-purple-100 text-purple-800";
      case "auxiliary": return "bg-orange-100 text-orange-800";
      default: return "bg-gray-100 text-gray-800";
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
        "provider_id", "model_name", "role", "desired_mode", "effective_mode",
        "configured_tool_calling_mode", "probe_status", "effective_mode_reason",
        "api_base", "error_message", "probed_at",
      ].join("\t");
      const rows = capabilities.map((item) =>
        [
          item.provider_id, item.model_name, item.role || "", item.desired_mode,
          item.effective_mode, item.configured_tool_calling_mode || "", item.probe_status,
          item.effective_mode_reason, item.api_base || "", item.error_message || "",
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
    setExpandedCapabilityRows((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleRagConfigChange = useCallback((key: string, value: unknown) => {
    setRagConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Save: send the full settings object (read-modify-write — preserves user-owned fields)
  const handleSave = useCallback(async () => {
    if (!settings) return;
    setSaving(true);
    try {
      const success = await onSave({
        tool_toggles: settings.tool_toggles,
        rag_config: ragConfig,
        language: settings.language,
        preferred_model: preferredModels.chat || null,
        preferred_models: preferredModels,
        analytics_preferences: settings.analytics_preferences,
      });
      if (success) {
        toast.success(t("settingsPanel.settingsSaved"));
      } else {
        toast.error(t("settingsPanel.failedToSaveSettings"));
      }
    } finally {
      setSaving(false);
    }
  }, [settings, ragConfig, preferredModels, onSave, t]);

  const handleReingestAll = useCallback(async () => {
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

  const handleResetAllDatabases = useCallback(async () => {
    setResetting(true);
    try {
      const result = await resetAllDatabases();
      if (result.errors?.length) {
        toast.warning(`${t("settingsPanel.resetSuccess")} (with warnings)`, {
          description: result.errors.join("; "),
        });
      } else {
        toast.success(t("settingsPanel.resetSuccess"));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : t("settingsPanel.resetFailed");
      toast.error(`${t("settingsPanel.resetFailed")}: ${message}`);
    } finally {
      setResetting(false);
    }
  }, [t]);

  if (loading) {
    return <div className="text-center py-8 text-gray-500">{t("common.loading")}</div>;
  }

  const roleModelOptions = (systemConfig?.model_roles || []).filter((r) =>
    ADMIN_MODEL_ROLES.includes(r.role)
  );

  return (
    <div className="space-y-6">

      {/* LLM Capability Diagnostics */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">{t("adminPage.llmSection")}</h3>
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
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">{t("settingsPanel.capabilityRole")}</th>
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
                        <td className="px-3 py-2">
                          {item.role && (
                            <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${getRoleBadgeClass(item.role)}`}>
                              {item.role}
                            </span>
                          )}
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
                        <td className="px-3 py-2">
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
                              <div><span className="font-semibold">{t("settingsPanel.detailConfiguredMode")}: </span><span>{item.configured_tool_calling_mode || t("metrics.na")}</span></div>
                              <div><span className="font-semibold">{t("settingsPanel.detailApiBase")}: </span><span>{item.api_base || t("metrics.na")}</span></div>
                              <div><span className="font-semibold">{t("settingsPanel.detailError")}: </span><span>{item.error_message || t("metrics.na")}</span></div>
                              <div><span className="font-semibold">{t("settingsPanel.detailBindTools")}: </span><span>{item.supports_bind_tools === null ? t("metrics.na") : String(item.supports_bind_tools)}</span></div>
                              <div><span className="font-semibold">{t("settingsPanel.detailVision")}: </span><span>{item.supports_vision === null || item.supports_vision === undefined ? t("metrics.na") : String(item.supports_vision)}</span></div>
                              <div><span className="font-semibold">{t("settingsPanel.detailReasoning")}: </span><span>{String(item.supports_reasoning ?? false)}</span></div>
                              <div><span className="font-semibold">{t("settingsPanel.detailMismatch")}: </span><span>{String(item.capability_mismatch)}</span></div>
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

      {/* RAG Operational Configuration */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("adminPage.ragSection")}</h3>
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
              {t("settingsPanel.similarityThreshold")}
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={(ragConfig.pill_score_threshold as number) || 0.72}
              onChange={(e) => handleRagConfigChange("pill_score_threshold", parseFloat(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="mt-6 border-t border-gray-200 pt-4">
          <h4 className="text-md font-semibold text-gray-800 mb-2">{t("settingsPanel.reingestSectionTitle")}</h4>
          <p className="text-sm text-gray-600 mb-3">{t("settingsPanel.reingestDescription")}</p>
          <button
            type="button"
            onClick={() => setConfirmReingest(true)}
            disabled={reingesting}
            className="bg-amber-600 hover:bg-amber-700 disabled:bg-gray-400 text-white font-semibold px-4 py-2 rounded-lg transition-colors"
          >
            {reingesting ? t("settingsPanel.reingesting") : t("settingsPanel.reingestAllDocuments")}
          </button>
        </div>
      </div>

      {/* System Model Selection (vision + auxiliary) */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t("adminPage.modelSection")}</h3>
        {configLoading ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.loadingModels")}</p>
        ) : roleModelOptions.length === 0 ? (
          <p className="text-sm text-gray-500">{t("settingsPanel.noRoleModelsAvailable")}</p>
        ) : (
          <div className="space-y-4">
            {roleModelOptions.map((roleConfig) => {
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
          disabled={saving || !settings}
          className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold px-6 py-3 rounded-lg transition-colors"
        >
          {saving ? t("common.saving") : t("settingsPanel.saveSettings")}
        </button>
      </div>

      {/* Danger Zone */}
      <div className="bg-white rounded-lg shadow p-6 border border-red-200">
        <h3 className="text-lg font-semibold text-red-700 mb-4">{t("adminPage.dangerZoneSection")}</h3>
        <div>
          <h4 className="text-md font-semibold text-red-700 mb-2">{t("settingsPanel.resetSectionTitle")}</h4>
          <p className="text-sm text-gray-600 mb-3">{t("settingsPanel.resetDescription")}</p>
          <button
            type="button"
            onClick={() => setConfirmReset(true)}
            disabled={resetting}
            className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-semibold px-4 py-2 rounded-lg transition-colors"
          >
            {resetting ? t("settingsPanel.resetting") : t("settingsPanel.resetAllDatabases")}
          </button>
        </div>
      </div>

      <ConfirmDialog
        isOpen={confirmReingest}
        title={t("settingsPanel.reingestSectionTitle")}
        message={t("settingsPanel.confirmReingestAll")}
        variant="default"
        onConfirm={() => {
          setConfirmReingest(false);
          handleReingestAll();
        }}
        onCancel={() => setConfirmReingest(false)}
      />

      <ConfirmDialog
        isOpen={confirmReset}
        title={t("settingsPanel.resetSectionTitle")}
        message={t("settingsPanel.resetDescription")}
        variant="danger"
        requireChecked={true}
        checkboxLabel={t("confirmDialog.resetCheckboxLabel")}
        confirmLabel={t("settingsPanel.resetAllDatabases")}
        onConfirm={() => {
          setConfirmReset(false);
          handleResetAllDatabases();
        }}
        onCancel={() => setConfirmReset(false)}
      />
    </div>
  );
}
