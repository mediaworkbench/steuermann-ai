"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { ConfirmDialog } from "./ConfirmDialog";
import { SectionCard } from "@/components/product/SectionCard";
import {
  LLMCapabilityItem,
  fetchLLMCapabilities,
  UserSettings,
  fetchSystemConfig,
  triggerReingestAllDocuments,
  resetAllDatabases,
  type ResetOptions,
  type SystemConfig,
} from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import { Button } from "@/components/ui/Button";
import { Checkbox } from "@/components/ui/Checkbox";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";

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
  const [resetOptions, setResetOptions] = useState<ResetOptions>({
    conversations: true,
    workspace: true,
    memories: true,
    analytics: true,
    llm_probes: true,
  });

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
      case "native": return "bg-success/10 text-success";
      case "structured": return "bg-warning/10 text-warning";
      case "react": return "bg-info/10 text-info";
      default: return "bg-muted text-foreground";
    }
  }, []);

  const getRoleBadgeClass = useCallback((role?: string) => {
    switch (role) {
      case "chat": return "bg-indigo-100 text-indigo-800";
      case "vision": return "bg-purple-100 text-purple-800";
      case "auxiliary": return "bg-orange-100 text-orange-800";
      default: return "bg-muted text-foreground";
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
          item.provider_id,
          item.model_name,
          item.role || "",
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

  const toggleResetOption = useCallback((key: keyof ResetOptions) => {
    setResetOptions((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleResetAllDatabases = useCallback(async () => {
    setResetting(true);
    try {
      const result = await resetAllDatabases(resetOptions);
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
  }, [resetOptions, t]);

  if (loading) {
    return <div className="py-8 text-center text-muted-foreground">{t("common.loading")}</div>;
  }

  const roleModelOptions = (systemConfig?.model_roles || []).filter((r) =>
    ADMIN_MODEL_ROLES.includes(r.role)
  );

  return (
    <div className="space-y-6">

      {/* LLM Capability Diagnostics */}
      <SectionCard>
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h3 className="text-lg font-semibold text-foreground">{t("adminPage.llmSection")}</h3>
            <p className="text-sm text-muted-foreground">
              {t("settingsPanel.capabilitiesSubtitle")}
              {probeTtlSeconds !== null && (
                <span>{t("settingsPanel.capabilitiesTtl", { value: probeTtlSeconds })}</span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              onClick={handleCopyDiagnostics}
              disabled={capabilitiesLoading || copyingDiagnostics || capabilities.length === 0}
              variant="secondary"
              size="sm"
            >
              {copyingDiagnostics ? t("common.loading") : t("settingsPanel.copyDiagnostics")}
            </Button>
            <Button
              type="button"
              onClick={() => void loadCapabilities()}
              disabled={capabilitiesLoading}
              variant="secondary"
              size="sm"
            >
              {t("common.refresh")}
            </Button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-xs font-semibold text-muted-foreground">{t("settingsPanel.legendTitle")}</span>
          <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-success/10 text-success">{t("settingsPanel.legendNative")}</span>
          <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-warning/10 text-warning">{t("settingsPanel.legendStructured")}</span>
          <span className="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold bg-info/10 text-info">{t("settingsPanel.legendReact")}</span>
        </div>

        {capabilitiesLoading ? (
          <p className="text-sm text-muted-foreground">{t("settingsPanel.capabilitiesLoading")}</p>
        ) : capabilitiesError ? (
          <p className="text-sm text-destructive">{capabilitiesError}</p>
        ) : capabilities.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("settingsPanel.capabilitiesEmpty")}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full rounded-lg border border-border text-sm">
              <thead className="bg-surface-muted">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityModel")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityRole")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityDesired")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityEffective")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityProbeStatus")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityReason")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityProbedAt")}</th>
                  <th className="px-3 py-2 text-left font-semibold text-foreground">{t("settingsPanel.capabilityDetails")}</th>
                </tr>
              </thead>
              <tbody>
                {capabilities.map((item) => {
                  const rowKey = `${item.provider_id}:${item.model_name}`;
                  const expanded = !!expandedCapabilityRows[rowKey];
                  return (
                    <Fragment key={rowKey}>
                      <tr className="border-t border-border">
                        <td className="px-3 py-2 text-foreground">
                          <div className="font-medium">{item.model_name}</div>
                          <div className="text-xs text-muted-foreground">{item.provider_id}</div>
                        </td>
                        <td className="px-3 py-2">
                          {item.role && (
                            <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${getRoleBadgeClass(item.role)}`}>
                              {item.role}
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-muted-foreground">{item.desired_mode}</td>
                        <td className="px-3 py-2">
                          <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold ${getEffectiveModeBadgeClass(item.effective_mode)}`}>
                            {item.effective_mode}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-muted-foreground">{item.probe_status}</td>
                        <td className="px-3 py-2 text-muted-foreground">{item.effective_mode_reason}</td>
                        <td className="px-3 py-2 text-muted-foreground">{item.probed_at ? formatDateTime(item.probed_at) : t("metrics.na")}</td>
                        <td className="px-3 py-2">
                          <Button
                            type="button"
                            onClick={() => toggleCapabilityRow(rowKey)}
                            variant="secondary"
                            size="sm"
                            className="px-2 py-1 text-xs"
                          >
                            {expanded ? t("settingsPanel.hideDetails") : t("settingsPanel.showDetails")}
                          </Button>
                        </td>
                      </tr>
                      {expanded && (
                        <tr className="border-t border-border bg-surface-muted">
                          <td colSpan={8} className="px-3 py-3 text-xs text-muted-foreground">
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
                              <pre className="overflow-x-auto rounded border border-border bg-surface p-2">
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
      </SectionCard>

      {/* RAG Operational Configuration */}
      <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
        <h3 className="mb-4 text-lg font-semibold text-foreground">{t("adminPage.ragSection")}</h3>
        {configLoading && <div className="text-sm text-muted-foreground">{t("settingsPanel.loadingDefaults")}</div>}
        <div className="space-y-4">
          <div>
            <label className="mb-2 block text-sm font-medium text-foreground">
              {t("settingsPanel.knowledgeCollection", { value: systemConfig?.rag_defaults.collection_name || "framework" })}
            </label>
            <Input
              type="text"
              value={(ragConfig.collection as string) || systemConfig?.rag_defaults.collection_name || "framework"}
              onChange={(e) => handleRagConfigChange("collection", e.target.value)}
              placeholder={systemConfig?.rag_defaults.collection_name || "e.g., framework"}
            />
          </div>
          <div>
            <label className="mb-2 block text-sm font-medium text-foreground">
              {t("settingsPanel.similarityThreshold")}
            </label>
            <Input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={(ragConfig.pill_score_threshold as number) || 0.72}
              onChange={(e) => handleRagConfigChange("pill_score_threshold", parseFloat(e.target.value))}
            />
          </div>
        </div>

        <div className="mt-6 border-t border-border pt-4">
          <h4 className="mb-2 text-md font-semibold text-foreground">{t("settingsPanel.reingestSectionTitle")}</h4>
          <p className="mb-3 text-sm text-muted-foreground">{t("settingsPanel.reingestDescription")}</p>
          <Button
            type="button"
            onClick={() => setConfirmReingest(true)}
            disabled={reingesting}
            variant="secondary"
            className="bg-warning text-warning-foreground hover:bg-warning/90 disabled:bg-muted disabled:text-muted-foreground"
          >
            {reingesting ? t("settingsPanel.reingesting") : t("settingsPanel.reingestAllDocuments")}
          </Button>
        </div>
      </div>

      {/* System Model Selection (vision + auxiliary) */}
      <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
        <h3 className="mb-4 text-lg font-semibold text-foreground">{t("adminPage.modelSection")}</h3>
        {configLoading ? (
          <p className="text-sm text-muted-foreground">{t("settingsPanel.loadingModels")}</p>
        ) : roleModelOptions.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("settingsPanel.noRoleModelsAvailable")}</p>
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
                <div key={roleName} className="rounded-xl border border-border p-4">
                  <div className="mb-2">
                    <p className="text-sm font-semibold text-foreground">
                      {t("settingsPanel.roleModelLabel", { role: roleName })}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {t("settingsPanel.roleProviderLocked", { provider: roleConfig.provider_id })}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {t("settingsPanel.systemDefault", { value: roleDefaultModel })}
                    </p>
                    {roleConfig.model_load_error && (
                      <p className="text-xs text-warning">{roleConfig.model_load_error}</p>
                    )}
                  </div>
                  <Select
                    value={selectedModel}
                    onChange={(e) =>
                      setPreferredModels((prev) => ({
                        ...prev,
                        [roleName]: e.target.value === roleDefaultModel ? "" : e.target.value,
                      }))
                    }
                  >
                    {mergedRoleModels.map((model) => (
                      <option key={`${roleName}:${model}`} value={model}>
                        {model}
                      </option>
                    ))}
                  </Select>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Save */}
      <div className="flex gap-3">
        <Button
          onClick={handleSave}
          disabled={saving || !settings}
          className="flex-1"
        >
          {saving ? t("common.saving") : t("settingsPanel.saveSettings")}
        </Button>
      </div>

      {/* Danger Zone */}
      <div className="rounded-2xl border border-destructive/30 bg-surface p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-destructive mb-1">{t("adminPage.dangerZoneSection")}</h3>
        <p className="mb-5 text-sm text-muted-foreground">{t("adminPage.dangerZoneDescription")}</p>

        <div className="space-y-3 mb-5">
          {(
            [
              { key: "conversations", label: t("adminPage.resetConversationsLabel"), description: t("adminPage.resetConversationsDescription") },
              { key: "workspace",     label: t("adminPage.resetWorkspaceLabel"),     description: t("adminPage.resetWorkspaceDescription") },
              { key: "memories",      label: t("adminPage.resetMemoriesLabel"),      description: t("adminPage.resetMemoriesDescription") },
              { key: "analytics",     label: t("adminPage.resetAnalyticsLabel"),     description: t("adminPage.resetAnalyticsDescription") },
              { key: "llm_probes",    label: t("adminPage.resetLlmProbesLabel"),     description: t("adminPage.resetLlmProbesDescription") },
            ] as { key: keyof ResetOptions; label: string; description: string }[]
          ).map(({ key, label, description }) => (
            <label key={key} className="flex items-start gap-3 cursor-pointer group">
              <Checkbox
                type="checkbox"
                checked={resetOptions[key]}
                onChange={() => toggleResetOption(key)}
                className="mt-0.5"
              />
              <span className="flex flex-col">
                <span className="text-sm font-medium text-foreground">{label}</span>
                <span className="text-xs text-foreground/60 mt-0.5">{description}</span>
              </span>
            </label>
          ))}
        </div>

        {!Object.values(resetOptions).some(Boolean) && (
          <p className="text-xs text-warning mb-3">{t("adminPage.resetNoneSelected")}</p>
        )}

        <Button
          type="button"
          onClick={() => setConfirmReset(true)}
          disabled={resetting || !Object.values(resetOptions).some(Boolean)}
          variant="destructive"
        >
          {resetting ? t("settingsPanel.resetting") : t("adminPage.resetSelectedButton")}
        </Button>
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
        title={t("adminPage.dangerZoneSection")}
        message={t("adminPage.resetConfirmMessage")}
        variant="danger"
        requireChecked={true}
        checkboxLabel={t("confirmDialog.resetCheckboxLabel")}
        confirmLabel={t("adminPage.resetSelectedButton")}
        onConfirm={() => {
          setConfirmReset(false);
          handleResetAllDatabases();
        }}
        onCancel={() => setConfirmReset(false)}
      />
    </div>
  );
}
