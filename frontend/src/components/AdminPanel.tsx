"use client";

import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { ActionConfirmDialog } from "@/components/product/ActionConfirmDialog";
import { DangerConfirmDialog } from "@/components/product/DangerConfirmDialog";
import { DangerOptionsList } from "@/components/product/DangerOptionsList";
import { DangerSelectionActions } from "@/components/product/DangerSelectionActions";
import { DiagnosticsSectionCard } from "@/components/product/DiagnosticsSectionCard";
import { FormFieldLabel } from "@/components/product/FormFieldLabel";
import { PanelLoadingState } from "@/components/product/PanelLoadingState";
import { PrimarySaveBar } from "@/components/product/PrimarySaveBar";
import { RoleModelSelectionSection } from "@/components/product/RoleModelSelectionSection";
import { SectionStateText } from "@/components/product/SectionStateText";
import { SubsectionHeader } from "@/components/product/SubsectionHeader";
import { TitledSectionCard } from "@/components/product/TitledSectionCard";
import { buildCapabilitiesTableLabels } from "@/components/product/buildCapabilitiesTableLabels";
import { updatePreferredModelSelection } from "@/components/product/modelSelection";
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
import { Input } from "@/components/ui/Input";

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
    return <PanelLoadingState label={t("common.loading")} />;
  }

  const roleModelOptions = (systemConfig?.model_roles || []).filter((r) =>
    ADMIN_MODEL_ROLES.includes(r.role)
  );

  return (
    <div className="space-y-6">

      {/* LLM Capability Diagnostics */}
      <DiagnosticsSectionCard
        title={t("adminPage.llmSection")}
        description={
          <>
            {t("settingsPanel.capabilitiesSubtitle")}
            {probeTtlSeconds !== null && (
              <span>{t("settingsPanel.capabilitiesTtl", { value: probeTtlSeconds })}</span>
            )}
          </>
        }
        items={capabilities}
        loading={capabilitiesLoading}
        error={capabilitiesError}
        expandedRows={expandedCapabilityRows}
        onToggleRow={toggleCapabilityRow}
        formatDateTime={formatDateTime}
        labels={buildCapabilitiesTableLabels(t)}
        legendTitle={t("settingsPanel.legendTitle")}
        legendNative={t("settingsPanel.legendNative")}
        legendStructured={t("settingsPanel.legendStructured")}
        legendReact={t("settingsPanel.legendReact")}
        loadingLabel={t("settingsPanel.capabilitiesLoading")}
        emptyLabel={t("settingsPanel.capabilitiesEmpty")}
        copyLabel={t("settingsPanel.copyDiagnostics")}
        copyingLabel={t("common.loading")}
        refreshLabel={t("common.refresh")}
        copying={copyingDiagnostics}
        onCopy={handleCopyDiagnostics}
        onRefresh={() => void loadCapabilities()}
      />

      {/* RAG Operational Configuration */}
      <TitledSectionCard title={t("adminPage.ragSection")}>
        {configLoading && <SectionStateText>{t("settingsPanel.loadingDefaults")}</SectionStateText>}
        <div className="space-y-4">
          <div>
            <FormFieldLabel>
              {t("settingsPanel.knowledgeCollection", { value: systemConfig?.rag_defaults.collection_name || "framework" })}
            </FormFieldLabel>
            <Input
              type="text"
              value={(ragConfig.collection as string) || systemConfig?.rag_defaults.collection_name || "framework"}
              onChange={(e) => handleRagConfigChange("collection", e.target.value)}
              placeholder={systemConfig?.rag_defaults.collection_name || "e.g., framework"}
            />
          </div>
          <div>
            <FormFieldLabel>{t("settingsPanel.similarityThreshold")}</FormFieldLabel>
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
          <SubsectionHeader
            title={t("settingsPanel.reingestSectionTitle")}
            description={t("settingsPanel.reingestDescription")}
          />
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
      </TitledSectionCard>

      {/* System Model Selection (vision + auxiliary) */}
      <RoleModelSelectionSection
        title={t("adminPage.modelSection")}
        loading={configLoading}
        loadingLabel={t("settingsPanel.loadingModels")}
        emptyLabel={t("settingsPanel.noRoleModelsAvailable")}
        roleConfigs={roleModelOptions}
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
        disabled={!settings}
        label={t("settingsPanel.saveSettings")}
        loadingLabel={t("common.saving")}
      />

      {/* Danger Zone */}
      <TitledSectionCard
        title={t("adminPage.dangerZoneSection")}
        description={t("adminPage.dangerZoneDescription")}
        tone="danger"
        headerClassName="mb-5"
      >
        <DangerOptionsList
          options={(
            [
              { key: "conversations", label: t("adminPage.resetConversationsLabel"), description: t("adminPage.resetConversationsDescription") },
              { key: "workspace",     label: t("adminPage.resetWorkspaceLabel"),     description: t("adminPage.resetWorkspaceDescription") },
              { key: "memories",      label: t("adminPage.resetMemoriesLabel"),      description: t("adminPage.resetMemoriesDescription") },
              { key: "analytics",     label: t("adminPage.resetAnalyticsLabel"),     description: t("adminPage.resetAnalyticsDescription") },
              { key: "llm_probes",    label: t("adminPage.resetLlmProbesLabel"),     description: t("adminPage.resetLlmProbesDescription") },
            ] as { key: keyof ResetOptions; label: string; description: string }[]
          ).map(({ key, label, description }) => ({
            key,
            checked: resetOptions[key],
            onToggle: () => toggleResetOption(key),
            label,
            description,
            className: "group",
          }))}
        />

        <DangerSelectionActions
          hasSelection={Object.values(resetOptions).some(Boolean)}
          hintText={t("adminPage.resetNoneSelected")}
          onAction={() => setConfirmReset(true)}
          loading={resetting}
          loadingLabel={t("settingsPanel.resetting")}
          actionLabel={t("adminPage.resetSelectedButton")}
          disabled={resetting}
        />
      </TitledSectionCard>

      <ActionConfirmDialog
        isOpen={confirmReingest}
        title={t("settingsPanel.reingestSectionTitle")}
        message={t("settingsPanel.confirmReingestAll")}
        onConfirm={() => {
          setConfirmReingest(false);
          handleReingestAll();
        }}
        onCancel={() => setConfirmReingest(false)}
      />

      <DangerConfirmDialog
        isOpen={confirmReset}
        title={t("adminPage.dangerZoneSection")}
        message={t("adminPage.resetConfirmMessage")}
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
