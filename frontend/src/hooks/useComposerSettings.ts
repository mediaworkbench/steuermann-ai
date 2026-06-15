"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { fetchSystemConfig, fetchUserSettings, updateUserSettings } from "@/lib/api";
import type { SystemConfig } from "@/lib/api";

interface UseComposerSettingsResult {
  ragEnabled: boolean;
  ragConfig: Record<string, unknown>;
  handleRagToggle: () => Promise<void>;
  toolToggles: Record<string, boolean>; // saved Settings preference (gates composer-menu membership)
  allowedTools: string[] | null; // role-allowed tool ids (null = not yet loaded / no restriction)
  selectedChatModel: string;
  availableChatModels: string[];
  handleModelChange: (model: string) => Promise<void>;
  systemConfig: SystemConfig | null;
  soundEnabled: boolean;
  showMetrics: boolean;
  maxContextTokens: number | null;
}

export function useComposerSettings(): UseComposerSettingsResult {
  const [ragEnabled, setRagEnabled] = useState<boolean>(true);
  const [ragConfig, setRagConfig] = useState<Record<string, unknown>>({ collection: "", top_k: 5, enabled: true });
  const [toolToggles, setToolToggles] = useState<Record<string, boolean>>({});
  const [allowedTools, setAllowedTools] = useState<string[] | null>(null);
  const [chatModel, setChatModel] = useState<string | null>(null);
  const [availableChatModels, setAvailableChatModels] = useState<string[]>([]);
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const preferredModelsRef = useRef<Record<string, string | null>>({});

  const selectedChatModel = chatModel || systemConfig?.default_model || availableChatModels[0] || "";

  // Denominator for the context ring. Track the *selected* chat model's window
  // (falling back to the role default) so switching models updates the gauge.
  const chatRole = systemConfig?.model_roles?.find((r) => r.role === "chat");
  const maxContextTokens =
    chatRole?.context_windows?.[selectedChatModel] ??
    chatRole?.context_window_tokens ??
    null;

  // Load user settings on mount: RAG config, tool toggles, chat model, sound
  useEffect(() => {
    fetchUserSettings().then((s) => {
      if (!s) return;
      const cfg = (s.rag_config as Record<string, unknown>) || { top_k: 5, enabled: true };
      setRagConfig(cfg);
      setRagEnabled((cfg.enabled as boolean) !== false);
      if (s.tool_toggles) setToolToggles(s.tool_toggles);
      if (s.allowed_tools) setAllowedTools(s.allowed_tools);
      const model = s.preferred_models?.chat ?? s.preferred_model ?? null;
      setChatModel(model);
      preferredModelsRef.current = s.preferred_models ?? {};
      const prefs = s.analytics_preferences as Record<string, unknown> | undefined;
      setSoundEnabled((prefs?.sound_enabled as boolean) ?? true);
      setShowMetrics((prefs?.show_metrics_panel as boolean) ?? true);
    }).catch(() => {});
  }, []);

  // Load system config for tools list and available models
  useEffect(() => {
    fetchSystemConfig().then((config) => {
      if (!config) return;
      setSystemConfig(config);
      const chatRole = config.model_roles?.find((r) => r.role === "chat");
      if (chatRole) {
        setAvailableChatModels(
          Array.from(new Set([chatRole.default_model, ...chatRole.available_models].filter(Boolean)))
        );
      }
    });
  }, []);

  const handleRagToggle = useCallback(async () => {
    const next = !ragEnabled;
    const updated = { ...ragConfig, enabled: next };
    setRagEnabled(next);
    setRagConfig(updated);
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || "/api/proxy";
    await fetch(`${apiBase}/api/settings/me`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rag_config: updated }),
    });
  }, [ragEnabled, ragConfig]);

  const handleModelChange = useCallback(async (model: string) => {
    setChatModel(model);
    const merged = { ...preferredModelsRef.current, chat: model };
    preferredModelsRef.current = merged;
    await updateUserSettings({ preferred_model: model, preferred_models: merged });
  }, []);

  return {
    ragEnabled,
    ragConfig,
    handleRagToggle,
    toolToggles,
    allowedTools,
    selectedChatModel,
    availableChatModels,
    handleModelChange,
    systemConfig,
    soundEnabled,
    showMetrics,
    maxContextTokens,
  };
}
