"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { fetchSystemConfig, fetchUserSettings, updateUserSettings } from "@/lib/api";
import type { SystemConfig } from "@/lib/api";
import { CURRENT_USER_ID } from "@/lib/runtime";

interface UseComposerSettingsResult {
  ragEnabled: boolean;
  ragConfig: Record<string, unknown>;
  handleRagToggle: () => Promise<void>;
  toolToggles: Record<string, boolean>;
  handleToolToggle: (toolId: string) => Promise<void>;
  selectedChatModel: string;
  availableChatModels: string[];
  handleModelChange: (model: string) => Promise<void>;
  systemConfig: SystemConfig | null;
  soundEnabled: boolean;
  maxContextTokens: number | null;
}

export function useComposerSettings(): UseComposerSettingsResult {
  const [ragEnabled, setRagEnabled] = useState<boolean>(true);
  const [ragConfig, setRagConfig] = useState<Record<string, unknown>>({ collection: "", top_k: 5, enabled: true });
  const [toolToggles, setToolToggles] = useState<Record<string, boolean>>({});
  const [chatModel, setChatModel] = useState<string | null>(null);
  const [availableChatModels, setAvailableChatModels] = useState<string[]>([]);
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
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
    fetchUserSettings(CURRENT_USER_ID).then((s) => {
      if (!s) return;
      const cfg = (s.rag_config as Record<string, unknown>) || { collection: "", top_k: 5, enabled: true };
      setRagConfig(cfg);
      setRagEnabled((cfg.enabled as boolean) !== false);
      if (s.tool_toggles) setToolToggles(s.tool_toggles);
      const model = s.preferred_models?.chat ?? s.preferred_model ?? null;
      setChatModel(model);
      preferredModelsRef.current = s.preferred_models ?? {};
      const prefs = s.analytics_preferences as Record<string, unknown> | undefined;
      setSoundEnabled((prefs?.sound_enabled as boolean) ?? true);
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
    await fetch(`${apiBase}/api/settings/user/${CURRENT_USER_ID}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rag_config: updated }),
    });
  }, [ragEnabled, ragConfig]);

  const handleToolToggle = useCallback(async (toolId: string) => {
    // Intentionally asymmetric: undefined means enabled (default-on).
    // Do not simplify to !toolToggles[toolId] — that breaks the default-on behaviour.
    const next = { ...toolToggles, [toolId]: toolToggles[toolId] !== false ? false : true };
    setToolToggles(next);
    await updateUserSettings(CURRENT_USER_ID, { tool_toggles: next });
  }, [toolToggles]);

  const handleModelChange = useCallback(async (model: string) => {
    setChatModel(model);
    const merged = { ...preferredModelsRef.current, chat: model };
    preferredModelsRef.current = merged;
    await updateUserSettings(CURRENT_USER_ID, { preferred_model: model, preferred_models: merged });
  }, []);

  return {
    ragEnabled,
    ragConfig,
    handleRagToggle,
    toolToggles,
    handleToolToggle,
    selectedChatModel,
    availableChatModels,
    handleModelChange,
    systemConfig,
    soundEnabled,
    maxContextTokens,
  };
}
