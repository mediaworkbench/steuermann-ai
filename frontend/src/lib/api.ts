export type ToolGroup = "text" | "vision" | "auxiliary";

export interface ToolCatalogItem {
  id: string;
  label: string;
  group: ToolGroup;
}

export interface UserSettings {
  user_id: string;
  tool_toggles: Record<string, boolean>;
  rag_config: Record<string, unknown>;
  analytics_preferences: Record<string, unknown>;
  preferred_model: string | null;
  preferred_models: Record<string, string | null>;
  theme?: string;
  language: string;
  updated_at: string | null;
  // Server-computed allowlist of tool ids for this user's role (admin ⇒ all).
  // Read-only — never sent on save.
  allowed_tools?: string[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "/api/proxy";

export async function fetchUserSettings(): Promise<UserSettings | null> {
  try {
    const response = await fetch(`${API_BASE}/api/settings/me`);
    if (!response.ok) {
      console.error(`Failed to fetch settings: ${response.status}`);
      return null;
    }
    return (await response.json()) as UserSettings;
  } catch (error) {
    console.error("Error fetching settings:", error);
    return null;
  }
}

export async function updateUserSettings(
  settings: Partial<Omit<UserSettings, "user_id" | "updated_at">>
): Promise<UserSettings | null> {
  try {
    const response = await fetch(`${API_BASE}/api/settings/me`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (!response.ok) {
      console.error(`Failed to update settings: ${response.status}`);
      return null;
    }
    return (await response.json()) as UserSettings;
  } catch (error) {
    console.error("Error updating settings:", error);
    return null;
  }
}

export async function fetchAvailableModels(): Promise<string[]> {
  try {
    const response = await fetch(`${API_BASE}/api/models`);
    if (!response.ok) {
      console.error(`Failed to fetch models: ${response.status}`);
      return [];
    }
    const data = await response.json();
    return data.models || [];
  } catch (error) {
    console.error("Error fetching models:", error);
    return [];
  }
}

export interface SystemConfig {
  available_tools: ToolCatalogItem[];
  rag_defaults: { collection_name: string; top_k: number };
  default_model: string;
  framework_version: string;
  supported_languages: string[];
  model_roles: Array<{
    role: string;
    provider_id: string;
    default_model: string;
    available_models: string[];
    model_load_error?: string | null;
    max_tokens?: number | null;
    context_window_tokens?: number | null;
    context_windows?: Record<string, number>;
  }>;
  profile: {
    id: string;
    display_name: string;
    role_label: string;
    description?: string | null;
    app_name?: string | null;
    theme: {
      colors: Record<string, string>;
      fonts: Record<string, string>;
      radius: Record<string, string>;
      custom_css_vars: Record<string, string>;
    };
  };
}

export interface ReingestAllResult {
  status: string;
  source: string;
  collection: string;
  language: string;
  processed: number;
  skipped: number;
  errors: number;
  total_chunks: number;
  output_tail: string;
}

export interface LLMCapabilityItem {
  provider_id: string;
  model_name: string;
  role?: string;
  desired_mode: string;
  configured_tool_calling_mode?: string;
  effective_mode: string;
  effective_mode_reason: string;
  probe_status: string;
  capability_mismatch: boolean;
  supports_bind_tools: boolean | null;
  supports_tool_schema: boolean | null;
  supports_vision: boolean | null;
  supports_reasoning: boolean;
  api_base?: string | null;
  error_message?: string | null;
  metadata?: Record<string, unknown>;
  probed_at: string | null;
  capabilities: Record<string, unknown>;
}

export interface LLMCapabilitiesResponse {
  status: string;
  profile_id: string;
  probe_ttl_seconds: number;
  items: LLMCapabilityItem[];
}

export type ProviderHealthStatus = "online" | "degraded" | "offline";

export interface ProviderHealthEndpoint {
  roles: string[];
  api_base: string;
  reachable: boolean;
  detail: string;
}

export interface ProviderHealth {
  status: ProviderHealthStatus;
  providers: ProviderHealthEndpoint[];
  checked_at?: string;
  error?: string;
}

export async function fetchSystemConfig(): Promise<SystemConfig | null> {
  try {
    const response = await fetch(`${API_BASE}/api/system-config`);
    if (!response.ok) {
      console.error(`Failed to fetch system config: ${response.status}`);
      return null;
    }
    return (await response.json()) as SystemConfig;
  } catch (error) {
    console.error("Error fetching system config:", error);
    return null;
  }
}

export interface RoleToolsConfig {
  tools: ToolCatalogItem[];
  roles: Record<string, string[]>;
}

// Admin-only: full tool catalog + each configurable role's allowed tool ids.
export async function fetchRoleTools(): Promise<RoleToolsConfig | null> {
  try {
    const response = await fetch(`${API_BASE}/api/admin/role-tools`);
    if (!response.ok) {
      console.error(`Failed to fetch role tools: ${response.status}`);
      return null;
    }
    return (await response.json()) as RoleToolsConfig;
  } catch (error) {
    console.error("Error fetching role tools:", error);
    return null;
  }
}

// Admin-only: set the explicit allowed tool ids for a configurable role.
export async function updateRoleTools(
  role: string,
  allowedTools: string[]
): Promise<RoleToolsConfig | null> {
  try {
    const response = await fetch(`${API_BASE}/api/admin/role-tools/${encodeURIComponent(role)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ allowed_tools: allowedTools }),
    });
    if (!response.ok) {
      console.error(`Failed to update role tools: ${response.status}`);
      return null;
    }
    return (await response.json()) as RoleToolsConfig;
  } catch (error) {
    console.error("Error updating role tools:", error);
    return null;
  }
}

export interface HeartbeatRateConfig {
  heartbeat_rate_minutes: number;
  default_rate_minutes: number;
  enabled: boolean;
  source: "override" | "default";
  last_run: {
    task_name: string;
    status: string;
    duration_ms: number;
    fired_at: string;
  } | null;
}

// Admin-only: read the effective heartbeat beat rate (minutes) and its source.
export async function fetchHeartbeatRate(): Promise<HeartbeatRateConfig | null> {
  try {
    const response = await fetch(`${API_BASE}/api/admin/settings/heartbeat-rate`);
    if (!response.ok) {
      console.error(`Failed to fetch heartbeat rate: ${response.status}`);
      return null;
    }
    return (await response.json()) as HeartbeatRateConfig;
  } catch (error) {
    console.error("Error fetching heartbeat rate:", error);
    return null;
  }
}

// Admin-only: set the global heartbeat beat rate (minutes).
export async function updateHeartbeatRate(
  minutes: number
): Promise<HeartbeatRateConfig | null> {
  try {
    const response = await fetch(`${API_BASE}/api/admin/settings/heartbeat-rate`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heartbeat_rate_minutes: minutes }),
    });
    if (!response.ok) {
      console.error(`Failed to update heartbeat rate: ${response.status}`);
      return null;
    }
    return (await response.json()) as HeartbeatRateConfig;
  } catch (error) {
    console.error("Error updating heartbeat rate:", error);
    return null;
  }
}

export interface HeartbeatRun {
  task_name: string;
  user_id: string | null;
  status: string;
  duration_ms: number;
  fired_at: string | null;
  detail: Record<string, unknown>;
}

export interface HeartbeatTaskInfo {
  name: string;
  type: string;
  scope: "global" | "per_user";
  cooldown_seconds: number;
  enabled: boolean;
  last_run: HeartbeatRun | null;
}

// Admin-only: the configured heartbeat tasks and each task's most recent run.
export async function fetchHeartbeatTasks(): Promise<HeartbeatTaskInfo[] | null> {
  try {
    const response = await fetch(`${API_BASE}/api/admin/heartbeat/tasks`);
    if (!response.ok) {
      console.error(`Failed to fetch heartbeat tasks: ${response.status}`);
      return null;
    }
    const data = (await response.json()) as { tasks: HeartbeatTaskInfo[] };
    return data.tasks ?? [];
  } catch (error) {
    console.error("Error fetching heartbeat tasks:", error);
    return null;
  }
}

// Admin-only: the heartbeat run log, newest first, optionally filtered.
export async function fetchHeartbeatRuns(opts?: {
  task?: string;
  user?: string;
  limit?: number;
}): Promise<HeartbeatRun[] | null> {
  try {
    const params = new URLSearchParams();
    if (opts?.task) params.set("task", opts.task);
    if (opts?.user) params.set("user", opts.user);
    params.set("limit", String(opts?.limit ?? 50));
    const response = await fetch(`${API_BASE}/api/admin/heartbeat/runs?${params.toString()}`);
    if (!response.ok) {
      console.error(`Failed to fetch heartbeat runs: ${response.status}`);
      return null;
    }
    const data = (await response.json()) as { runs: HeartbeatRun[] };
    return data.runs ?? [];
  } catch (error) {
    console.error("Error fetching heartbeat runs:", error);
    return null;
  }
}

export async function triggerReingestAllDocuments(): Promise<ReingestAllResult> {
  const response = await fetch(`${API_BASE}/api/ingestion/reingest-all`, {
    method: "POST",
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;

  if (!response.ok) {
    const detail = payload?.detail;
    if (typeof detail === "string") {
      throw new Error(detail);
    }
    if (detail?.message) {
      throw new Error(detail.message);
    }
    throw new Error(`Failed to trigger re-ingestion: ${response.status}`);
  }

  return payload as ReingestAllResult;
}

export interface ResetOptions {
  conversations: boolean;
  workspace: boolean;
  memories: boolean;
  analytics: boolean;
  llm_probes: boolean;
}

export interface UserResetOptions {
  conversations: boolean;
  workspace: boolean;
  memories: boolean;
}

export async function resetMyData(
  options: UserResetOptions,
): Promise<{ status: string; errors: string[] }> {
  const response = await fetch(`${API_BASE}/api/user/reset-my-data`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  const payload = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) {
    throw new Error(payload?.detail || `Reset failed: ${response.status}`);
  }
  return payload;
}

export async function resetAllDatabases(options: ResetOptions): Promise<{ status: string; errors: string[] }> {
  const response = await fetch(`${API_BASE}/api/admin/reset-all-databases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  const payload = response.headers.get("content-type")?.includes("application/json")
    ? await response.json()
    : null;
  if (!response.ok) {
    throw new Error(payload?.detail || `Reset failed: ${response.status}`);
  }
  return payload;
}

export async function fetchLLMCapabilities(): Promise<LLMCapabilitiesResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/llm/capabilities`);
    if (!response.ok) {
      console.error(`Failed to fetch LLM capabilities: ${response.status}`);
      return null;
    }
    return (await response.json()) as LLMCapabilitiesResponse;
  } catch (error) {
    console.error("Error fetching LLM capabilities:", error);
    return null;
  }
}

export async function fetchProviderHealth(): Promise<ProviderHealth | null> {
  try {
    const response = await fetch(`${API_BASE}/api/llm/health`, { cache: "no-store" });
    if (!response.ok) {
      console.error(`Failed to fetch provider health: ${response.status}`);
      return null;
    }
    return (await response.json()) as ProviderHealth;
  } catch (error) {
    console.error("Error fetching provider health:", error);
    return null;
  }
}

// Force a full LLM capability reprobe (refreshes tool-calling mode + model metadata).
// Best-effort: callers should not block on the result.
export async function triggerLLMReprobe(): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/llm/reprobe`, { method: "POST" });
  } catch (error) {
    console.error("Error triggering LLM reprobe:", error);
  }
}

export async function fetchMetrics() {
  try {
    const response = await fetch(`${API_BASE}/api/metrics`);
    if (!response.ok) {
      console.error(`Failed to fetch metrics: ${response.status}`);
      return null;
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return null;
  }
}

// Metrics API

export interface MetricsData {
  requests: Record<string, number>;
  tokens: Record<string, number>;
  latency: Record<string, number>;
  sessions: Record<string, number>;
  memory_ops: Record<string, number>;
  memory_ops_by_status: Record<string, number>;
  llm_calls: Record<string, number>;
  attachments?: Record<string, number>;
  attachment_retries?: Record<string, number>;
  profile_guardrails?: Record<string, number>;
  timestamp?: string;
}

export async function fetchMetricsData(): Promise<MetricsData | null> {
  try {
    const response = await fetch(`${API_BASE}/api/metrics`);
    if (!response.ok) {
      console.error(`Failed to fetch metrics: ${response.status}`);
      return null;
    }
    return (await response.json()) as MetricsData;
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return null;
  }
}

// Analytics API functions

export interface UsageTrend {
  date: string;
  requests: number;
  users: number;
}

export interface UsageTrendsResponse {
  period_days: number;
  trends: UsageTrend[];
  total_requests: number;
  unique_users: number;
}

export interface TokenConsumption {
  date: string;
  total_tokens: number;
  avg_tokens: number;
  requests: number;
}

export interface TokenConsumptionResponse {
  period_days: number;
  consumption: TokenConsumption[];
  total_tokens: number;
  avg_tokens_per_request: number;
}

export interface LatencyData {
  date: string;
  avg_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  requests: number;
}

export interface LatencyAnalysisResponse {
  period_days: number;
  latency_data: LatencyData[];
  overall_avg_ms: number;
  overall_min_ms: number;
  overall_max_ms: number;
  total_requests: number;
}

export interface MemoryTrendPoint {
  date: string;
  loads: number;
  updates: number;
  errors: number;
  error_rate: number;
  avg_quality_score: number;
}

export interface MemoryRetrievalQualityData {
  retrieval_signals_total: number;
  retrieved_with_prior_rating: number;
  retrieved_without_prior_rating: number;
  prior_rating_coverage: number;
  rating_bucket_distribution: Record<string, number>;
  rated_after_retrieval_total: number;
  feedback_coverage: number;
  timestamp: string;
}

export interface MessageQualityPoint {
  date: string;
  up_count: number;
  down_count: number;
  total_feedback: number;
  total_assistant_messages: number;
  net_score: number;
}

export interface MessageQualityResponse {
  period_days: number;
  quality_data: MessageQualityPoint[];
  total_up: number;
  total_down: number;
  total_feedback: number;
  total_assistant_messages: number;
  net_score: number;
  feedback_rate: number;
}

export interface MemoryTrendsResponse {
  period_days: number;
  trends: MemoryTrendPoint[];
  totals: {
    loads: number;
    updates: number;
    errors: number;
    error_rate: number;
  };
}

export async function fetchUsageTrends(days: number = 30): Promise<UsageTrendsResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/analytics/usage-trends?days=${days}`);
    if (!response.ok) {
      console.error(`Failed to fetch usage trends: ${response.status}`);
      return null;
    }
    return (await response.json()) as UsageTrendsResponse;
  } catch (error) {
    console.error("Error fetching usage trends:", error);
    return null;
  }
}

export async function fetchTokenConsumption(days: number = 30): Promise<TokenConsumptionResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/analytics/token-consumption?days=${days}`);
    if (!response.ok) {
      console.error(`Failed to fetch token consumption: ${response.status}`);
      return null;
    }
    return (await response.json()) as TokenConsumptionResponse;
  } catch (error) {
    console.error("Error fetching token consumption:", error);
    return null;
  }
}

export async function fetchLatencyAnalysis(days: number = 30): Promise<LatencyAnalysisResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/analytics/latency-analysis?days=${days}`);
    if (!response.ok) {
      console.error(`Failed to fetch latency analysis: ${response.status}`);
      return null;
    }
    return (await response.json()) as LatencyAnalysisResponse;
  } catch (error) {
    console.error("Error fetching latency analysis:", error);
    return null;
  }
}

export async function fetchMemoryTrends(days: number = 30): Promise<MemoryTrendsResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/analytics/memory-trends?days=${days}`);
    if (!response.ok) {
      console.error(`Failed to fetch memory trends: ${response.status}`);
      return null;
    }
    return (await response.json()) as MemoryTrendsResponse;
  } catch (error) {
    console.error("Error fetching memory trends:", error);
    return null;
  }
}

export async function fetchMemoryRetrievalQuality(): Promise<MemoryRetrievalQualityData | null> {
  try {
    const response = await fetch(`${API_BASE}/api/analytics/memory-retrieval-quality`);
    if (!response.ok) {
      console.error(`Failed to fetch memory retrieval quality: ${response.status}`);
      return null;
    }
    return (await response.json()) as MemoryRetrievalQualityData;
  } catch (error) {
    console.error("Error fetching memory retrieval quality:", error);
    return null;
  }
}

export async function fetchMessageQuality(days: number = 30): Promise<MessageQualityResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/analytics/message-quality?days=${days}`);
    if (!response.ok) {
      console.error(`Failed to fetch message quality: ${response.status}`);
      return null;
    }
    return (await response.json()) as MessageQualityResponse;
  } catch (error) {
    console.error("Error fetching message quality:", error);
    return null;
  }
}
// ── Conversations API ─────────────────────────────────────────────────

import type {
  ChatResponse,
  Conversation,
  ConversationAttachment,
  ConversationListResponse,
  ConversationDetailResponse,
  PersistedMessage,
  WorkspaceActionRequest,
} from "@/lib/types";

export async function createConversation(
  title: string = "New conversation",
  language: string = "en",
): Promise<Conversation | null> {
  try {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, language }),
    });
    if (!response.ok) {
      console.error(`Failed to create conversation: ${response.status}`);
      return null;
    }
    return (await response.json()) as Conversation;
  } catch (error) {
    console.error("Error creating conversation:", error);
    return null;
  }
}

export async function fetchConversations(
  limit: number = 50,
  offset: number = 0,
  q?: string,
): Promise<ConversationListResponse | null> {
  try {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
    });
    if (q && q.trim()) params.set("q", q.trim());
    const response = await fetch(`${API_BASE}/api/conversations?${params}`);
    if (!response.ok) {
      console.error(`Failed to fetch conversations: ${response.status}`);
      return null;
    }
    return (await response.json()) as ConversationListResponse;
  } catch (error) {
    console.error("Error fetching conversations:", error);
    return null;
  }
}

export async function fetchConversation(
  conversationId: string,
): Promise<ConversationDetailResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`);
    if (!response.ok) {
      console.error(`Failed to fetch conversation: ${response.status}`);
      return null;
    }
    return (await response.json()) as ConversationDetailResponse;
  } catch (error) {
    console.error("Error fetching conversation:", error);
    return null;
  }
}

export async function updateConversation(
  conversationId: string,
  updates: { title?: string; pinned?: boolean; language?: string; title_manual?: boolean },
): Promise<Conversation | null> {
  try {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    if (!response.ok) {
      console.error(`Failed to update conversation: ${response.status}`);
      return null;
    }
    return (await response.json()) as Conversation;
  } catch (error) {
    console.error("Error updating conversation:", error);
    return null;
  }
}

export async function deleteConversation(conversationId: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
      method: "DELETE",
    });
    return response.status === 204;
  } catch (error) {
    console.error("Error deleting conversation:", error);
    return false;
  }
}

export async function setMessageFeedback(
  conversationId: string,
  messageId: number,
  feedback: "up" | "down" | null,
): Promise<PersistedMessage | null> {
  try {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/messages/${messageId}/feedback`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback }),
      },
    );
    if (!response.ok) return null;
    return (await response.json()) as PersistedMessage;
  } catch (error) {
    console.error("Error setting feedback:", error);
    return null;
  }
}

export async function rateMemory(memoryId: string, rating: number): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/memories/${memoryId}/rate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    });
    return response.ok;
  } catch (error) {
    console.error("Error rating memory:", error);
    return false;
  }
}

export async function fetchMemories(
  limit: number = 50,
  offset: number = 0,
): Promise<import("@/lib/types").MemoryListResponse | null> {
  try {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
    });
    const response = await fetch(`${API_BASE}/api/memories?${params}`);
    if (!response.ok) {
      console.error(`Failed to fetch memories: ${response.status}`);
      return null;
    }
    return (await response.json()) as import("@/lib/types").MemoryListResponse;
  } catch (error) {
    console.error("Error fetching memories:", error);
    return null;
  }
}

export async function fetchMemoryStats(): Promise<import("@/lib/types").MemoryStats | null> {
  try {
    const response = await fetch(`${API_BASE}/api/memories/stats`);
    if (!response.ok) {
      console.error(`Failed to fetch memory stats: ${response.status}`);
      return null;
    }
    return (await response.json()) as import("@/lib/types").MemoryStats;
  } catch (error) {
    console.error("Error fetching memory stats:", error);
    return null;
  }
}

export async function deleteMemory(memoryId: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/memories/${memoryId}`, {
      method: "DELETE",
    });
    return response.ok;
  } catch (error) {
    console.error("Error deleting memory:", error);
    return false;
  }
}


export async function exportConversation(
  conversationId: string,
  format: "json" | "markdown" = "json",
): Promise<string | Record<string, unknown> | null> {
  try {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/export?fmt=${format}`,
    );
    if (!response.ok) return null;
    if (format === "markdown") return await response.text();
    return await response.json();
  } catch (error) {
    console.error("Error exporting conversation:", error);
    return null;
  }
}

export async function fetchConversationAttachments(
  conversationId: string,
): Promise<ConversationAttachment[]> {
  try {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}/attachments`);
    if (!response.ok) return [];
    const data = await response.json();
    return (data.attachments || []) as ConversationAttachment[];
  } catch (error) {
    console.error("Error fetching attachments:", error);
    return [];
  }
}

export async function uploadConversationAttachment(
  conversationId: string,
  file: File,
): Promise<ConversationAttachment | null> {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}/attachments`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      console.error("Failed to upload attachment:", error?.detail || response.status);
      return null;
    }
    const data = await response.json();
    return (data.attachment || null) as ConversationAttachment | null;
  } catch (error) {
    console.error("Error uploading attachment:", error);
    return null;
  }
}

export async function deleteConversationAttachment(
  conversationId: string,
  attachmentId: string,
): Promise<boolean> {
  try {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/attachments/${attachmentId}`,
      { method: "DELETE" },
    );
    return response.status === 204;
  } catch (error) {
    console.error("Error deleting attachment:", error);
    return false;
  }
}

export async function attachWorkspaceDocumentToConversation(
  conversationId: string,
  documentId: string,
): Promise<ConversationAttachment | null> {
  try {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/attachments/from-workspace`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: documentId }),
      },
    );
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      console.error("Failed to attach workspace document:", error?.detail || response.status);
      return null;
    }
    const data = await response.json();
    return (data.attachment || null) as ConversationAttachment | null;
  } catch (error) {
    console.error("Error attaching workspace document:", error);
    return null;
  }
}

export async function clearAllWorkspaceDocuments(): Promise<number> {
  try {
    const response = await fetch(`${API_BASE}/api/workspace/documents`, {
      method: "DELETE",
    });
    if (!response.ok) return 0;
    const data = await response.json();
    return typeof data.deleted === "number" ? data.deleted : 0;
  } catch (error) {
    console.error("Error clearing workspace documents:", error);
    return 0;
  }
}

export async function runWorkspaceAction(
  conversationId: string,
  message: string,
  workspaceAction: WorkspaceActionRequest,
  language: string = "en",
): Promise<ChatResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        language,
        conversation_id: conversationId,
        workspace_action: workspaceAction,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => null);
      console.error("Workspace action failed:", error?.detail || response.status);
      return null;
    }

    return (await response.json()) as ChatResponse;
  } catch (error) {
    console.error("Error running workspace action:", error);
    return null;
  }
}

// ── RAG knowledge explorer (admin) ─────────────────────────────────────────

export interface RagSearchHit {
  id: string | number | null;
  score: number;
  text: string;
  file_name: string;
  file_path: string;
  chunk_index: number | null;
  chunk_count: number | null;
  detected_language: string | null;
  language_confidence: number | null;
  above_cutoff: boolean;
  metadata: Record<string, unknown>;
}

export interface RagSearchResponse {
  items: RagSearchHit[];
  count: number;
  query: string;
  collection: string;
  top_k: number;
  production_threshold: number;
}

export interface RagCollection {
  name: string;
  points_count: number | null;
}

export interface RagCollectionsResponse {
  collections: RagCollection[];
  default_collection: string;
}

export interface RagSearchParams {
  q: string;
  topK?: number;
  collection?: string;
}

/** Search the RAG knowledge base. Throws with the backend detail on failure. */
export async function searchRag(params: RagSearchParams): Promise<RagSearchResponse> {
  const search = new URLSearchParams({ q: params.q });
  if (params.topK != null) search.set("top_k", String(params.topK));
  if (params.collection) search.set("collection", params.collection);

  const response = await fetch(`${API_BASE}/api/rag/search?${search}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Knowledge base search failed: ${response.status}`);
  }
  return (await response.json()) as RagSearchResponse;
}

/** List Qdrant collections with point counts. Returns null on failure. */
export async function fetchRagCollections(): Promise<RagCollectionsResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/rag/collections`);
    if (!response.ok) {
      console.error(`Failed to fetch RAG collections: ${response.status}`);
      return null;
    }
    return (await response.json()) as RagCollectionsResponse;
  } catch (error) {
    console.error("Error fetching RAG collections:", error);
    return null;
  }
}

// ── Admin: user & role management (administrator-only; enforced server-side) ──

export interface AdminUser {
  user_id: string;
  username: string;
  email: string;
  role_name: string | null;
  status: string;
  must_change_password: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface AdminRole {
  role_id: number;
  role_name: string;
  description?: string | null;
}

/** List users (paginated). Returns null on failure. */
export async function fetchUsers(
  limit: number = 100,
  offset: number = 0,
): Promise<{ users: AdminUser[]; total: number } | null> {
  try {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    const response = await fetch(`${API_BASE}/api/admin/users?${params}`);
    if (!response.ok) {
      console.error(`Failed to fetch users: ${response.status}`);
      return null;
    }
    return (await response.json()) as { users: AdminUser[]; total: number };
  } catch (error) {
    console.error("Error fetching users:", error);
    return null;
  }
}

/** List the fixed roles. Returns [] on failure. */
export async function fetchRoles(): Promise<AdminRole[]> {
  try {
    const response = await fetch(`${API_BASE}/api/admin/roles`);
    if (!response.ok) return [];
    const data = await response.json();
    return (data.roles || []) as AdminRole[];
  } catch (error) {
    console.error("Error fetching roles:", error);
    return [];
  }
}

async function _detail(response: Response, fallback: string): Promise<string> {
  const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
  return payload?.detail || fallback;
}

/** Create a user. Throws Error(detail) on failure (e.g. 409 duplicate, 400 bad role). */
export async function createUser(body: {
  username: string;
  email: string;
  role: string;
}): Promise<{ user: AdminUser; temporary_password: string }> {
  const response = await fetch(`${API_BASE}/api/admin/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await _detail(response, "Failed to create user"));
  return (await response.json()) as { user: AdminUser; temporary_password: string };
}

/** Update a user's role/status and/or reset their password. Throws Error(detail) on failure. */
export async function updateUser(
  userId: string,
  body: { role?: string; status?: string; reset_password?: boolean },
): Promise<{ user: AdminUser; temporary_password?: string }> {
  const response = await fetch(`${API_BASE}/api/admin/users/${encodeURIComponent(userId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await _detail(response, "Failed to update user"));
  return (await response.json()) as { user: AdminUser; temporary_password?: string };
}

/** Delete a user. Throws Error(detail) on failure (e.g. last-admin / self guardrails). */
export async function deleteUser(userId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/admin/users/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 204) {
    throw new Error(await _detail(response, "Failed to delete user"));
  }
}
