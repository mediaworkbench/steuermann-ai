import { CURRENT_USER_ID } from "@/lib/runtime";

export interface UserSettings {
  user_id: string;
  tool_toggles: Record<string, boolean>;
  rag_config: Record<string, unknown>;
  analytics_preferences: Record<string, unknown>;
  preferred_model: string | null;
  preferred_models: Record<string, string | null>;
  language: string;
  updated_at: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "/api/proxy";

export async function fetchUserSettings(userId: string): Promise<UserSettings | null> {
  try {
    const response = await fetch(`${API_BASE}/api/settings/user/${userId}`);
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
  userId: string,
  settings: Partial<Omit<UserSettings, "user_id" | "updated_at">>
): Promise<UserSettings | null> {
  try {
    const response = await fetch(`${API_BASE}/api/settings/user/${userId}`, {
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
  available_tools: Array<{ id: string; label: string }>;
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
  userId: string = CURRENT_USER_ID,
  title: string = "New conversation",
  language: string = "en",
): Promise<Conversation | null> {
  try {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, title, language }),
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
  userId: string = CURRENT_USER_ID,
  limit: number = 50,
  offset: number = 0,
  q?: string,
): Promise<ConversationListResponse | null> {
  try {
    const params = new URLSearchParams({
      user_id: userId,
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
  updates: { title?: string; pinned?: boolean; language?: string },
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
  userId: string = CURRENT_USER_ID,
  limit: number = 50,
  offset: number = 0,
): Promise<import("@/lib/types").MemoryListResponse | null> {
  try {
    const params = new URLSearchParams({
      user_id: userId,
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

export async function fetchMemoryStats(
  userId: string = CURRENT_USER_ID,
): Promise<import("@/lib/types").MemoryStats | null> {
  try {
    const response = await fetch(`${API_BASE}/api/memories/stats?user_id=${encodeURIComponent(userId)}`);
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
  userId: string = CURRENT_USER_ID,
): Promise<ConversationAttachment | null> {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_id", userId);

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
  userId: string = CURRENT_USER_ID,
  language: string = "en",
): Promise<ChatResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        user_id: userId,
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

export type RagSearchMode = "raw" | "production";

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
  mode: RagSearchMode;
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
  mode?: RagSearchMode;
  topK?: number;
  collection?: string;
}

/** Search the RAG knowledge base. Throws with the backend detail on failure. */
export async function searchRag(params: RagSearchParams): Promise<RagSearchResponse> {
  const search = new URLSearchParams({ q: params.q });
  if (params.mode) search.set("mode", params.mode);
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
