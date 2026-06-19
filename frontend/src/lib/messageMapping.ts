import type { MapData, Message, PersistedMessage, Source, ToolResultDetail, WeatherData } from "./types";

/**
 * Convert a persisted (DB) message into the local UI `Message` shape.
 *
 * Single source of truth for this mapping — used by the live chat runtime
 * (`ChatSessionContext`) and by any read-only surface that loads a stored
 * conversation (e.g. the `/chats` answer-evidence drawer). Restores the
 * Inspector `nodeTrace` and the answer `metrics` from persisted metadata so a
 * reloaded message drives the same evidence surfaces as a live one.
 */
export function toUiMessage(
  pm: PersistedMessage,
  formatTime: (value: Date | string | number) => string,
): Message {
  return {
    role: pm.role === "system" ? "assistant" : pm.role,
    content: pm.content,
    thinking: (pm.metadata?.thinking_content as string | undefined) ?? undefined,
    timestamp: pm.created_at ? formatTime(pm.created_at) : undefined,
    persistedId: pm.id,
    feedback: pm.feedback ?? undefined,
    // Inspector trace restored from persisted metadata (snake_case → camelCase).
    nodeTrace: Array.isArray(pm.metadata?.node_trace)
      ? (pm.metadata.node_trace as Array<Record<string, unknown>>).map((n) => ({
          node: String(n.node ?? ""),
          sequence: Number(n.sequence ?? 0),
          durationMs: typeof n.duration_ms === "number" ? n.duration_ms : null,
          status: n.status === "error" ? ("error" as const) : ("success" as const),
        }))
      : undefined,
    metrics: {
      output_tokens: (pm.metadata?.output_tokens as number | undefined) ?? pm.tokens_used ?? undefined,
      input_tokens: (pm.metadata?.input_tokens as number | undefined) ?? undefined,
      response_time_ms: pm.response_time_ms ?? undefined,
      model: pm.model_name ?? undefined,
      tools_executed: [
        ...(pm.tools_used?.map((t) => ({ name: t.name, status: t.status })) ?? []),
        ...(pm.metadata?.rag_attempted
          ? [{ name: "knowledge_base" as const, status: "success" as const }]
          : []),
      ],
      tool_results_detail: pm.metadata?.tool_results_detail as ToolResultDetail[] | undefined,
      sources: pm.metadata?.sources as Source[] | undefined,
      rag_attempted: (pm.metadata?.rag_attempted as boolean | undefined) ?? undefined,
      rag_doc_count: (pm.metadata?.rag_doc_count as number | undefined) ?? undefined,
      attachments_used: pm.metadata?.attachments_used as
        | Array<{ id: string; original_name: string }>
        | undefined,
      documents_used: pm.metadata?.documents_used as
        | Array<{ id: string; filename: string; version: number }>
        | undefined,
      memories_used: pm.metadata?.memories_used as
        | Array<{
            memory_id: string;
            text?: string;
            user_rating?: number | null;
            importance_score?: number | null;
            is_related?: boolean;
          }>
        | undefined,
      map_data: pm.metadata?.map_data as MapData | undefined,
      weather_data: pm.metadata?.weather_data as WeatherData | undefined,
    },
  };
}
