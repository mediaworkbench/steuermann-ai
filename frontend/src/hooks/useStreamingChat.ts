"use client";

import { useCallback, useRef, useState } from "react";
import type { ChatResponse, NodeTraceEntry } from "@/lib/types";

// ── Public API ────────────────────────────────────────────────────────

export interface StreamingChatParams {
  message: string;
  userId: string;
  conversationId: string | null;
  attachmentIds: string[];
  documentIds: string[];
  ragEnabled: boolean;
}

/** Signalled mid-stream when the model is about to write back to a document. */
export interface WritebackPending {
  documentId: string;
  filename: string;
}

export interface UseStreamingChatReturn {
  streamingContent: string;
  isStreaming: boolean;
  streamError: string | null;
  streamWarning: string | null;
  toolCallStatus: { name: string; status: "start" | "end"; label: string } | null;
  nodeStatus: string | null;
  nodeTrace: NodeTraceEntry[];
  finalMetadata: ChatResponse["metadata"] | null;
  writebackPending: WritebackPending | null;
  wasCancelled: boolean;
  thinkingContent: string;
  isThinking: boolean;
  sendMessage: (params: StreamingChatParams) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

// ── Helpers ───────────────────────────────────────────────────────────

function buildMetadataFromSSE(parsed: Record<string, unknown>): ChatResponse["metadata"] {
  const toolResults = (parsed.tool_results as Record<string, unknown>) ?? {};
  const toolKeys = Object.keys(toolResults);
  const ragAttempted = parsed.rag_attempted as boolean | undefined;
  const knowledgeCtx = parsed.knowledge_context;
  if (ragAttempted || (Array.isArray(knowledgeCtx) && knowledgeCtx.length > 0)) {
    toolKeys.push("knowledge_base");
  }

  return {
    tokens_used: (parsed.tokens_used as number) ?? 0,
    input_tokens: parsed.input_tokens as number | undefined,
    output_tokens: parsed.output_tokens as number | undefined,
    model_used: parsed.model_used as string | undefined,
    tools_executed: toolKeys,
    tool_results_detail: parsed.tool_results_detail as ChatResponse["metadata"]["tool_results_detail"],
    sources: parsed.sources as ChatResponse["metadata"]["sources"],
    rag_attempted: parsed.rag_attempted as boolean | undefined,
    rag_doc_count: parsed.rag_doc_count as number | undefined,
    memories_used: parsed.loaded_memory as ChatResponse["metadata"]["memories_used"],
    workspace_document_writeback: null,
    map_data: parsed.map_data as ChatResponse["metadata"]["map_data"],
    context_breakdown: parsed.context_breakdown as ChatResponse["metadata"]["context_breakdown"],
  };
}

// ── Hook ──────────────────────────────────────────────────────────────

export function useStreamingChat(): UseStreamingChatReturn {
  const [streamingContent, setStreamingContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streamWarning, setStreamWarning] = useState<string | null>(null);
  const [toolCallStatus, setToolCallStatus] = useState<{ name: string; status: "start" | "end"; label: string } | null>(null);
  const [nodeStatus, setNodeStatus] = useState<string | null>(null);
  const [nodeTrace, setNodeTrace] = useState<NodeTraceEntry[]>([]);
  const [finalMetadata, setFinalMetadata] = useState<ChatResponse["metadata"] | null>(null);
  const [writebackPending, setWritebackPending] = useState<WritebackPending | null>(null);
  const [wasCancelled, setWasCancelled] = useState(false);
  const [thinkingContent, setThinkingContent] = useState("");
  const [isThinking, setIsThinking] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const cancelledRef = useRef(false);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    abortControllerRef.current?.abort();
    readerRef.current?.cancel().catch(() => undefined);
    setIsStreaming(false);
    setNodeStatus(null);
  }, []);

  const reset = useCallback(() => {
    cancel();
    setStreamingContent("");
    setStreamError(null);
    setFinalMetadata(null);
    setWritebackPending(null);
    setWasCancelled(false);
    setToolCallStatus(null);
    setNodeStatus(null);
    setNodeTrace([]);
    setThinkingContent("");
    setIsThinking(false);
  }, [cancel]);

  const sendMessage = useCallback(
    async (params: StreamingChatParams): Promise<void> => {
      // Cancel any in-flight request first
      abortControllerRef.current?.abort();
      cancelledRef.current = false;

      // Reset streaming state
      setStreamingContent("");
      setIsStreaming(true);
      setStreamError(null);
      setToolCallStatus(null);
      setNodeStatus(null);
      setNodeTrace([]);
      setFinalMetadata(null);
      setWritebackPending(null);
      setWasCancelled(false);
      setThinkingContent("");
      setIsThinking(false);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const apiBase = process.env.NEXT_PUBLIC_API_BASE || "/api/proxy";

      try {
        const response = await fetch(`${apiBase}/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: params.message,
            user_id: params.userId,
            conversation_id: params.conversationId,
            attachment_ids: params.attachmentIds,
            document_ids: params.documentIds,
            rag_enabled: params.ragEnabled,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        if (!response.body) {
          throw new Error("Response body is null");
        }

        const reader = response.body.getReader();
        readerRef.current = reader;
        const decoder = new TextDecoder(undefined, { fatal: false });
        let buffer = "";

        // ── SSE read loop ──────────────────────────────────────────────
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // SSE events are delimited by double newline
          const parts = buffer.split("\n\n");
          buffer = parts.pop() ?? "";

          // Accumulate token/thinking deltas across all events in this chunk so we
          // call setState once per network read rather than once per token event.
          // Each setStreamingContent call schedules a React re-render + full markdown
          // re-parse; batching per-chunk keeps the main thread free between reads.
          let tokenDelta = "";
          let thinkingDelta = "";
          let streamEnded = false;

          for (const part of parts) {
            const trimmed = part.trim();
            if (!trimmed) continue;

            // Parse event type and data from the SSE block
            let eventType = "message";
            let dataLine = "";
            for (const line of trimmed.split("\n")) {
              if (line.startsWith("event: ")) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith("data: ")) {
                dataLine = line.slice(6).trim();
              }
            }

            if (dataLine === "[DONE]") {
              streamEnded = true;
              break;
            }

            if (!dataLine) continue;

            let parsed: Record<string, unknown>;
            try {
              parsed = JSON.parse(dataLine) as Record<string, unknown>;
            } catch {
              continue;
            }

            switch (eventType) {
              case "token":
                tokenDelta += (parsed.delta as string) ?? "";
                break;

              case "tool_call":
                setToolCallStatus({
                  name: parsed.name as string,
                  status: parsed.status as "start" | "end",
                  label: (parsed.label as string) || (parsed.name as string),
                });
                break;

              case "node":
                setNodeStatus((parsed.label as string) ?? null);
                break;

              case "node_state":
                setNodeTrace((prev) => [
                  ...prev,
                  {
                    node: parsed.node as string,
                    sequence: parsed.sequence as number,
                    durationMs: (parsed.duration_ms as number | null) ?? null,
                    status: (parsed.status as "success" | "error") ?? "success",
                  },
                ]);
                break;

              case "metadata":
                setFinalMetadata(buildMetadataFromSSE(parsed));
                setNodeStatus(null);
                break;

              case "writeback_pending":
                setWritebackPending({
                  documentId: parsed.document_id as string,
                  filename: parsed.filename as string,
                });
                break;

              case "writeback":
                // Merge even if `metadata` hasn't arrived yet — the writeback result
                // must never be dropped. Seed a minimal metadata object if needed.
                setFinalMetadata((prev) => ({
                  ...(prev ?? { tokens_used: 0, tools_executed: [] }),
                  workspace_document_writeback:
                    parsed as ChatResponse["metadata"]["workspace_document_writeback"],
                }));
                break;

              case "thinking_start":
                setIsThinking(true);
                break;

              case "thinking":
                thinkingDelta += (parsed.delta as string) ?? "";
                break;

              case "thinking_end":
                setIsThinking(false);
                break;

              case "warning":
                setStreamWarning((parsed.message as string) ?? null);
                break;

              case "error":
                throw new Error((parsed.message as string) ?? "Unknown stream error");
            }
          }

          // Flush accumulated deltas once per chunk — a single setState per read()
          if (tokenDelta) setStreamingContent((prev) => prev + tokenDelta);
          if (thinkingDelta) setThinkingContent((prev) => prev + thinkingDelta);

          if (streamEnded) {
            setIsStreaming(false);
            setNodeStatus(null);
            return;
          }
        }
      } catch (err) {
        if (cancelledRef.current || (err instanceof DOMException && err.name === "AbortError")) {
          setWasCancelled(true);
        } else {
          setStreamError(err instanceof Error ? err.message : "Unknown streaming error");
          setThinkingContent(""); // stream errored before any content was committed
        }
      } finally {
        if (cancelledRef.current) setWasCancelled(true);
        setIsStreaming(false);
        setNodeStatus(null);
        setToolCallStatus(null);
        setIsThinking(false);
        readerRef.current = null;
        abortControllerRef.current = null;
      }
    },
    [],
  );

  return {
    streamingContent,
    isStreaming,
    streamError,
    streamWarning,
    toolCallStatus,
    nodeStatus,
    nodeTrace,
    finalMetadata,
    writebackPending,
    wasCancelled,
    thinkingContent,
    isThinking,
    sendMessage,
    cancel,
    reset,
  };
}
