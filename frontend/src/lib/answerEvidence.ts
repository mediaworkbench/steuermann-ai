import type { MessageMetrics, ToolExecution, ToolResultDetail, MemoryReference, Source, MapData } from "./types";

/**
 * Normalized, render-ready evidence for a single assistant answer. Derived once
 * from a `MessageMetrics` and shared by the in-stream chips, the workspace
 * evidence tabs, and MetricsPanel so the same data is never derived three ways.
 *
 * Conventions baked in here (so every surface agrees):
 *  - `tools` excludes the synthetic `knowledge_base` pseudo-tool — RAG is
 *    represented by `knowledgeBaseUsed` / `sources`, not as a tool.
 *  - `sources` are the *shown* sources: web sources always, RAG sources only
 *    when documents were actually injected (mirrors the chat source badges).
 */
export interface AnswerEvidence {
  tools: ToolExecution[];
  toolCount: number;
  /** Per-tool args + result preview (Outputs tab). Excludes `knowledge_base`. */
  toolResults: ToolResultDetail[];
  memories: MemoryReference[];
  memoryCount: number;
  knowledgeBaseUsed: boolean;
  ragAttempted: boolean;
  ragDocCount: number;
  sources: Source[];
  webSources: Source[];
  ragSources: Source[];
  sourceCount: number;
  documents: Array<{ id: string; filename: string; version: number }>;
  documentCount: number;
  attachments: Array<{ id: string; original_name: string }>;
  mapData?: MapData;
  hasEvidence: boolean;
}

export const EMPTY_ANSWER_EVIDENCE: AnswerEvidence = {
  tools: [],
  toolCount: 0,
  toolResults: [],
  memories: [],
  memoryCount: 0,
  knowledgeBaseUsed: false,
  ragAttempted: false,
  ragDocCount: 0,
  sources: [],
  webSources: [],
  ragSources: [],
  sourceCount: 0,
  documents: [],
  documentCount: 0,
  attachments: [],
  mapData: undefined,
  hasEvidence: false,
};

export function deriveAnswerEvidence(metrics: MessageMetrics | null | undefined): AnswerEvidence {
  if (!metrics) return EMPTY_ANSWER_EVIDENCE;

  const rawTools = metrics.tools_executed ?? [];
  const tools = rawTools.filter((tool) => tool.name !== "knowledge_base");
  const toolResults = (metrics.tool_results_detail ?? []).filter(
    (tr) => tr.name !== "knowledge_base",
  );

  const memories = metrics.memories_used ?? [];
  const ragAttempted = Boolean(metrics.rag_attempted);
  const ragDocCount = metrics.rag_doc_count ?? 0;
  const knowledgeBaseUsed = ragAttempted || rawTools.some((tool) => tool.name === "knowledge_base");

  const rawSources = metrics.sources ?? [];
  const webSources = rawSources.filter((s) => s.type === "web");
  // RAG sources are only meaningful when documents were actually retrieved.
  const ragSources = ragDocCount > 0 ? rawSources.filter((s) => s.type === "rag") : [];
  const sources = [...webSources, ...ragSources];

  const documents = metrics.documents_used ?? [];
  const attachments = metrics.attachments_used ?? [];
  const mapData = metrics.map_data;

  const hasEvidence =
    tools.length > 0 ||
    memories.length > 0 ||
    sources.length > 0 ||
    documents.length > 0 ||
    knowledgeBaseUsed ||
    Boolean(mapData);

  return {
    tools,
    toolCount: tools.length,
    toolResults,
    memories,
    memoryCount: memories.length,
    knowledgeBaseUsed,
    ragAttempted,
    ragDocCount,
    sources,
    webSources,
    ragSources,
    sourceCount: sources.length,
    documents,
    documentCount: documents.length,
    attachments,
    mapData,
    hasEvidence,
  };
}
