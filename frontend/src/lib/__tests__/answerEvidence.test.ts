import { deriveAnswerEvidence, EMPTY_ANSWER_EVIDENCE } from "@/lib/answerEvidence";
import type { MessageMetrics } from "@/lib/types";

describe("deriveAnswerEvidence", () => {
  test("returns empty evidence for null/undefined", () => {
    expect(deriveAnswerEvidence(null)).toEqual(EMPTY_ANSWER_EVIDENCE);
    expect(deriveAnswerEvidence(undefined).hasEvidence).toBe(false);
  });

  test("excludes the knowledge_base pseudo-tool but flags knowledgeBaseUsed", () => {
    const metrics: MessageMetrics = {
      tools_executed: [
        { name: "web_search_mcp", status: "success" },
        { name: "knowledge_base", status: "success" },
      ],
    };
    const e = deriveAnswerEvidence(metrics);
    expect(e.tools.map((t) => t.name)).toEqual(["web_search_mcp"]);
    expect(e.toolCount).toBe(1);
    expect(e.knowledgeBaseUsed).toBe(true);
    expect(e.hasEvidence).toBe(true);
  });

  test("counts RAG sources only when documents were retrieved", () => {
    const base: MessageMetrics = {
      sources: [
        { type: "web", label: "example.com", url: "https://example.com" },
        { type: "rag", label: "doc-1", url: null },
      ],
    };
    expect(deriveAnswerEvidence({ ...base, rag_doc_count: 0 }).sourceCount).toBe(1); // web only
    const withDocs = deriveAnswerEvidence({ ...base, rag_doc_count: 2 });
    expect(withDocs.sourceCount).toBe(2); // web + rag
    expect(withDocs.ragSources).toHaveLength(1);
  });

  test("derives memory and document counts and a map flag", () => {
    const metrics: MessageMetrics = {
      memories_used: [{ memory_id: "1", text: "x" }],
      documents_used: [{ id: "d1", filename: "a.md", version: 1 }],
      map_data: { type: "location", zoom: 10, osm_url: "https://osm" },
    };
    const e = deriveAnswerEvidence(metrics);
    expect(e.memoryCount).toBe(1);
    expect(e.documentCount).toBe(1);
    expect(e.hasEvidence).toBe(true);
    expect(e.mapData).toBeDefined();
  });
});
