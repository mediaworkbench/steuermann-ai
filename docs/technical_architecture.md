# Steuermann - Technical Architecture

## **1. Summary**

This document defines the technical architecture for the Steuermann - a domain-agnostic, on-premise agentic AI system designed for internal deployment across multiple specialized domains.

**Core Principles:**

- LangGraph orchestration (containerized, independent service)
- FastAPI adapter layer (settings management, metrics exposure, auth)
- Modern frontend (Next.js + React production stack)
- Profile-based customization via declarative overlays
- Unified technology stack (PostgreSQL, Qdrant, Prometheus, external LLM providers)
- Language-specific optimization per profile
- Separate but integrated RAG/ingestion pipeline
- Clear separation of concerns (orchestration, adaptation, presentation)

**Target Deployment:**

- On-premise Docker environments
- Small to medium internal teams
- Regulated domains (medical, financial, operational)

---

## **2. System Architecture Overview**

### **2.1 High-Level Component Diagram (Production)**

```text
┌────────────────────────────────────────────────────────────────────┐
│                           User Layer                               │
│  ┌────────────────────┐                                               │
│  │  Next.js UI        │                                               │
│  │  (Port 3000)       │                                               │
│  │  ├─ Chat           │                                               │
│  │  ├─ Settings       │                                               │
│  │  ├─ Metrics        │                                               │
│  │  └─ Dashboards     │                                               │
│  └────────┬───────────┘                                               │
└───────────┼────────────────────────────────────────────────────────┘
            │
┌───────────┼────────────────────────────────────────────────────────┐
│           │     Application Layer                                   │
│  ┌────────▼──────────┐             ┌──────────────────────┐   │
│  │  FastAPI Adapter  │             │  Prometheus Scraper  │   │
│  │  (Port 8001)      │             │  (Port 9090)         │   │
│  │  ├─ /api/chat     │             └──────────────────────┘   │
│  │  ├─ /api/settings │                                             │
│  │  ├─ /api/metrics  │─────────────┘                             │
│  │  ├─ /api/analytics│                                           │
│  │  ├─ /api/models   │                                           │
│  │  └─ /api/system-config │  supported_languages, profile UI     │
│  └────────┬──────────┘                                           │
│           │                                                      │
│  ┌────────▼──────────────┐                                       │
│  │ LangGraph Orchestrator │◄───────────► Prometheus metrics      │
│  │ (Port 8000, internal)  │              http://langgraph:8000   │
│  │ ├─ Graph execution     │              /metrics                │
│  │ ├─ Tool routing        │                                      │
│  │ ├─ Memory management   │                                      │
│  │ └─ RAG retrieval       │                                      │
│  └────────┬────────────────┘                                      │
│           │                                                      │
│  ┌────────▼──────────────┐                                       │
│  │  Ingestion Service    │  (Optional Module)                    │
│  │  (On-demand or sched) │                                       │
│  └────────┬────────────────┘                                      │
└───────────┼──────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────┐
│           │         Data Layer                                    │
│  ┌────────▼──────────┐    ┌────────────────┐  ┌──────────────┐  │
│  │  PostgreSQL       │    │    Qdrant      │  │  MCP Tools   │  │
│  │  - Checkpoints    │    │  - Embeddings  │  │  - web_search│  │
│  │  - Users/Roles    │    │  - Memory      │  │  - extract   │  │
│  │  - Sessions       │    │  - RAG chunks  │  │  - datetime  │  │
│  │  - Audit Logs     │    └────────────────┘  └──────────────┘  │
│  └───────────────────┘                                           │
└───────────────────────────────────────────────────────────────────┘
            │
┌───────────┼──────────────────────────────────────────────────────┐
│           │         External Provider Endpoints                    │
│  ┌────────▼───────────────────────────────┐                       │
│  │  LLM Providers (configured per profile) │                       │
│  │  - LM Studio / Ollama / OpenRouter      │                       │
│  │  - Any OpenAI-compatible API endpoint    │                       │
│  └─────────────────────────────────────────┘                       │
└───────────────────────────────────────────────────────────────────┘
```

### **2.2 Technology Stack Summary**

- Orchestration: LangGraph (`>=1.1.0`) - Workflow engine (containerized)
- Multi-Agent: CrewAI (`>=1.14.0`) - Collaborative reasoning
- LLM Integration: LangChain + langchain-litellm (`>=1.2.0 / 0.6.4`) - Unified LLM interface via LiteLLM router
- Adapter: FastAPI (`>=0.135.0`) - Settings/auth/metrics API
- Frontend: Next.js + React (`>=16.0 + React 19`) - Modern chat and settings UI
- Vector Store: Qdrant (`>=1.7.0`) - RAG embeddings and Mem0 internal vector storage
- Database: PostgreSQL (`>=15.0`) - Structured data, checkpoints, users/roles
- Memory: Mem0 OSS embedded (`>=2.0.1`) - Memory abstraction with embedded Qdrant-backed storage
- Embeddings: Remote provider abstraction (Current) - Config-driven embeddings via remote OpenAI-compatible endpoint
- Monitoring: Prometheus (Latest) - Metrics collection and alerting
- LLM Providers: External provider endpoints (Latest) - Configured via profile-owned provider registry

---

## **3. Repository & Profile Model**

### **3.1 Repository Shape**

This repository is the shared template codebase. Domain behavior is added through profile overlays, prompt overrides, plugins, and deployment-specific environment settings.

### **3.2 Versioning**

- The package metadata currently reports version `0.3.0` in `pyproject.toml`.
- Public release positioning is still experimental beta.
- Treat profile overlays as configuration compatibility surfaces that should be validated against the exact repository revision you deploy.

### **3.3 Profile Overlay Structure**

```text
steuermann-ai/
├── pyproject.toml
├── config/
│   ├── core.yaml
│   ├── agents.yaml
│   ├── features.yaml
│   ├── tools.yaml
│   ├── prompts/
│   └── profiles/
│       └── <profile_id>/
│           ├── profile.yaml
│           ├── core.yaml
│           ├── agents.yaml
│           ├── features.yaml
│           ├── tools.yaml
│           ├── ui.yaml
│           └── prompts/
├── frontend/
├── backend/
├── monitoring/
├── tests/
└── universal_agentic_framework/
```

**Key principles:**

- Prefer profile overlays over direct edits to `universal_agentic_framework/orchestration/`.
- Keep tool descriptions and `tool_routing` configuration aligned.
- Keep prompt overrides close to the active profile instead of branching framework code.
- Validate profile behavior against the current repository revision after upgrades.

---

## **4. LangGraph Execution Architecture**

**Status note:** This section mixes implemented behavior with advanced or experimental extension paths. LangGraph orchestration, semantic tool routing, RAG retrieval, checkpointing, and frontend-backed operations are active in the current stack. Crew customization remains an advanced area and should not be treated as the default deployment path.

### **4.1 Deployment Mode**

**Container-Based Architecture** (production):

- LangGraph runs as standalone service (port 8000, internal)
- FastAPI adapter provides API layer (port 8001)
- Next.js frontend for user interface (port 3000)
- Services communicate via HTTP
- Independent scaling and deployment

### **4.2 Graph State Management**

**GraphState Schema** (defined in `universal_agentic_framework/orchestration/graph_builder.py`):

```python
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langgraph.checkpoint.base import UntrackedValue

class GraphState(TypedDict, total=False):

    # Conversation / input
    messages: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]               # Uploaded conversation attachments from adapter
    attachment_context: List[Dict[str, Any]]         # Prompt-ready normalized attachment snippets
    workspace_documents: List[Dict[str, Any]]        # User workspace documents resolved by adapter
    workspace_document_context: List[Dict[str, Any]] # Prompt-ready normalized workspace snippets
    workspace_writeback_requested: bool
    workspace_writeback_document: Optional[Dict[str, Any]]  # Full document for writeback (bypasses 600-token truncation)

    # Session identity
    user_id: str
    session_id: str          # Session identifier for co-occurrence tracking
    language: str
    fork_name: str           # Fork identifier for Prometheus metrics (legacy label key)
    profile_id: str          # Active deployment profile identifier
    user_settings: Dict[str, Any]           # tool_toggles, rag_config, preferred_model, theme, language
    llm_capability_probes: List[Dict[str, Any]]  # Adapter-provided probe snapshots per provider

    # Memory / RAG
    loaded_memory: List[Dict[str, Any]]
    digest_context: List[Dict[str, Any]]     # Digest subset from loaded memory retrieval
    memory_analytics: Dict[str, Any]         # Importance scores, related count
    knowledge_context: List[Dict[str, Any]]  # RAG retrieved documents
    rag_attempted: bool  # True if Qdrant was queried this turn (False on all skip paths)
    rag_doc_count: int   # Docs above score_threshold injected into prompt

    # Tools
    loaded_tools: Annotated[List[Any], UntrackedValue(list)]              # BaseTool instances — not checkpointed
    candidate_tools: Annotated[List[Dict[str, Any]], UntrackedValue(list)]  # Layer 1 output — not checkpointed
    tool_calling_mode: str          # "native" | "structured" | "react"
    tool_calling_mode_reason: str   # Why the mode was selected or downgraded
    prefilter_intents: Dict[str, Any]           # Intent detection results from Layer 1
    tool_results: Dict[str, str]                # Tool name → execution result
    tool_execution_results: Dict[str, Dict[str, Any]]  # Tool name → structured execution envelope
    routing_metadata: Dict[str, str]            # Tool name → reason for selection

    # Crews
    crew_results: Dict[str, Any]  # Multi-agent crew execution results

    # Tokens / metrics
    tokens_used: int
    turn_tokens_used: int    # Current invocation token usage for per-turn budgeting
    input_tokens: int        # Input token count (separate tracking)
    output_tokens: int       # Output token count (separate tracking)
    provider_used: str       # Actual provider used for response generation
    model_used: str          # Actual model used for response generation
    summary_text: str
    digest_chain: List[Dict[str, Any]]  # Rolling conversation digest metadata
    sources: List[Dict[str, Any]]       # [{type: "web"|"rag", label: str, url: str|None}]
    query_embedding: List[float]        # Precomputed user-message embedding (reused by RAG)
```

### **4.3 Persistence Layer**

**Checkpoint Storage:**

- Backend: always PostgreSQL (`PostgresSaver`) — SQLite support and the `enabled` flag have been removed
- `CHECKPOINTER_POSTGRES_DSN` environment variable (or `checkpointing.postgres_dsn` in `config/core.yaml`) required at startup; raises `ValueError` if missing
- Load-at-edge pattern: both `/invoke` and `/stream` pre-fetch accumulated messages from the checkpoint via `aget_tuple()` and merge them with each new user message before graph invocation
- Per-session continuity via `configurable.thread_id = session_id`; ephemeral requests (no `conversation_id`) omit `thread_id` so no checkpoint rows are written
- Startup pruning + periodic fire-and-forget pruning every 100 invocations keep the checkpoint table flat
- `GraphState.loaded_tools` and `candidate_tools` use `Annotated[..., UntrackedValue(list)]` to exclude non-serializable `BaseTool` instances from checkpoint writes

### **4.4 Graph Execution Flow (Three-Tier Tool Selection)**

```text
User Request (Next.js)
  │
  ├──> FastAPI Adapter
  │
  └──> LangGraph Service
  │
  ├──> Load checkpoint (if exists)
  ├──> Initialize GraphState
  ├──> Execute graph nodes
  │    │
  │    ├──> [_route_start] START conditional edge
  │    │    ├──> crews (direct dispatch — bypass tool nodes entirely)
  │    │    └──> load_tools (normal path)
  │    │
  │    ├──> load_tools
  │    │    └──> Load all available tools from registry
  │    │
  │    ├──> prefilter_tools (LAYER 1 — Semantic Pre-filter)
  │    │    ├──> Embed user query (result stored as query_embedding — reused by RAG)
  │    │    ├──> Score query against tool descriptions (cosine similarity)
  │    │    ├──> Apply intent boosting (+0.2 for datetime, calc, URL, analyze_image,
  │    │    │     ocr, analyze_document, analyze_chart, read_barcodes patterns)
  │    │    ├──> Apply gates: min_top_score (0.7), min_spread (0.10)
  │    │    ├──> Filter by similarity_threshold (0.55), top-K (5)
  │    │    └──> Store candidate_tools in state (NO execution)
  │    │
  │    ├──> [route_tool_strategy] (conditional edge)
  │    │    ├──> call_tools_native     (LAYER 2 — bind_tools, LLM decides)
  │    │    ├──> call_tools_structured (LAYER 2 — JSON schema in prompt)
  │    │    ├──> call_tools_react      (LAYER 2 — ReAct loop)
  │    │    └──> no_tools              (0 candidates → skip Layer 2)
  │    │    Each Layer 2 node includes LAYER 3 — schema validation + retry
  │    │
  │    ├──> after_tool_call (convergence point — all Layer 2 paths merge here)
  │    │
  │    ├──> [route_to_crew] (conditional edge)
  │    │    ├──> research_crew / analytics_crew / code_generation_crew / planning_crew
  │    │    ├──> crew_chain / crew_parallel
  │    │    └──> memory_query_cache (no crew → continue normal flow)
  │    │
  │    ├──> memory_query_cache  (Redis cache probe for memory)
  │    ├──> load_memory
  │    ├──> retrieve_knowledge  (RAG, score_threshold: 0.6)
  │    │    Three skip layers before any Qdrant call:
  │    │      1. System: feature flag + rag.enabled config
  │    │      2. User: rag_config.enabled per-session toggle
  │    │      3. Intent: skip_rag heuristic (greeting/math/datetime/tool-meta)
  │    │    rag_attempted=True / rag_doc_count=N set only when Qdrant is queried
  │    │
  │    ├──> memory_cache_store
  │    │
  │    ├──> respond
  │    │    ├──> Inject tool_results into context
  │    │    ├──> Inject knowledge_context from RAG
  │    │    └──> Invoke LLM (chat role)
  │    │
  │    ├──> compress_conversation  (rolling digest, bounded history)
  │    ├──> summarize
  │    ├──> update_memory
  │    └──> cache_stats
  │
  └──> Return response
        │
        └──> FastAPI Adapter
              │
              └──> Next.js (render to user)
```

#### Key Design: Three-Tier Tool Selection

The framework uses a three-tier architecture combining semantic scoring with model-driven decisions:

- **Layer 1** (always runs): Semantic pre-filter scores all tools, applies gates, returns top-K candidates. Model-agnostic — works with any embedding model.
- **Layer 2** (model-driven): The LLM receives candidate tools and decides which to call. Mode is configurable per provider: `native` (bind_tools), `structured` (JSON in prompt), or `react` (Thought→Action→Observation loop).
- **Layer 3** (validation + deterministic safety): Tool call arguments validated against schema. Parse failures trigger re-prompt with error feedback, up to `max_retries` attempts. Native mode also applies deterministic URL safety fallbacks for webpage extraction when model calls are malformed or omitted.

Benefits:

- ✅ Works with models that support tool_calling (GPT-4, LFM2, etc.) via native mode
- ✅ Works with models WITHOUT tool_calling (Gemma, Llama, etc.) via structured/react mode
- ✅ Semantic pre-filter reduces token cost (only relevant tools sent to model)
- ✅ Min-top-score gate skips Layer 2 entirely for conversational queries
- ✅ Deterministic extraction fallback prevents URL prompts from failing when native tool calls are missing/malformed
- ✅ No model-specific code — mode selection is configuration-driven

#### **4.4.1 Tool-Calling Mode Enforcement**

Mode validation and enforcement is implemented at all tool routing layers.

The framework validates that the resolved tool-calling mode matches the invoked Layer 2 node, ensuring consistency across the pipeline:

**Flow:**

1. **Prefilter Node** (`node_prefilter_tools`, Layer 1)
   - Resolves `tool_calling_mode` and `tool_calling_mode_reason` from probe data
   - Downgrades native → structured if probe shows capability_mismatch or error status
   - Stores mode and reason in GraphState

2. **Route Decision** (`route_tool_strategy`, conditional edge)
   - Routes to appropriate Layer 2 node based on `tool_calling_mode`
   - Logs routing decision with mode, reason, and candidate tool names
   - Fallback to structured mode if invalid mode detected

3. **Layer 2 Node Validation** (native/structured/react nodes)
   - Each node validates that `tool_calling_mode` matches expected invocation
   - `_validate_and_log_tool_calling_mode()` function checks mode consistency
   - Non-blocking enforcement: logs warnings on mismatch but proceeds gracefully
   - Ensures audit trail for debugging mode decisions

**Implementation Details:**

```python
# Mode validation at Layer 2 invocation
def _validate_and_log_tool_calling_mode(
    state: GraphState, 
    expected_mode: str, 
    node_name: str,
    fork_name: str
) -> Tuple[bool, str]:
    """Validate tool-calling mode matches expectations at invocation.
    
    Returns (is_valid, reason_string) tuple for audit logging.
    """
    actual_mode = state.get("tool_calling_mode", "structured")
    candidates = state.get("candidate_tools", [])
    
    # Skip validation if no candidates (mode irrelevant)
    if not candidates:
        return True, "no_candidates"
    
    is_valid = actual_mode == expected_mode
    reason = f"{actual_mode}_mode_valid" if is_valid else \
             f"mode_mismatch_expected_{expected_mode}_got_{actual_mode}"
    
    log_level = "info" if is_valid else "warning"
    logger.log(log_level, 
        f"Tool calling mode validation ({node_name})",
        expected_mode=expected_mode, actual_mode=actual_mode,
        is_valid=is_valid, candidates=len(candidates))
    
    return is_valid, reason
```

**Probe Lookup — Exact Match Required:**

`resolve_effective_tool_calling_mode()` requires an exact `(provider_id, model_name)` match from the probe results table. If probe data exists for the provider but not for the specific model, it falls back to `structured` rather than silently using a different model's probe results.

**Downgrade Logic:**

Mode is downgraded from native → structured when:

- `capability_mismatch` = true (probe detected `bind_tools()` not supported)
- `status` ∈ {"warning", "error"} (probe execution failed or found issues)
- `supports_bind_tools` = false (model doesn't support native tool binding)
- No probe data available for the exact `(provider_id, model_name)` pair

Downgrade reason: `probe_capability_mismatch_downgrade`

**Reason Tracking:**

Reason field propagates through entire pipeline:

- `configured_native_probe_ok` — probe shows capability OK, native mode safe
- `configured_non_native_mode` — provider configured for structured/react
- `configured_native_no_probe` — native configured but no probe data (risky, proceed cautiously)
- `probe_capability_mismatch_downgrade` — probe detected mismatch, downgraded to structured
- `configured_native_probe_provider_not_found` — provider not found in probe results
- `probe_model_not_found_forced_structured` — provider found in probe results but not the specific model

**Curated Reprobe:**

`LLMCapabilityProbeRunner.reprobe_for_model(model_name)` reprobes only the providers whose configured model matches the given name. The settings model-change trigger calls this instead of a full `run()`, avoiding unnecessary probes of unrelated models.

**Benefits:**

- ✅ Prevents mode mismatches at invocation (consistency across layers)
- ✅ Non-blocking validation (graceful degradation, never breaks chat flow)
- ✅ Comprehensive audit trail (reason field aids debugging)
- ✅ Automatic downgrade (safer than failing on unsupported models)
- ✅ Works across all provider types (LM Studio, Ollama, OpenAI-compatible)

### **4.5 Response Handling**

**Streaming path (primary):** `POST /api/chat/stream` returns `text/event-stream`. LangGraph exposes a matching `POST /stream` endpoint using `GRAPH.astream_events(version="v2")`. SSE event types:

| Event | Payload | Description |
| --- | --- | --- |
| `token` | `{"delta": "..."}` | LLM output chunk |
| `thinking_start` | `{}` | Reasoning chain begins (model emitting chain-of-thought) |
| `thinking` | `{"delta": "..."}` | Reasoning token chunk |
| `thinking_end` | `{}` | Reasoning chain complete |
| `tool_call` | `{"name": "...", "status": "start"/"end"}` | Tool invocation boundary |
| `node` | `{"node": "...", "label": "..."}` | Graph node status (RAG, memory) |
| `metadata` | `{tokens_used, model_used, sources, thinking_content, ...}` | Final response metadata |
| `error` | `{"message": "..."}` | Error during generation |
| `[DONE]` | _(final data line)_ | Stream complete |

Reasoning tokens are intercepted in `server.py` before they reach the `token` event via two complementary paths: (1) a tag-parser state machine strips `<think>`, `<thinking>`, and `<reflection>` blocks from `chunk.content` and emits them as `thinking_*` events; (2) when `chunk.additional_kwargs["reasoning_content"]` is present (LM Studio / LiteLLM native reasoning field), tokens are emitted directly without tag parsing. Accumulated thinking text is included in the `metadata` event as `thinking_content` and persisted to the message's JSONB metadata column.

Workspace writeback requests (`save`/`rewrite` intent on open documents) are transparently routed to the synchronous `POST /api/chat` path because writeback requires guaranteed delivery. The Next.js proxy detects `text/event-stream` content-type and pipes the response body directly (no `arrayBuffer()` buffering). The frontend `useStreamingChat` hook reads `response.body` via `getReader()`, parses SSE blocks, and exposes `streamingContent`, `isStreaming`, `thinkingContent`, `isThinking`, `nodeStatus`, `toolCallStatus`, `finalMetadata`, `wasCancelled`, `cancel()`.

**Synchronous path (fallback / writeback):** `POST /api/chat` blocks until the full LangGraph result is available and returns a single JSON `ChatResponse`.

### **4.6 Async Execution Reliability**

Runtime helper paths for performance/cache and crew result caching execute async operations even when an event loop is already active.

- Previous behavior in these helpers could skip operations in loop-active contexts.
- Current behavior executes coroutines in worker-thread event loops when required.
- Result: cache/compression and crew-cache paths are no longer silent no-ops under async server execution contexts.

---

## **5. Multi-Agent Layer: CrewAI Integration**

**Status:** Fully implemented (4 crews: Research, Analytics, Code Gen, Planning). Disabled by default via `multi_agent_crews: false` in `config/features.yaml`.

CrewAI crews are wrapped as LangGraph nodes — crews are workers invoked by graph nodes, not orchestrators. Each crew returns structured output merged back into `GraphState["crew_results"]`.

**Process types:** Sequential (default, predictable), Hierarchical (manager delegates), Consensus (multi-agent voting).

**Configuration:** `config/agents.yaml` defines crew agents, roles, tools, and process types.

→ **Full guide:** [CrewAI Extension Guide](crewai_extension_guide.md)

---

## **6. Memory Architecture**

### **6.1 Three-Tier Memory Model**

```text
┌─────────────────────────────────────────────────────────┐
│  Tier 1: Operational State (GraphState)                 │
│  - Ephemeral, session-scoped                            │
│  - Current messages, node outputs                       │
│  - Checkpointed for resumability only                   │
└─────────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────────┐
│  Tier 2: Semantic Long-Term Memory (Mem0 OSS)           │
│  - User facts, preferences, summaries                   │
│  - Vector similarity search via Qdrant-backed Mem0      │
│  - Loaded at session start, updated explicitly          │
└─────────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────────┐
│  Tier 3: Artifacts & Knowledge (PostgreSQL + Qdrant)    │
│  - Documents, reports, decisions                        │
│  - Domain knowledge collections                         │
│  - Retrieved deterministically or via search            │
└─────────────────────────────────────────────────────────┘
```

### **6.2 Memory Operations as Graph Nodes**

Memory is loaded explicitly at graph start and written only by dedicated memory nodes — never automatic. The `load_memory_node` queries via the `Mem0MemoryBackend` adapter for user facts/preferences/history. The `update_memory_node` distills new conversation into long-term memory via LLM summarization and persists via Mem0.

Mem0 extraction behavior is profile-configurable through `memory.mem0.infer_enabled`. When enabled, Mem0 performs extraction/deduplication before persistence; when disabled, the pipeline stores summary text via verbatim fallback. Extraction model selection follows `llm.roles.auxiliary`.

**LLM role usage summary:** `llm.roles.chat` drives primary generation. `llm.roles.auxiliary` is used by `node_summarize`, Mem0 extraction, RAG query rewriting, and workspace intent classification. `llm.roles.vision` is invoked exclusively by `analyze_image_tool` — the chat model calls the tool explicitly when it needs to analyze an uploaded image or image URL; the tool makes a direct httpx POST to the vision provider's `/chat/completions` endpoint using the OpenAI vision message format. `llm.roles.embedding` drives semantic routing and RAG retrieval.

Mem0 API contract note: the adapter is aligned to Mem0's filters-based API shape (v3+). Entity scoping for memory search/list/delete is expected via `filters={"user_id": ...}` rather than legacy top-level entity kwargs. Direct item operations follow the canonical OSS SDK signatures: `get(memory_id)`, `delete(memory_id)`, and `update(memory_id, data=..., metadata=...)`.

Compression path includes rolling digest metadata on summary messages (`digest_id`, `previous_digest_id`, message counts) so older context can be chained across turns while retaining recent raw messages.

### **6.3 Context Priority Rules (Response Assembly)**

At response generation time, context is ordered by reliability:

- **Primary**: current-turn `tool_results` (fresh tool execution output)
- **Secondary**: `knowledge_context` from RAG
- **Tertiary**: `loaded_memory` as background (`PAST CONTEXT (LOW PRIORITY)` when web tools are used)

If prior memory conflicts with current-turn tool outputs, tool outputs take precedence. This prevents stale conversational memory from overriding fresh extraction/search results while preserving useful continuity.

### 6.4 Qdrant Collection Strategy

**Naming convention:** `{profile_id}_{type}_{language}` (e.g., `medical-ai-de_memory_de`)

Advanced memory behavior is integrated into this architecture: semantic similarity retrieval, recency-aware ranking, frequency-aware prioritization, and cross-session linking through metadata and retrieval patterns.

### 6.5 Memory Ownership And Durability (Concise)

- Short-memory digest chain source of truth: `GraphState.digest_context` (session-scoped, bounded).
- Long-memory source of truth: Mem0 OSS with Qdrant-backed storage, accessed via `Mem0MemoryBackend`.
- Co-occurrence links durable source: PostgreSQL `co_occurrence_edges` (user-partitioned, indexed by user+memory and user+updated_at).
- Co-occurrence retrieval path: durable-first reads with in-memory fallback, plus bounded fanout and periodic prune maintenance.

Digest flow used by memory updates:

1. Compression/summarization creates digest metadata.
2. `node_update_memory` passes digest chain to `update_memory_node`.
3. `Mem0MemoryBackend.upsert` stores digest metadata with memory records.
4. `load_memory_node` can surface digest context on later retrieval.

Critical verification expectation:

- End-to-end digest metadata persistence must be validated in tests.
- Co-occurrence persistence, restart durability, and index/schema checks are validated in tests.

---

## **7. LLM Integration Layer**

### **7.1 LLM Provider Abstraction**

All LLM access goes through `LLMFactory` using `langchain-litellm`. A single `ChatLiteLLM` instance handles any provider (LM Studio, OpenAI-compatible) via LiteLLM's unified model string format (`openai/model`). Multi-model routing with automatic retries uses `ChatLiteLLMRouter` wrapping `litellm.Router`.

```python
from langchain_litellm import ChatLiteLLM, ChatLiteLLMRouter
from litellm import Router

# Single model — used by get_model()
# IMPORTANT: api_base and api_key must be top-level kwargs, NOT inside model_kwargs
def build_litellm_chat(provider_config, model_name: str) -> ChatLiteLLM:
    chat_kwargs = {
        "model": model_name,          # e.g. "openai/liquid/lfm2-24b-a2b"
        "temperature": provider_config.temperature,
        "max_tokens": provider_config.max_tokens,
    }
    if provider_config.api_base:
        chat_kwargs["api_base"] = str(provider_config.api_base)
    if provider_config.api_key:
        chat_kwargs["api_key"] = provider_config.api_key
    return ChatLiteLLM(**chat_kwargs)

# Router — used by get_router_model() in graph_builder.py
def build_router(primary_model: str, fallback_model: str | None) -> ChatLiteLLMRouter:
    model_list = [
        {"model_name": "primary", "litellm_params": {"model": primary_model}},
    ]
    if fallback_model:
        model_list.append(
            {"model_name": "fallback", "litellm_params": {"model": fallback_model}}
        )
    router = Router(model_list=model_list, num_retries=3, retry_after=1)
    return ChatLiteLLMRouter(router=router)
```

**Legacy providers removed**: `langchain-openai` and `langchain-ollama` are no longer dependencies. Runtime provider and model configuration is owned by the active profile overlay in `config/profiles/<profile_id>/core.yaml`.

### **7.2 Language-Specific Model Selection**

```yaml
# config/profiles/starter/core.yaml
llm:
  providers:
    lmstudio:
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      api_key: "lm-studio"
      models:
        de: "openai/liquid/lfm2-24b-a2b"
        en: "openai/liquid/lfm2-24b-a2b"
      temperature: 0.7
      max_tokens: 4096

    openrouter:
      api_base: "${LLM_PROVIDERS_OPENROUTER_API_BASE}"
      api_key: "${OPENROUTER_API_KEY}"
      models:
        de: "openrouter/openai/gpt-4o-mini"
        en: "openrouter/openai/gpt-4o-mini"
      temperature: 0.3

  roles:
    chat:
      providers:
        - provider_id: lmstudio
        - provider_id: openrouter
    embedding:
      providers:
        - provider_id: lmstudio
      config_only: true
    vision:
      providers:
        - provider_id: lmstudio
      config_only: true
    auxiliary:
      providers:
        - provider_id: lmstudio
      config_only: true

  router:
    routing_strategy: simple-shuffle
    num_retries: 3
    retry_after: 1
```

**LM Studio model ID convention:** LM Studio's `/v1/models` returns raw IDs like `liquid/lfm2-24b-a2b`. LiteLLM requires the `openai/` provider prefix. Always configure models as `openai/<lm-studio-raw-id>` (e.g. `openai/liquid/lfm2-24b-a2b`). Bare or wrongly-prefixed IDs will cause `BadRequestError: LLM Provider NOT provided`.

**Runtime Selection** — `LLMFactory.get_router_model(language)` builds a `ChatLiteLLMRouter` from the `chat` role provider chain and forwards retry/routing policy from `llm.router`:

```python
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

def get_router_model(self, language: str) -> ChatLiteLLMRouter:
    model_list = []
    for index, ref in enumerate(self.config.llm.roles.chat.providers):
        provider = self.config.llm.get_provider(ref.provider_id)
        model_name = provider.models.get(language) or provider.models.get("en")
        label = "primary" if index == 0 else "fallback"
        model_list.append({"model_name": label, "litellm_params": {"model": model_name}})

    router = Router(
        model_list=model_list,
        routing_strategy=self.config.llm.router.routing_strategy,
        num_retries=self.config.llm.router.num_retries,
        retry_after=self.config.llm.router.retry_after,
    )
    return ChatLiteLLMRouter(router=router)
```

### **7.3 Token Tracking**

**Model-Aware Input Counting:**

`count_tokens_for_model(model_name, text)` in `universal_agentic_framework/llm/budget.py` uses `litellm.token_counter()` with tiktoken for model-specific token counting. It is called in `node_summarize` for pre-call node_tokens estimation. Falls back to `estimate_tokens()` (character-based approximation) if tiktoken raises an exception for the given model ID.

**Real LLM Usage Capture:**

Post-call token accounting uses the real LLM-reported `usage_metadata` (`input_tokens`, `output_tokens`) captured from the `on_chat_model_end` event in `server.py`. This event fires once per LLM call with the fully-merged `AIMessage`; it is the reliable source for real token counts. The captured values are forwarded to the frontend via the SSE `metadata` event as the numerator for the context ring indicator. A secondary capture path via `on_chat_model_stream` covers providers that embed usage in the final streaming chunk; `on_chat_model_end` values take precedence when non-zero.

If `usage_metadata` is absent, `_tokens_from_usage()` returns `(0, char/4 estimate)` for the output only; `state["input_tokens"]` accumulates `actual_input_tokens` (zero when unavailable) and serves as a fallback in the metadata payload.

**Observability Fields (no enforcement):**

`tokens_used`, `turn_tokens_used`, `input_tokens`, and `output_tokens` are retained in `GraphState` and exposed via Prometheus metrics and SSE metadata. The `tokens` configuration keys (`default_budget`, `per_turn_budget_ratio`, etc.) are still parsed from profile YAML but have no enforcement effect — no node raises `TokenBudgetExceeded` or discards responses based on token counts.

---

## **8. Three-Tier Tool Selection Architecture**

### **8.1 Design Philosophy**

The framework uses a **three-tier tool selection architecture** that combines semantic scoring (model-agnostic) with model-driven tool calling:

**Layer 1 — Semantic Pre-filter (always runs):**

1. Embed user query using configured embedding model
2. Score query against all tool descriptions (cosine similarity)
3. Apply intent boosting (+0.2 for detected datetime, calculation, URL, and image intents)
4. Apply gates: min_top_score (0.7) and min_spread (0.10)
5. Filter candidates by similarity_threshold (0.55) and top-K (5)
6. Store candidates in state — NO tools are executed at this layer

**Layer 2 — Model-Driven Tool Calling (configurable per provider):**

- `native`: Tools bound via `bind_tools()` — the LLM decides which to call (or none)
- `structured`: Tool JSON schemas injected into system prompt — LLM outputs JSON tool call. The system prompt footer is conditional: when `force_tool_use=True` (explicit web search intent from `prefilter_intents`, or top candidate score ≥ 0.75) the footer reads "You MUST call one of the available tools listed above. Do not respond with plain text — output ONLY the JSON tool call." Otherwise it reads "If no tool is needed, respond normally in plain text." When `force_tool_use=True` and the model responds with plain text instead of a JSON tool call (silent declination), the node appends a stricter retry message ("You did not call a tool. You MUST call one of the listed tools now.") and re-invokes the model — up to `max_retries` (2) retry attempts, counted against the same budget as parse-failure retries.
- `react`: ReAct loop (Thought → Action → Observation) — for weaker models

**Layer 3 — Output Validation + Retry (built into Layer 2 nodes):**

- Validate tool call arguments against schema
- Re-prompt with error feedback on parse failure
- Retry on silent declination when `force_tool_use=True` (structured mode only)
- Max `max_retries` (2) attempts before falling back to no-tool response

**Benefits:** Semantic layer eliminates irrelevant tools before the model sees them (token savings). Model-driven layer lets the LLM make intelligent decisions. Validation layer handles parsing failures gracefully.

**Performance:** ~50-100ms overhead for Layer 1 (5 tools). Min-top-score gate skips Layer 2 entirely for non-tool queries (saves ~3s LLM call).

### **8.2 Design Decisions**

| Decision | Rationale |
| --- | --- |
| Default `structured` not `native` | Safe for all models; native is opt-in per provider |
| Semantic pre-filter always runs | Reduces token context even for capable models; prevents bloated tool lists |
| Intent detection → score boost (not forced execution) | Let the model decide; heuristics inform but don't override |
| Top-K default 5 (not 10) | Fewer tools = better model accuracy and less token cost |
| `react` mode as third tier | Covers weaker on-premise models that can't follow JSON schemas reliably |
| Three state fields preserved | `candidate_tools`, `tool_results`, `tool_execution_results` — downstream nodes stay compatible |

### **8.3 Model-Specific Notes**

**LFM2-24B-A2B (Liquid AI):**

- Supports native function calling via `<|tool_call_start|>` / `<|tool_call_end|>` special tokens
- Also supports JSON output via system prompt instruction
- LM Studio serves it under "default" tool support (not "native") — LM Studio injects a custom system prompt and parses generically
- Validated working with native mode via LM Studio's OpenAI-compatible `/v1/chat/completions` endpoint

**LM Studio Tool-Calling:**

- Full OpenAI-compatible tool-calling via `tools` parameter in `/v1/chat/completions`
- "Native" parser support: Qwen, Llama-3.1/3.2, Mistral (model must have chat template + LM Studio parser)
- "Default" parser: All other models (including LFM2) — LM Studio injects a system prompt and parses generically
- If a model's default parsing is unreliable, switch that model to `model_tool_calling: structured` in config

### **8.4 LiteLLM Integration**

`LLMFactory` uses `langchain-litellm` as the sole provider abstraction. `langchain-openai` and `langchain-ollama` have been removed from the dependency tree.

- `ChatLiteLLM` — single model calls; provider detected from model string prefix (`openai/`)
- `ChatLiteLLMRouter` wrapping `litellm.Router` — primary→fallback routing with `num_retries=3`; `get_router_model()` in `graph_builder.py` uses this path
- LiteLLM handles provider detection, streaming normalization, retries, and tool format translation
- The framework retains all orchestration logic (graph, routing, memory, RAG, crews)
- **`api_base` / `api_key` construction rule**: must be passed as top-level `ChatLiteLLM` constructor kwargs. Placing them only in `model_kwargs` causes `AuthenticationError`.
- **CrewAI tool adapter**: `base.py::_get_tools_for_agent` wraps all `langchain_core.tools.BaseTool` subclasses (including `DateTimeTool`, `MCPServerTool`, etc.) into CrewAI-compatible adapters automatically.
- **Preferred-model validation**: `_validate_preferred_model` in `backend/routers/chat.py` normalizes any input format (raw `liquid/lfm2-24b-a2b`, bare `lfm2-24b-a2b`, or prefixed `openai/liquid/lfm2-24b-a2b`) to the canonical `openai/<id>` form.
- **Date anchoring**: `node_generate_response` appends a compact `[Today: YYYY-MM-DD...]` line to the system prompt; Research Crew Searcher task receives `current_date` / `current_year` as kickoff inputs.

→ **Implementation details:** [Tool Development Guide](tool_development_guide.md)

---

## **9. Tool & Plugin System**

Tools are discovered via a hybrid approach:

- **Core tools** are auto-discovered from `universal_agentic_framework/tools/` subdirectories with `tool.yaml` manifests
- **Profile tools** are explicitly registered in `config/tools.yaml`
- **MCP tools** are HTTP services wrapped with `MCPServerTool`

Each tool declares its type (`langchain_tool` or `mcp_server`), dependencies, permissions, and cost estimates in its `tool.yaml` manifest.

→ **Full guide (creating tools, manifests, MCP servers, routing, testing):** [Tool Development Guide](tool_development_guide.md)

---

> For deployment, operations, ingestion, frontend, monitoring, Docker topology, security, and upgrade paths, see [Deployment Guide](deployment_guide.md).

---

## **10. Key Design Decisions Summary**

| Decision Area | Choice | Rationale |
| --- | --- | --- |
| Repository model | Shared template with profile overlays | Keeps domain customization declarative and local to profiles |
| Orchestration | LangGraph (dedicated internal service) | Clear service boundaries and independent scaling |
| Persistence | PostgreSQL | Unified storage, ACID guarantees |
| Vector Store | Qdrant | Purpose-built, self-hosted, performant |
| LLM Providers | LM Studio / Ollama / OpenRouter (or other compatible APIs) | Provider choice is deployment-owned, not part of app runtime |
| Embeddings | OpenAI-compatible endpoint | Keeps runtime and ingestion aligned on a single remote interface |
| Frontend | Next.js + FastAPI | Modern React stack, production-ready |
| Monitoring | Prometheus | Simple, proven metrics and alerting |
| Multi-Agent | CrewAI as advanced extension path | Available, but not the baseline deployment requirement |
| Tool Discovery | Hybrid | Core auto, profile explicit |
| Ingestion | Optional built-in module | Complete solution, can be disabled |
| Language | Primary profile language with optional UI language list | Balances quality with multilingual support |

---

## **11. Open Questions & Future Considerations**

### **11.1 Current Scope**

This architecture covers:

- ✅ Single-user sessions (no multi-user collaboration)
- ✅ On-premise deployment only
- ✅ Trusted internal tools and agents
- ✅ German and English languages (extensible to others)
- ✅ Opt-in user-facing authentication (Next.js session cookies + rate limiting)

### **11.2 Potential Future Enhancements**

**Not in current scope, but architecturally possible:**

1. **Multi-user Collaboration** — shared graph states, real-time UI, conflict resolution
2. **Cloud Deployment** — Kubernetes manifests, managed services, auto-scaling
3. **Advanced Memory** — graph-based (Neo4j), temporal reasoning, cross-session sharing
4. **API-First Architecture** — public API, webhook support, SDK generation
5. **Compliance Certifications** — HIPAA, GDPR (data encryption at rest, audit tooling)

---

## **Appendix A: Technology Justifications**

### **Why Dedicated LangGraph Service (Current)?**

- **Clear boundaries**: Orchestration is isolated from API adapter and UI layers.
- **Independent scaling**: LangGraph, FastAPI, and Next.js can scale separately.
- **Operational visibility**: Internal service metrics and logs are easier to reason about.
- **Safer upgrades**: Adapter/frontend changes do not require orchestrator process coupling.

### **Why PostgreSQL for Checkpoints?**

- Already in stack (no additional service)
- ACID guarantees for critical state
- Audit trail capabilities
- Rich query capabilities for analysis

### **Why Qdrant vs Alternatives?**

- **vs Chroma**: Better performance, production-ready
- **vs Pinecone**: Self-hosted requirement
- **vs pgvector**: Dedicated vector search > Postgres extension

### **Why External Provider Endpoints vs Bundled Model Runtime?**

- Keeps the application runtime provider-agnostic and profile-driven
- Allows independent scaling/lifecycle of model infrastructure and app services
- Supports local providers (for example LM Studio or Ollama) and remote APIs (for example OpenRouter)
- Avoids coupling release cadence of provider runtimes to framework deployment

### **Why Next.js + FastAPI Architecture?**

- **Separation of Concerns**: Frontend, adapter, orchestration decoupled
- **Scalability**: Each service can scale independently
- **Modern UI**: React + TypeScript + Tailwind CSS v4 + Material Symbols Outlined for professional interfaces
- **API Flexibility**: FastAPI provides fast, documented REST endpoints
- **Deployment-oriented**: Proven stack choices with experimental-beta product maturity
- **Security Boundary**: Next.js owns user login, while FastAPI stays behind the trusted proxy boundary

---

## **Appendix B: Glossary**

| Term | Definition |
| --- | --- |
| Profile | Domain-specific deployment of the template (e.g., medical-ai-de) |
| Template | Base universal framework maintained in this repository |
| Graph | LangGraph workflow definition |
| Node | Single step in a graph execution |
| Crew | Multi-agent CrewAI team for collaborative reasoning |
| Checkpoint | Saved graph state for resumability |
| Collection | Qdrant vector store namespace |
| Plugin | Tool or MCP server registered with the framework |
| Ingestion | Process of loading documents into vector store |

*This document is complemented by:*

- **[README.md](../README.md)** (current runtime snapshot and recent changes)
- **[Configuration Reference](configuration.md)** (full schema documentation)
- **[Deployment Guide](deployment_guide.md)** (Docker topology, ingestion, security, operations)
