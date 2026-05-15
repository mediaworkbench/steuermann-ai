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

- The package metadata currently reports version `0.0.1` in `pyproject.toml`.
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
├── plugins/
├── frontend/
├── backend/
├── monitoring/
├── tests/
└── universal_agentic_framework/
```

**Key principles:**

- Prefer profile overlays and plugins over direct edits to `universal_agentic_framework/core/`.
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

**GraphState Schema:**

```python
from typing import Any, Dict, List, TypedDict

class GraphState(TypedDict, total=False):
  """Core state for current graph executions."""
  messages: List[Dict[str, Any]]
  user_id: str
  language: str

  # Memory and knowledge
  loaded_memory: List[Dict[str, Any]]
  knowledge_context: List[Dict[str, Any]]

  # Tools
  loaded_tools: List[Any]
  tool_results: Dict[str, str]

  # Execution tracking
  tokens_used: int
  summary_text: str
```

### **4.3 Persistence Layer**

**Checkpoint Storage:**

- Backend: configurable (`sqlite` for local/dev, `postgres` for production)
- Runtime selection: `checkpointing` config block plus `CHECKPOINTER_*` env overrides
- Enables conversation resumption across restarts
- Per-session continuity via `configurable.thread_id = session_id` in `/invoke`

**Configuration:**

```python
# Selected by runtime config/env:
# - sqlite: langgraph.checkpoint.sqlite.SqliteSaver
# - postgres: langgraph.checkpoint.postgres.PostgresSaver
# - fallback: no checkpointer (compile without checkpointing)
```

**Local/dev default pattern:**

- `CHECKPOINTER_ENABLED=false` by default
- when enabled locally: `CHECKPOINTER_BACKEND=sqlite`
- sqlite file path persisted via docker volume mount (`./data/checkpoints`)

**Production pattern:**

- `CHECKPOINTER_ENABLED=true`
- `CHECKPOINTER_BACKEND=postgres`
- `CHECKPOINTER_POSTGRES_DSN` provided via environment

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
  │    ├──> load_tools_node
  │    │    └──> Load all available tools from registry
  │    │
  │    ├──> prefilter_tools_node (LAYER 1 — Semantic Pre-filter)
  │    │    ├──> Embed user query
  │    │    ├──> Score query against tool descriptions (cosine similarity)
  │    │    ├──> Apply intent boosting (datetime, calc, file, URL intents)
  │    │    ├──> Apply gates: min_top_score (0.7), min_spread (0.10)
  │    │    ├──> Filter by similarity_threshold (0.55), top-K (5)
  │    │    └──> Store candidate_tools in state (NO execution)
  │    │
  │    ├──> [route_tool_strategy] (conditional edge)
  │    │    ├──> call_tools_native   (LAYER 2 — bind_tools, LLM decides)
  │    │    ├──> call_tools_structured (LAYER 2 — JSON schema in prompt)
  │    │    ├──> call_tools_react    (LAYER 2 — ReAct loop)
  │    │    └──> no_tools            (0 candidates → skip Layer 2)
  │    │    Each Layer 2 node includes LAYER 3 — schema validation + retry
  │    │
  │    ├──> load_memory_node
  │    ├──> retrieve_knowledge_node (RAG, score_threshold: 0.6)
  │    │
  │    ├──> respond_node
  │    │    ├──> Inject tool_results into context
  │    │    ├──> Inject knowledge_context from RAG
  │    │    └──> Invoke LLM
  │    │
  │    ├──> summarization_node
  │    └──> update_memory_node
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

**Downgrade Logic:**

Mode is downgraded from native → structured when:

- `capability_mismatch` = true (probe detected `bind_tools()` not supported)
- `status` ∈ {"warning", "error"} (probe execution failed or found issues)
- `supports_bind_tools` = false (model doesn't support native tool binding)
- No probe data available for configured provider (fallback to no-downgrade)

Downgrade reason: `probe_capability_mismatch_downgrade`

**Reason Tracking:**

Reason field propagates through entire pipeline:

- `configured_native_probe_ok` — probe shows capability OK, native mode safe
- `configured_non_native_mode` — provider configured for structured/react
- `configured_native_no_probe` — native configured but no probe data (risky, proceed cautiously)
- `probe_capability_mismatch_downgrade` — probe detected mismatch, downgraded to structured
- `configured_native_probe_provider_not_found` — provider not found in probe results

**Benefits:**

- ✅ Prevents mode mismatches at invocation (consistency across layers)
- ✅ Non-blocking validation (graceful degradation, never breaks chat flow)
- ✅ Comprehensive audit trail (reason field aids debugging)
- ✅ Automatic downgrade (safer than failing on unsupported models)
- ✅ Works across all provider types (LM Studio, Ollama, OpenAI-compatible)

### **4.5 Response Handling**

The FastAPI adapter streams responses from LangGraph back to the Next.js frontend. Streaming support is built-in using Server-Sent Events (SSE) for real-time message updates.

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

### **7.3 Token Budgeting**

**Per-Node Budget Enforcement:**

```python
def execute_node_with_budget(
    node_fn: Callable,
    state: GraphState,
    budget: int
) -> GraphState:
    """Execute node with token budget enforcement."""

    if state["tokens_used"] >= state["token_budget"]:
        raise TokenBudgetExceeded(
            f"Budget exhausted: {state['tokens_used']}/{state['token_budget']}"
        )

    # Track tokens
    with TokenCounter() as counter:
        result = node_fn(state)

    state["tokens_used"] += counter.total_tokens

    return result
```

---

## **8. Three-Tier Tool Selection Architecture**

### **8.1 Design Philosophy**

The framework uses a **three-tier tool selection architecture** that combines semantic scoring (model-agnostic) with model-driven tool calling:

**Layer 1 — Semantic Pre-filter (always runs):**

1. Embed user query using configured embedding model
2. Score query against all tool descriptions (cosine similarity)
3. Apply intent boosting (+0.2 for detected datetime, calculation, file ops, URL intents)
4. Apply gates: min_top_score (0.7) and min_spread (0.10)
5. Filter candidates by similarity_threshold (0.55) and top-K (5)
6. Store candidates in state — NO tools are executed at this layer

**Layer 2 — Model-Driven Tool Calling (configurable per provider):**

- `native`: Tools bound via `bind_tools()` — the LLM decides which to call (or none)
- `structured`: Tool JSON schemas injected into system prompt — LLM outputs JSON tool call
- `react`: ReAct loop (Thought → Action → Observation) — for weaker models

**Layer 3 — Output Validation + Retry (built into Layer 2 nodes):**

- Validate tool call arguments against schema
- Re-prompt with error feedback on parse failure
- Max `max_retries` (2) attempts before falling back to no-tool response

**Benefits:** Semantic layer eliminates irrelevant tools before the model sees them (token savings). Model-driven layer lets the LLM make intelligent decisions. Validation layer handles parsing failures gracefully.

**Performance:** ~50-100ms overhead for Layer 1 (5 tools). Min-top-score gate skips Layer 2 entirely for non-tool queries (saves ~3s LLM call).

### **8.2 Design Decisions**

| Decision                                              | Rationale                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Default `structured` not `native`                     | Safe for all models; native is opt-in per provider                                             |
| Semantic pre-filter always runs                       | Reduces token context even for capable models; prevents bloated tool lists                     |
| Intent detection → score boost (not forced execution) | Let the model decide; heuristics inform but don't override                                     |
| Top-K default 5 (not 10)                              | Fewer tools = better model accuracy and less token cost                                        |
| `react` mode as third tier                            | Covers weaker on-premise models that can't follow JSON schemas reliably                        |
| Three state fields preserved                          | `candidate_tools`, `tool_results`, `tool_execution_results` — downstream nodes stay compatible |

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

### **8.4 LiteLLM Integration** *(completed 2026-04-30)*

`LLMFactory` uses `langchain-litellm` as the sole provider abstraction. `langchain-openai` and `langchain-ollama` have been removed from the dependency tree.

- `ChatLiteLLM` — single model calls; provider detected from model string prefix (`openai/`)
- `ChatLiteLLMRouter` wrapping `litellm.Router` — primary→fallback routing with `num_retries=3`; `get_router_model()` in `graph_builder.py` uses this path
- LiteLLM handles provider detection, streaming normalization, retries, and tool format translation
- The framework retains all orchestration logic (graph, routing, memory, RAG, crews)
- **`api_base` / `api_key` construction rule**: must be passed as top-level `ChatLiteLLM` constructor kwargs. Placing them only in `model_kwargs` causes `AuthenticationError`.
- **CrewAI tool adapter**: `base.py::_get_tools_for_agent` wraps all `langchain_core.tools.BaseTool` subclasses (including `DateTimeTool`, `MCPServerTool`, etc.) into CrewAI-compatible adapters automatically.
- **Preferred-model validation**: `_validate_preferred_model` in `backend/routers/chat.py` normalizes any input format (raw `liquid/lfm2-24b-a2b`, bare `lfm2-24b-a2b`, or prefixed `openai/liquid/lfm2-24b-a2b`) to the canonical `openai/<id>` form.
- **Date anchoring**: `node_generate_response` appends a compact `[Today: YYYY-MM-DD...]` line to the system prompt; Research Crew Searcher task receives `current_date` / `current_year` as kickoff inputs.

→ **Implementation details, security sandbox, rate limiter:** [Tool Development Guide](tool_development_guide.md)

---

## **9. Tool & Plugin System**

Tools are discovered via a hybrid approach:

- **Core tools** are auto-discovered from `universal_agentic_framework/tools/` subdirectories with `tool.yaml` manifests
- **Profile tools** are explicitly registered in `config/tools.yaml`
- **MCP tools** are HTTP services wrapped with `MCPServerTool`

Each tool declares its type (`langchain_tool` or `mcp_server`), dependencies, permissions, and cost estimates in its `tool.yaml` manifest.

→ **Full guide (creating tools, manifests, MCP servers, routing, sandbox, testing):** [Tool Development Guide](tool_development_guide.md)

---

## **10. Ingestion Pipeline Architecture**

### **10.1 Service Design**

**Ingestion Service** (optional module, default enabled):

- Runs in watch mode by default: on startup it performs an initial sweep of the mounted source path and ingests any existing documents before watching for new files.
- Periodic fallback check every 30 seconds catches files missed by watchdog.
- Auto-deletes chunks when source files are removed from watched folder.
- Target language is read from `config/profiles/<profile_id>/core.yaml` via `core.ingestion.language` (default `de` in this repo).
- All languages accepted; chunks tagged with detected language metadata.

```text
┌──────────────────────────────────────────────────────┐
│  Ingestion Service (Separate Docker Container)       │
│                                                       │
│  ┌────────────────┐      ┌──────────────────┐       │
│  │  File Watcher  │─────►│  Processing      │       │
│  │  /data/ingest/ │      │  Pipeline        │       │
│  └────────────────┘      └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │  Document Parser │       │
│                          │  - PDF           │       │
│                          │  - DOCX          │       │
│                          │  - CSV/Excel     │       │
│                          │  - Markdown      │       │
│                          │  - Text (.txt)   │       │
│                          └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │  Chunker         │       │
│                          │  (Language-aware)│       │
│                          └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │  Language        │       │
│                          │  Detection       │       │
│                          │  (Tag metadata)  │       │
│                          └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │  Embedder        │       │
│                          │  sentence-       │       │
│                          │  transformers    │       │
│                          └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │  Qdrant Writer   │       │
│                          │  (Collection:    │       │
│                          │ {profile}_{type})│       │
│                          └──────────────────┘       │
└──────────────────────────────────────────────────────┘

Supported formats: PDF, DOCX, Markdown (.md, .markdown), and Text (.txt).
Recursive subdirectory discovery with **/*.ext patterns.
```

### **10.2 Ingestion Configuration**

```yaml
# config/ingestion.yaml
ingestion:
  enabled: true

  source:
    type: "filesystem" # or "s3"
    path: "/mnt/knowledge-sources"
    watch: true # Auto-ingest new files

  processing:
    chunk_size: 512
    chunk_overlap: 50
    language: "de" # Must match profile language

  embedding:
    model: "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: 32

  collections:
    - name: "medical-ai-de-procedures"
      description: "Clinical procedures and protocols"
      file_patterns: ["**/*.pdf", "**/protocols_*.md"]
      metadata:
        category: "clinical"
        priority: "high"

    - name: "medical-ai-de-drugs"
      description: "Medication database"
      file_patterns: ["**/*.csv", "**/*medications*.xlsx"]
      metadata:
        category: "pharmaceutical"
        priority: "medium"

  validation:
    language_detection: true
    reject_threshold: 0.8 # Not used for rejection; all languages accepted and tagged
```

**Note:** Language validation accepts all documents and tags chunks with `detected_language`, `language_confidence`, and `target_language` metadata fields.

### **10.3 Ingestion CLI**

```bash
# Manual ingestion trigger
poetry run ingest --config config/ingestion.yaml --source /path/to/docs

# Watch mode (auto-ingest)
poetry run ingest --watch

# Re-index specific collection
poetry run ingest --collection medical-ai-de-procedures --reindex

# Validate without ingesting
poetry run ingest --validate-only
```

### **10.4 Language Detection & Tagging**

**Note:** All documents are accepted and tagged with detected language metadata.

```python
from langdetect import detect_langs
from pathlib import Path

class IngestionValidator:
    """Detect document language and tag chunks with metadata."""

    def __init__(self, target_language: str, threshold: float):
        self.target_language = target_language
        self.threshold = threshold  # Not used for rejection

    def should_accept(self, text: str) -> tuple[bool, str, str, float]:
        """Detect language and return acceptance decision with metadata.

        Returns:
            (should_accept, reason, detected_language, confidence)
        """
        try:
            lang_probs = detect_langs(text)
            detected = lang_probs[0].lang
            confidence = lang_probs[0].prob

            # Always accept, just tag with detected language
            return True, f"Accepted with language: {detected}", detected, confidence
        except Exception as e:
            # Accept even if detection fails
            return True, f"Language detection failed, accepting anyway", "unknown", 0.0
```

Each chunk in Qdrant includes:

```python
{
    "text": "chunk content...",
    "file_path": "/data/ingest/document.pdf",
    "target_language": "de",        # Expected language from config
    "detected_language": "en",      # Actually detected language
    "language_confidence": 0.87,    # Detection confidence (0.0-1.0)
    "chunk_index": 0,
    "chunk_count": 5
}
```

### **10.5 Collection Metadata Schema**

```python
{
    "collection_name": "medical-ai-de-procedures",
    "metadata": {
        "profile": "medical-ai-de",
        "language": "de",
        "category": "clinical",
        "version": "2024-01-15",
        "document_count": 1247,
        "last_updated": "2024-01-15T10:30:00Z",
        "source_path": "/mnt/knowledge-sources/procedures/"
    },
    "vectors": {...}
}
```

---

## **11. Frontend Architecture: Next.js + FastAPI + LangGraph**

### **11.1 Production Architecture**

The production stack separates concerns across three containers:

```text
┌─────────────────────────────────────┐
│  Next.js Frontend (Port 3000)       │
│  - Chat interface + conversations   │
│  - User profile & settings          │
│  - Analytics & metrics dashboards   │
│  - Admin panel (users, roles)       │
│  - Toast notifications (sonner)     │
│  - React 19 + TypeScript + Tailwind v4 │
└───────────┬─────────────────────────┘
            ↓ HTTP/WebSocket
┌───────────┴─────────────────────────┐
│  FastAPI Adapter (Port 8001)        │
│  - /api/chat → LangGraph            │
│  - /api/conversations (CRUD, search, export) │
│  - /api/settings (CRUD)             │
│  - /api/metrics                     │
│  - /api/analytics (trends, tokens, latency) │
│  - /api/models, /api/system-config │
│  - Protected via trusted proxy token boundary │
└───────────┬─────────────────────────┘
            ↓ HTTP
┌───────────┴─────────────────────────┐
│  LangGraph Service (Port 8000)      │
│  - Graph orchestration              │
│  - Tool routing                     │
│  - Memory management                │
│  - /metrics (Prometheus)            │
└─────────────────────────────────────┘
```

### **11.2 Chat Interface**

The frontend provides 5 pages, 17 components, and 5 hooks — all fully wired to backend APIs.

**Pages:**

| Route       | Purpose                                                                     | Backend APIs                                           |
| ----------- | --------------------------------------------------------------------------- | ------------------------------------------------------ |
| `/`         | Chat with conversation persistence                                          | `/api/chat`, `/api/conversations/*`                    |
| `/profile`  | Backward-compatible redirect to `/settings`                                 | -                                                      |
| `/settings` | Unified account + settings panel (tools, RAG, model, theme, saved language) | `/api/settings/*`, `/api/models`, `/api/system-config` |
| `/metrics`  | Real-time system metrics and historical trends                              | `/api/metrics`, `/api/analytics/*`                     |

**Key Hooks:**

- `useConversations` — conversation CRUD, archive, bulk ops, export (with toast notifications)
- `useSettings` — user settings CRUD
- `useMetrics` — real-time metrics with auto-refresh
- `useAnalytics` — usage trends, token consumption, latency analysis

**Language flow:**

- The chat view does not maintain an independent language selector.
- The saved user setting is the source of truth for chat language.
- `/api/system-config` returns `supported_languages` so the settings UI can render the allowed language list per profile.

**Frontend (Next.js/React):**

```typescript
// frontend/src/hooks/useChat.ts
import { useState } from "react";

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (content: string) => {
    setIsLoading(true);
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: content }),
    });

    const result = await response.json();
    setMessages((prev) => [...prev, result.message]);
    setIsLoading(false);
  };

  return { messages, sendMessage, isLoading };
}
```

**Backend Adapter (FastAPI):**

```python
# backend/routers/chat.py
from fastapi import APIRouter, HTTPException
import httpx

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    """Proxy chat request to LangGraph service."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://langgraph:8000/invoke",
            json={
                "messages": [{"role": "user", "content": request.message}],
                "user_id": request.user_id,
                "language": request.language or "en",
            },
            timeout=60.0
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="LangGraph error")

    return response.json()
```

### **11.3 UI Customization**

**Theming (Tailwind CSS v4):**

Tailwind v4 uses CSS-based configuration via `@import "tailwindcss"` in `globals.css`. The base palette is defined in a `@theme` block:

```css
/* frontend/src/app/globals.css */
@import "tailwindcss";

@theme {
  --color-evergreen: #042a2b;
  --color-light-cyan: #cdedf6;
  --color-pacific-blue: #5eb1bf;
  --color-atomic-tangerine: #ef7b45;
  --color-burnt-tangerine: #d84727;

  --font-sans: "Open Sans", sans-serif;
  --font-display: "Open Sans", sans-serif;
}
```

**Icon System:**

Icons use Material Symbols Outlined (local woff2 font at `public/fonts/MaterialSymbolsOutlined.woff2`) via a reusable `Icon` component:

```typescript
// frontend/src/components/Icon.tsx
import { Icon } from './Icon';

<Icon name="smart_toy" size={18} className="text-white" />
<Icon name="settings" size={24} />
```

**Custom Components:**

```typescript
// frontend/src/components/custom/PatientPanel.tsx
import { Icon } from '@/components/Icon';

export function PatientPanel({ patientId }: { patientId: string }) {
  const { data } = usePatient(patientId);

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-bold text-evergreen flex items-center gap-2">
        <Icon name="person" size={20} />
        Patient Context
      </h3>
      <div>Name: {data.name}</div>
      <div>Diagnoses: {data.diagnoses.join(', ')}</div>
    </div>
  );
}
```

---

## **12. Monitoring & Observability**

**Key metrics** (Prometheus):

- `langgraph_requests_total` — Request count by profile/status (label key is legacy `fork_name`)
- `langgraph_tokens_used_total` — Token consumption by model/node
- `langgraph_request_duration_seconds` — Request latency
- `langgraph_active_sessions` — Concurrent sessions

**Logging:** JSON-formatted via `structlog` with trace IDs (`profile_id`, `session_id`, `user_id`).

**Frontend metrics dashboard** ([http://localhost:3000/metrics](http://localhost:3000/metrics)): Requests, tokens, latency, active sessions, memory operations, LLM calls.

**Data sources:** PostgreSQL (conversations, audit), Prometheus (real-time), Qdrant (collection stats via HTTP).

→ **Full details (alerting, architecture, operations checklist):** [Monitoring & Observability](monitoring.md)

---

## **13. Configuration Management**

Configuration uses hierarchical loading: **Base → Profile Overlay → Environment Variables**. All configs are validated by Pydantic schemas.

**Key files:**

- `config/core.yaml` — deployment-global defaults (database, memory, checkpointing)
- `config/profiles/<profile_id>/core.yaml` — runtime LLM providers, roles, router policy, RAG, token budgets, ingestion
- `config/agents.yaml` — CrewAI crew definitions
- `config/tools.yaml` — Tool enable/disable and overrides
- `config/features.yaml` — Feature flags with dependency validation

Profile customization is declarative — profiles override configs and register components, while direct core edits are avoided unless the shared template itself is being improved.

→ **Full schema reference and loading details:** [Configuration](configuration.md)

---

## **13. Docker Deployment Architecture**

### **13.1 Service Composition**

**docker-compose.yml (Template):**

```yaml
version: "3.8"

services:
  langgraph-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.langgraph
    ports:
      - "8123:8123"
    environment:
      - PROFILE_ID=${PROFILE_ID}
      - DATABASE_URL=postgresql://user:pass@postgres:5432/framework
      - QDRANT_HOST=qdrant
    depends_on:
      - postgres
      - qdrant
    volumes:
      - ./config:/app/config:ro
      - ./plugins:/app/plugins:ro
    restart: unless-stopped

  nextjs:
    build:
      context: ./frontend
      dockerfile: ../docker/Dockerfile.nextjs
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8001
    volumes:
      - ./frontend:/app:ro

  fastapi:
    build:
      context: .
      dockerfile: docker/Dockerfile.fastapi
    ports:
      - "8001:8001"
    environment:
      - LANGGRAPH_URL=http://langgraph:8000
    volumes:
      - ./backend:/app/backend:ro
      - ./config:/app/config:ro
    depends_on:
      - langgraph-server
    restart: unless-stopped

  ingestion:
    build:
      context: .
      dockerfile: docker/Dockerfile.ingestion
    environment:
      - QDRANT_HOST=qdrant
      - PROFILE_ID=${PROFILE_ID}
    volumes:
      - ./config:/app/config:ro
      - /mnt/knowledge-sources:/data/ingest:ro
    depends_on:
      - qdrant
    restart: unless-stopped
    profiles:
      - ingestion # Optional service

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=framework
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=framework
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    restart: unless-stopped

volumes:
  postgres-data:
  qdrant-data:
  prometheus-data:

networks:
  default:
    name: framework-network
```

### **13.2 Profile-Specific Overrides**

**docker-compose.override.yml (in profile deployment repo):**

```yaml
version: "3.8"

services:
  # Add profile-specific services
  medical-api:
    image: medical-api:latest
    ports:
      - "9000:9000"
    environment:
      - API_KEY=${MEDICAL_API_KEY}

  # Override configurations
  nextjs:
    environment:
      - NEXT_PUBLIC_CUSTOM_MEDICAL_ENDPOINT=http://medical-api:9000
```

### **13.3 Host Network Configuration**

**Access to local provider endpoints on host (example: LM Studio/Ollama):**

```yaml
services:
  langgraph:
    extra_hosts:
      - "host.docker.internal:host-gateway" # Linux
    environment:
      - LLM_PROVIDERS_LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1
      - LLM_PROVIDERS_OLLAMA_API_BASE=http://host.docker.internal:11434/v1
```

**Platform-specific:**

- macOS/Windows: `host.docker.internal` works natively
- Linux: Requires `--add-host=host.docker.internal:host-gateway`

---

## **14. Security & Access Control**

**Service isolation:** LangGraph runs on internal-only network. FastAPI bridges internal/external. Next.js is user-facing only.

**Secrets:** Environment variables (`.env`, not committed). Docker Secrets supported for production.

**Authentication (opt-in):** Next.js login with signed HttpOnly session cookies, trusted proxy forwarding to FastAPI, and rate limiting (slowapi). Disabled by default via `AUTH_ENABLED=false`.

**Chat attachments and workspace documents (current implementation):**

- Conversation uploads are handled by `POST /api/conversations/{conversation_id}/attachments`.
- Uploads are validated as UTF-8 text (`.txt`, `.md`, `.markdown`) and stored as user-scoped workspace documents.
- Conversation attachment rows are references to those canonical workspace documents (same document ID is reused).
- Persistent workspace document APIs are exposed under `POST/GET/PUT/DELETE /api/workspace/documents` and `GET /api/workspace/documents/{id}/download`.
- Chat requests support both `attachment_ids` and `document_ids`; resolved workspace document content is injected into LangGraph state as `workspace_documents`.
- LangGraph injects both attachment context and workspace document context into prompt construction.
- If the user message includes save-back intent and exactly one workspace document is in context, assistant output is written back to that document (version increment, metadata returned as `workspace_document_writeback`).

**Legacy workspace editing path (still present, opt-in):**

- Conversation-scoped workspace actions (`copy_to_workspace`, `read_workspace_file`, `write_workspace_file`, `write_revised_copy`) remain available behind `CHAT_WORKSPACE_ENABLED=true` and explicit edit intent.
- Filesystem operations are confined to configured roots with normalized path checks.

**Audit logging:** All data access logged to PostgreSQL `audit_log` table with user_id, action, resource_id, timestamp.

→ **Full details:** [SECURITY.md](../.github/SECURITY.md)

---

## **15. Development Workflow**

**Local setup:** `poetry install`, configure `.env`, `docker compose up -d`, `cd frontend && npm run dev`.

**Testing:** Unit tests (config, LLM factory, memory) + integration tests (full graph with mocked LLMs). Target: 70%+ coverage. Run via `poetry run pytest`.

**CI/CD:** This project does not use GitHub Actions or external CI. Run tests locally and rely on human code review.

→ **Full workflow (commands, PROJECT_LOG, testing strategy):** See [copilot-instructions.md](../.github/copilot-instructions.md)

---

## **16. Performance Considerations**

- **Token budgets:** Default 10,000 per request, per-node budgets enforced, automatic summarization when approaching limits
- **Caching:** LLM response caching (optional), session-scoped memory query caching, Qdrant collection caching
- **Concurrency:** Container-based — each session gets independent graph execution in LangGraph service, horizontal scaling via additional containers

→ **Full details:** [Performance Optimization](performance_optimization.md)

---

## **17. Disaster Recovery & Backup**

### **17.1 Data Persistence**

**What needs backup:**

1. PostgreSQL (checkpoints, artifacts, audit logs)
2. Qdrant (vector collections)
3. Configuration files
4. Knowledge source files (if stored locally)

**What doesn't need backup:**

- Docker images (reproducible)
- LangGraph code (in git)
- Temporary caches

### **17.2 Backup Strategy**

**PostgreSQL:**

```bash
# Daily automated backup
docker exec postgres pg_dump -U framework framework > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i postgres psql -U framework framework < backup_20240115.sql
```

**Qdrant:**

```bash
# Snapshot API
curl -X POST 'http://localhost:6333/collections/medical-ai-de-memory/snapshots'

# Download snapshot
curl 'http://localhost:6333/collections/medical-ai-de-memory/snapshots/snapshot-2024-01-15.snapshot' \
  --output snapshot.snapshot

# Restore
curl -X PUT 'http://localhost:6333/collections/medical-ai-de-memory/snapshots/upload' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @snapshot.snapshot
```

### **17.3 Disaster Recovery Plan**

1. **Data Loss**: Restore from latest backup (< 24h old)
2. **Service Failure**: Docker restart policies handle transient failures
3. **Full System Recovery**:

```bash
# Restore configuration
git clone <https://github.com/mediaworkbench/medical-ai-de.git>
docker compose up -d postgres qdrant
# ... restore backups

# Start services
docker compose up -d
```

---

## **18. Migration & Upgrade Path**

### **18.1 Repository Upgrade Process**

Use a repository-first upgrade flow:

1. Review [README.md](../README.md) and this architecture document for architecture or runtime changes.
2. Pull the target repository revision.
3. Compare profile overlays against updated defaults in `config/*.yaml` and `config/profiles/starter/`.
4. Rebuild the stack with `docker compose up -d --build`.
5. Run the relevant regression slice before promoting the revision.

### **18.2 Typical Compatibility Checks**

Focus on these surfaces after an upgrade:

- Configuration keys in `config/profiles/<profile_id>/core.yaml`, `config/features.yaml`, `config/tools.yaml`, and `config/agents.yaml`
- Prompt override file layout under `config/prompts/` and `config/profiles/<profile_id>/prompts/`
- Profile metadata surfaced through the FastAPI and chat metadata contracts
- Monitoring dashboards and metrics label expectations
- Ingestion and retrieval alignment for `rag.collection_name`, embeddings, and chunking settings

### **18.3 Breaking-Change Discipline**

When a revision changes a compatibility surface:

- update the starter profile if it is the new baseline
- update the affected docs pages in the same change
- rerun the targeted regression tests for the touched subsystem
- document the change in the relevant docs page and README release notes section

### **18.4 Minimum Validation**

```bash
poetry run pytest -q
docker compose up -d --build
```

### **18.5 Example Breaking Changes**

```markdown
# Example: profile compatibility review

1. Configuration key renamed or removed
2. Prompt file structure changed
3. Metrics label or dashboard assumptions changed
4. Chat metadata contract changed
```

### **18.6 Example Node Compatibility Change**

- `load_memory()` → `load_memory_node(state)`
- Action: update custom nodes or wrappers that still depend on the old call shape

---

## **19. Key Design Decisions Summary**

| Decision Area    | Choice                                                  | Rationale                                                        |
| ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------- |
| Repository model | Shared template with profile overlays                   | Keeps domain customization declarative and local to profiles     |
| Orchestration    | LangGraph (dedicated internal service)                  | Clear service boundaries and independent scaling                 |
| Persistence      | PostgreSQL                                              | Unified storage, ACID guarantees                                 |
| Vector Store     | Qdrant                                                  | Purpose-built, self-hosted, performant                           |
| LLM Providers    | LM Studio / Ollama / OpenRouter (or other compatible APIs) | Provider choice is deployment-owned, not part of app runtime     |
| Embeddings       | OpenAI-compatible endpoint                              | Keeps runtime and ingestion aligned on a single remote interface |
| Frontend         | Next.js + FastAPI                                       | Modern React stack, production-ready                             |
| Monitoring       | Prometheus                                              | Simple, proven metrics and alerting                              |
| Multi-Agent      | CrewAI as advanced extension path                       | Available, but not the baseline deployment requirement           |
| Tool Discovery   | Hybrid                                                  | Core auto, profile explicit                                      |
| Ingestion        | Optional built-in module                                | Complete solution, can be disabled                               |
| Language         | Primary profile language with optional UI language list | Balances quality with multilingual support                       |

---

## **20. Open Questions & Future Considerations**

### **20.1 Current Scope**

This architecture covers:

- ✅ Single-user sessions (no multi-user collaboration)
- ✅ On-premise deployment only
- ✅ Trusted internal tools and agents
- ✅ German and English languages (extensible to others)
- ✅ Opt-in user-facing authentication (Next.js session cookies + rate limiting)

### **20.2 Potential Future Enhancements**

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

| Term       | Definition                                                       |
| ---------- | ---------------------------------------------------------------- |
| Profile    | Domain-specific deployment of the template (e.g., medical-ai-de) |
| Template   | Base universal framework maintained in this repository           |
| Graph      | LangGraph workflow definition                                    |
| Node       | Single step in a graph execution                                 |
| Crew       | Multi-agent CrewAI team for collaborative reasoning              |
| Checkpoint | Saved graph state for resumability                               |
| Collection | Qdrant vector store namespace                                    |
| Plugin     | Tool or MCP server registered with the framework                 |
| Ingestion  | Process of loading documents into vector store                   |

*This document is complemented by:*

- **[README.md](../README.md)** (current runtime snapshot and recent changes)
- **[Configuration Reference](configuration.md)** (full schema documentation)
