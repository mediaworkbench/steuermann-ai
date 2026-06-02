# Configuration Reference

Reference for the runtime configuration files used by this repository.

---

## Overview

The runtime configuration model is:

1. **Repository defaults** in `config/*.yaml`
2. **Profile overlays** in `config/profiles/<profile_id>/*.yaml`
3. **Environment variables** in `.env` and container runtime settings

**Loading order:** repository defaults → profile overlay → environment variables

The docs use **profile** as the product term. The schema uses the top-level config key `profile` throughout.

`PROFILE_ID` is required at runtime. `base` is no longer a runnable profile id; it only refers to repository-level defaults.

Configuration files remain directly editable. The `steuermann` CLI is a validation, diagnostics, and scaffolding companion; it does not replace manual YAML or `.env` editing.

---

## Core Configuration (`core.yaml`)

### Profile Identification

```yaml
profile:
  name: "medical-ai-de" # Unique profile identifier
  language: "de" # Primary language (ISO 639-1: en, de, fr, es, etc.)
  supported_languages: ["de", "en"] # Languages exposed to the settings UI for this profile
  locale: "de_DE" # Locale for formatting
  timezone: "Europe/Berlin" # Timezone for timestamps
```

**Required fields:** `name`, `language`

**Language behavior:**

- `profile.language` is the profile's primary/default language.
- `profile.supported_languages` is optional and controls which languages the settings UI offers.
- If `supported_languages` is not set, the backend falls back to prompt file languages, then to `profile.language`.
- Chat requests use the saved user setting as the source of truth for language selection.

### Prompt Configuration

Prompt text is stored in per-language files rather than inline inside `core.yaml`.

**Base prompts:** `config/profiles/<id>/prompts/<language>.yaml`

**Profile overrides:** `config/profiles/<profile_id>/prompts/<language>.yaml`

Example:

```text
config/
├── core.yaml
├── prompts/
│   ├── en.yaml
│   └── de.yaml
└── profiles/
    └── starter/
        └── prompts/
            ├── en.yaml
            └── de.yaml
```

Example prompt file:

```yaml
response_system: "You are a helpful AI assistant."
synthesis: "Synthesize the available information into a concise answer."
synthesis_with_sources: "Synthesize the available information and cite the provided sources."
language_enforcement: "Respond in English unless the user explicitly asks otherwise."
```

**Resolution order:** base prompt file → profile prompt override → environment override where supported by runtime.

### LLM Configuration

Runtime LLM configuration lives in the active profile overlay: `config/profiles/<profile_id>/core.yaml`. Repository-level `config/core.yaml` is limited to deployment-global settings such as database, memory, and checkpointing.

Each role (`chat`, `embedding`, `vision`, `auxiliary`) is a flat block with a single provider. The `llm.providers` registry is synthesized at parse time from these role blocks and is not a YAML-configurable field.

```yaml
# config/profiles/starter/core.yaml
llm:
  roles:
    chat:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/google/gemma-4-e4b"
      temperature: 0.3          # 0.0-2.0; higher = more creative
      max_tokens: 16384
      timeout: 600

    embedding:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/text-embedding-granite-embedding-278m-multilingual"
      temperature: 0.0
      timeout: 300

    vision:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/google/gemma-4-e4b"
      temperature: 0.2
      max_tokens: 2048
      timeout: 600

    auxiliary:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/google/gemma-4-e2b"
      temperature: 0.2
      max_tokens: 16384
      timeout: 600

  router:
    routing_strategy: simple-shuffle
    num_retries: 3
    retry_after: 1
    enable_pre_call_checks: true
    default_max_parallel_requests: 4
```

**Required roles:** `chat` and `embedding` must be present in every profile overlay. `vision` and `auxiliary` are optional — omitting them is valid.

**Role purposes:**

- `chat` — primary conversational LLM *(required)*
- `embedding` — vector embeddings for tool routing and memory retrieval *(required)*
- `vision` — multimodal/image processing requests *(optional; actively used by `analyze_image_tool`, `ocr_tool`, `analyze_document_tool`, and `analyze_chart_tool`; library tools `image_metadata_tool` and `read_barcodes_tool` do not require it; falls back to `chat` role if omitted)*
- `auxiliary` — Mem0 memory extraction, conversation summarization, RAG query rewriting/expansion, workspace intent classification, structured tool retry re-prompts, and conversation auto-titling; requires at least 16k context window; starter profile uses 16384 *(optional; falls back to `chat` role if omitted)*

**Model strings use LiteLLM's `provider/model-name` format:**

- `ollama/llama-3.1-8b` — Local Ollama (default docker-compose endpoint: `http://host.docker.internal:11434`)
- `openai/gpt-4o` — OpenAI API (set `api_key` or `OPENAI_API_KEY` env var)
- `anthropic/claude-3-5-sonnet-20241022` — Anthropic API (set `api_key` or `ANTHROPIC_API_KEY`)
- `openai/google/gemma-4-e4b` — LM Studio OpenAI-compatible server (set `api_base` to `http://host.docker.internal:1234/v1`)
- Any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) works with the same pattern

**LM Studio vs Ollama:** Configure the active provider's endpoint via the corresponding env var in `.env`. For LM Studio (port `1234`) set `LLM_PROVIDERS_LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1`; for Ollama (port `11434`) set `LLM_PROVIDERS_OLLAMA_API_BASE=http://host.docker.internal:11434/v1`. LM Studio requires the `openai/` prefix for all model IDs — bare IDs and the `lm_studio/` prefix are not recognised by the langchain-litellm adapter.

**Multi-provider fallbacks:** Each role targets a single provider. Cross-provider failover is configured at the router level via `llm.router.fallbacks`:

```yaml
llm:
  router:
    fallbacks:
      - {"openai/google/gemma-4-e4b": ["openrouter/openai/gpt-4o-mini"]}
    default_fallbacks: ["openrouter/openai/gpt-4o-mini"]
```

**Tool calling modes:**

- `structured` (default) — Tool schemas injected into the system prompt as JSON. The model outputs a JSON tool call. Works with any model.
- `native` — Uses `bind_tools()` — the model decides which tools to call. Declare intent in YAML; the probe confirms or downgrades at runtime.
- `react` — ReAct-style loop (Thought → Action → Observation). Best for weaker models that can't follow JSON schemas reliably.

Mode is declared per-model in YAML under the role and confirmed at runtime by the capability probe. Two-step resolution:

1. **YAML declaration** — set `model_tool_calling` on the role. Omitting the key defaults to `structured`.
2. **Runtime probe confirmation** — if `native` is declared, the probe checks `supports_bind_tools` and `supports_tool_schema` for the active model. A stale, missing, or failed probe automatically downgrades to `structured`.

```yaml
llm:
  roles:
    chat:
      provider_id: "lmstudio"
      api_base: $LLM_PROVIDERS_LMSTUDIO_API_BASE
      model: "openai/google/gemma-4-e4b"
      temperature: 0.3
      max_tokens: 32768
      timeout: 600
      # Declare native mode when the model supports it.
      # Leave commented out to use the default (structured).
      # model_tool_calling:
      #   openai/google/gemma-4-e4b: "native"
```

Multiple models can be mapped independently within a single role:

```yaml
model_tool_calling:
  openai/openai/gpt-4o: "native"
  openai/google/gemma-4-e4b: "structured"
```

**Native reliability note:**

- Native mode includes deterministic safety fallbacks for webpage extraction URL prompts.
- If a model omits tool calls or emits malformed extraction args, the graph can infer the URL from user intent and execute extraction safely.
- In response generation, current-turn tool outputs are treated as higher priority than long-term memory if they conflict.

**Temperature guidelines:**

- `0.0-0.3` - Deterministic, factual responses
- `0.4-0.7` - Balanced creativity and accuracy
- `0.8-1.0` - Creative, varied responses
- `1.0+` - Highly creative (use with caution)

### Memory Configuration

```yaml
memory:
  vector_store:
    type: "mem0" # Mem0 OSS embedded mode (hard cutover)
    host: "qdrant" # Docker service name or hostname for Mem0 vector storage
    port: 6333 # Qdrant port
    collection_prefix: "${PROFILE_ID}" # Mem0 collection name uses this prefix (set via env var)

  embeddings:
    model: "text-embedding-granite-embedding-278m-multilingual" # embedding model
    dimension: 768 # Must match model output dimension
    provider: "remote" # Remote OpenAI-compatible provider
    remote_endpoint: "${EMBEDDING_SERVER}" # Required when provider=remote
    batch_size: 32 # Embeddings per batch

  retention:
    session_memory_days: 90 # Delete session memories after N days
    user_memory_days: 365 # Delete user memories after N days

  mem0:
    search_limit: 10 # Internal Mem0 retrieval window before local reranking
    infer_enabled: true # Enables Mem0 extraction/inference before persistence
    custom_instructions: null # Optional extraction guidance for Mem0
    co_occurrence_fanout_cap: 5 # Maximum related memories appended per retrieval
    co_occurrence_related_top_k_per_memory: 5 # Per-primary-memory related lookup depth
    co_occurrence_prune_interval_seconds: 300 # Maintenance interval for stale edge pruning
```

**Embedding provider notes:**

- `provider: "remote"` uses an OpenAI-compatible embeddings endpoint (for example LM Studio).
- Keep `dimension` synchronized with the selected model, otherwise Qdrant collection compatibility issues occur.
- If migrating dimensions (for example `384 -> 768`), recreate existing vector collections.
- Profile overlays may override `memory.embeddings.*` in `config/profiles/<profile_id>/core.yaml`, so embedding model selection can be profile-specific.

**Memory feedback ratings (importance scoring):**

- Users can rate retrieved memories from `1` to `5` stars via `POST /api/memories/{memory_id}/rate`.
- Ratings are persisted as `metadata.user_rating` through the Mem0 adapter and automatically feed the feedback factor in memory importance scoring.
- Authorization is enforced per memory owner (`user_id`) and non-owned memories return `403`.

**Mem0 extraction behavior:**

- `memory.mem0.infer_enabled: true` enables Mem0 extraction/deduplication and structured memory updates.
- `memory.mem0.infer_enabled: false` skips extraction and stores summary text verbatim via fallback path.
- Mem0 extraction uses the model configured for `llm.roles.auxiliary`.
- For local models, use a sufficiently large context window for extraction workloads (practical baseline: 16k minimum, 32k preferred).

**Mem0 co-occurrence durability controls:**

- `memory.mem0.co_occurrence_fanout_cap` bounds retrieval-time related-memory fanout.
- `memory.mem0.co_occurrence_related_top_k_per_memory` controls per-primary related-edge scan depth.
- `memory.mem0.co_occurrence_prune_interval_seconds` controls maintenance cadence for pruning stale co-occurrence edges.

**Environment variables:**

```bash
EMBEDDING_SERVER=http://host.docker.internal:1234/v1       # LM Studio embeddings endpoint
LLM_PROVIDERS_LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1  # LM Studio chat endpoint
LLM_PROVIDERS_OLLAMA_API_BASE=http://host.docker.internal:11434/v1   # Ollama chat endpoint
LLM_PROVIDERS_OPENROUTER_API_BASE=https://openrouter.ai/api/v1       # OpenRouter endpoint
```

`EMBEDDING_SERVER` and the `LLM_PROVIDERS_*_API_BASE` vars are independent — they can point to the same LM Studio instance or to separate servers. Only set the vars for providers you are actually using.

### RAG Retrieval Configuration

```yaml
rag:
  enabled: true
  collection_name: "framework" # Must match ingestion collection
  top_k: 5 # Max results per query
  pill_score_threshold: 0.6 # Minimum similarity score (filters irrelevant results)
  query_rewriting:
    enabled: true       # Rewrite/expand queries via auxiliary model before Qdrant search
    num_variants: 2     # 1 = single rewritten query; 2-3 = multi-query expansion (union results)
  with_payload:
    - text
    - file_path
    - detected_language
    - language_confidence
  with_vectors: false # Return vectors in response
  timeout_seconds: 30
```

**Notes:**

- `rag.collection_name` is the single collection identifier for both ingestion and retrieval (see [docs/ingestion.md](docs/ingestion.md)).
- `pill_score_threshold` filters low-similarity results client-side and server-side. Default `0.6` prevents irrelevant documents from leaking into the context. Set to `null` to disable.
- `with_payload` can be `true` (all fields) or a list of specific payload fields.
- `query_rewriting.num_variants` controls how many semantically varied search queries are generated. `1` rewrites the query once; `2` or `3` produces multiple variants whose Qdrant results are unioned before deduplication, improving recall for ambiguous or short queries. Uses `llm.roles.auxiliary`.

### User-Level RAG Override (`rag_config` in user settings)

Per-user RAG settings are stored in the `user_settings` table and saved via `POST /api/settings/user/{user_id}`. The `rag_config` field is always written as a complete object — partial updates wipe omitted keys:

```json
{
  "rag_config": {
    "enabled": true,
    "collection": "",
    "top_k": 5
  }
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | `boolean` | `true` | Per-session Knowledge Base toggle. When `false`, `retrieve_knowledge` returns early before any Qdrant call. Can also be overridden per request via the `rag_enabled` field in `POST /api/chat`. |
| `collection` | `string` | `""` | Override the profile's `rag.collection_name` for this user. Empty string uses the profile default. |
| `top_k` | `integer` | `5` | Override the profile's `rag.top_k` for this user. |

The frontend chat bar `Database` button and the Settings "Enable Knowledge Base" checkbox both write to this field. The toggle handler must spread the full current `rag_config` object (not just `{enabled: ...}`) to avoid wiping `collection` and `top_k`.

**Adaptive skip:** in addition to the user toggle, RAG is automatically skipped for trivial queries (greetings, pure math/datetime, tool meta-questions) regardless of `enabled`. See `docs/technical_architecture.md` section 4.4 for the three-layer skip logic.

### Database Configuration

```yaml
database:
  url: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
  pool_size: 10 # Connection pool size
  echo: false # Log all SQL queries (debug only)
```

**URL format:** `postgresql://[user]:[password]@[host]:[port]/[database]`

**Environment variable substitution:**

```yaml
database:
  url: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
```

### Token Configuration

```yaml
tokens:
  default_budget: 10000 # Parsed but not enforced — retained for observability
  per_turn_budget_ratio: 0.4
  per_node_budgets:
    summarization_node: 2000
    update_memory: 2000
```

**Token tracking (no enforcement):**

- `tokens_used`, `input_tokens`, and `output_tokens` are accumulated in `GraphState` each turn and exposed via Prometheus metrics and the SSE `metadata` event.
- Token budget enforcement (`require_tokens`, `TokenBudgetExceeded`, per-node hard limits) has been removed from all graph nodes. The `tokens.*` configuration keys are still parsed for future use but have no enforcement effect — no node discards a completed response based on token counts.
- Conversation length is controlled by the compression threshold (`llm.roles.chat.max_tokens * 0.75`) and the `SUMMARIZE_*` env vars, not by token budget caps.

### Checkpointing Configuration

LangGraph checkpointing uses PostgreSQL and is always on. The only configurable field is the DSN:

```yaml
checkpointing:
  postgres_dsn: "${CHECKPOINTER_POSTGRES_DSN}"
```

Set `CHECKPOINTER_POSTGRES_DSN` in `.env` or pass it directly:

```bash
CHECKPOINTER_POSTGRES_DSN=postgresql://user:pass@postgres:5432/framework
```

**Behavior notes:**

- `PostgresSaver` is the sole checkpointing backend — there is no SQLite fallback and no `enabled` flag.
- Per-session continuity is maintained via `configurable.thread_id = session_id`. Ephemeral requests that omit `conversation_id` skip checkpointing entirely.
- Startup pruning and periodic pruning (every 100 invocations) keep the checkpoints table flat.

### Tool Routing Configuration (Three-Tier Architecture)

```yaml
tool_routing:
  similarity_threshold: 0.55 # Cosine similarity threshold for Layer 1 pre-filter
  embedding_model: "text-embedding-granite-embedding-278m-multilingual" # Optional override
  top_k: 5 # Max candidates passed to Layer 2
  intent_boost: 0.2 # Score boost for intent-matched tools (0.0-0.5)
  max_retries: 2 # Layer 3 retry count on parse failure
  min_top_score: 0.7 # Skip Layer 2 if no tool scores above this
  min_spread: 0.10 # Clear candidates if score spread < this (flat distribution)
```

**Architecture:**

Tool selection uses a three-tier architecture:

1. **Layer 1 — Semantic Pre-filter** (always runs): Embeds user query, scores all tools via cosine similarity, applies intent boosting, returns top-K candidates. Does NOT execute tools.
2. **Layer 2 — Model-Driven Tool Calling**: The LLM receives candidate tools and decides which (if any) to call. Mode is resolved per model from `model_tool_calling` and then validated against fresh probe results (probe can downgrade native to structured).
3. **Layer 3 — Output Validation + Retry**: Validates tool call arguments against schema, re-prompts on parse failure up to `max_retries` times.

**Filtering gates (Layer 1):**

- `min_top_score` gate: If no tool scores above this value, skip Layer 2 entirely (the query is conversational, not tool-worthy). Prevents unnecessary LLM tool-calling calls.
- `min_spread` gate: If all tool scores are within `min_spread` of each other (flat distribution), clear all candidates. Prevents noise when no tool is clearly relevant.
- `similarity_threshold`: After gates pass, only tools scoring above this are forwarded as candidates.

**Tuning guidelines:**

- `similarity_threshold: 0.55` — Filters the noise floor (unrelated tools typically score 0.42-0.57 with multilingual embeddings).
- `min_top_score: 0.7` — Requires at least one confident match. With `intent_boost: 0.2`, target tools typically score 0.84-1.0.
- `min_spread: 0.10` — Catches flat distributions where no tool stands out.
- Lower `similarity_threshold` for more candidates (recall↑); raise for fewer (precision↑).
- `intent_boost` adds score to tools matching detected intents (datetime, calculation, file ops, URL extraction).
- `top_k` limits candidates sent to the model — fewer = better accuracy, less token cost.

**Notes:**

- Routing works with any LLM family — semantic scoring is model-agnostic.
- If `embedding_model` is not set, routing reuses `memory.embeddings.model`.

---

## Agents Configuration (`agents.yaml`)

### Crew Definition

```yaml
crews:
  research_crew: # Unique crew identifier
    enabled: true # Enable/disable crew
    process: "sequential" # Options: sequential, hierarchical, consensus

    agents:
      researcher: # Agent role identifier
        role: "Research Specialist"
        goal: "Find accurate and relevant information"
        backstory: "Expert researcher with 10 years experience in medical literature"
        tools: # Tools this agent can use
          - web_search
          - pubmed_search
        llm_override: null # Optional: use different LLM for this agent

      synthesizer:
        role: "Information Synthesizer"
        goal: "Combine research into actionable insights"
        backstory: "Skilled at distilling complex information into clear recommendations"
        tools: []

    max_iterations: 5 # Max iterations before terminating
    timeout_seconds: 300 # Timeout for crew execution
```

**Process types:**

**Sequential:**

```yaml
process: "sequential"
# Agents execute tasks in order defined
# Output of one feeds into next
# Most predictable and debuggable
```

**Hierarchical:**

```yaml
process: "hierarchical"
# Manager agent delegates to workers
# Requires capable manager LLM
# Best for complex workflows
```

**Consensus:**

```yaml
process: "consensus"
# Multiple agents vote on decisions
# Higher token cost
# Best for critical decisions
```

### Agent Best Practices

**Good role definitions:**

```yaml
role: "Medical Literature Researcher"
goal: "Find peer-reviewed evidence for treatment options"
backstory: "Clinical researcher with expertise in evidence-based medicine and systematic reviews"
```

**Avoid:**

```yaml
role: "Helper" # Too generic
goal: "Help user" # Not specific enough
backstory: "" # Missing context
```

---

## Tools Configuration (`tools.yaml`)

### Tool Registration

```yaml
tools:
  - name: "patient_database"                    # Unique tool identifier
    path: "universal_agentic_framework/tools/patient_database"  # Path relative to repository root
    enabled: true

  - name: "medical_calculator"
    path: "universal_agentic_framework/tools/medical_calculator"
    enabled: true

  - name: "datetime_tool"                        # Built-in framework tool
    path: "universal_agentic_framework/tools/datetime"
    enabled: true
```

**Tool discovery:**

- All tools (built-in and custom) live in `universal_agentic_framework/tools/<name>/`
- Each tool directory requires `__init__.py`, `tool.py`, and `tool.yaml`
- All tools must be explicitly listed here to be loaded (`loading_mode: explicit`)
- A `plugins/` directory for profile-specific tools is planned but does not exist yet

---

## Features Configuration (`features.yaml`)

### Feature Flags

```yaml
# Core features
multi_agent_crews: false # Enable CrewAI crews (globally disabled by default)
long_term_memory: true   # Enable Mem0/Qdrant memory system
ingestion_service: true  # Enable document ingestion service
rag_retrieval: true      # Enable RAG retrieval in the graph

# Memory retrieval tuning
memory_importance_scoring: true  # Enable importance-based reranking
memory_include_related: true     # Fetch related memories via co-occurrence
memory_top_k: 5                  # Number of primary memories to retrieve

# UI features
ui_tool_visualization: true # Show tool calls in UI
ui_token_counter: true      # Show token usage
ui_export_chat: true        # Allow chat export

# Crew features (only relevant when multi_agent_crews: true)
crew_result_caching: true
crew_cache_ttl_seconds: 3600
crew_result_validation: true
crew_chaining: false
crew_parallel_execution: false

# Memory layer emergency rollback switches
memory_load_enabled: true           # Enable load_memory_node retrieval path
memory_update_enabled: true         # Enable update_memory_node persistence path
memory_co_occurrence_enabled: true  # Enable co-occurrence tracking updates
memory_digest_chain_enabled: true   # Enable digest metadata propagation
```

**Deployment-global flags** (`ingestion_service`, `authentication`, `monitoring`) cannot be overridden in a profile overlay — only in `config/features.yaml`.

**Memory emergency controls:**

- These switches are intended for development rollback and diagnostics.
- Default values should remain `true` in normal operation.
- Turning them off bypasses specific memory pipeline segments without changing core config schemas.

---

## Ingestion Configuration

**Note:** Only active if `features.ingestion_service: true`. Ingestion settings live under the `ingestion:` key inside `config/profiles/<profile_id>/core.yaml` — there is no separate `ingestion.yaml` file.

```yaml
# config/profiles/starter/core.yaml
ingestion:
  source_path: $RAG_DATA_PATH          # Path to document directory (env var)
  language: "de"                        # Primary language for ingestion
  language_threshold: 0.8              # Minimum language detection confidence
  embedding_batch_size: 32             # Documents embedded per batch
  upsert_batch_size: 128               # Vectors upserted per Qdrant batch
  file_concurrency: 1                  # Parallel file processing workers
  incremental_mode: true               # Skip unchanged files (hash-based)
  phase_timing: true                   # Emit per-phase timing logs
  reingest_timeout_seconds: 1800       # Timeout for full reingest via API
```

**Supported file types:**

- PDF (`.pdf`)
- Word documents (`.docx`)
- CSV/Excel (`.csv`, `.xlsx`, `.xls`)
- Markdown (`.md`, `.markdown`)
- Text (`.txt`)

**Collection naming:** The RAG collection is configured separately via `rag.collection_name` in the profile overlay (see RAG Retrieval Configuration above). Use the same value when running `steuermann ingest`.

**Running ingestion:**

```bash
poetry run steuermann ingest ingest --source ./data/rag-data --collection <name>
poetry run steuermann ingest watch  --source ./data/rag-data --collection <name>
```

---

## Frontend Configuration

**Note:** Next.js frontend is configured via environment variables and next.config.js

Key settings:

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: [http://localhost:8001](http://localhost:8001))
- Tailwind CSS v4 theming via `@theme` block in `globals.css` (colors, fonts, spacing)
- Component configuration in `frontend/src/config/`

```yaml
show_tool_calls: true # Display tool invocations
max_message_width: "800px" # Max message width
```

```yaml
messages:
  welcome: "Willkommen beim medizinischen Assistenten"
  error: "Ein Fehler ist aufgetreten"
  thinking: "Einen Moment bitte..."

custom_components:
  - name: "PatientContextPanel"
    position: "sidebar" # Options: sidebar, header, footer
    module: "frontend.custom.components.PatientContextPanel"
```

---

## Environment Variables

### Required Variables

```bash
# Profile identification
PROFILE_ID=starter

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=framework
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=framework

# Vector Store
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# LLM provider endpoints (set the ones matching your active providers)
LLM_PROVIDERS_LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1
LLM_PROVIDERS_OLLAMA_API_BASE=http://host.docker.internal:11434/v1
LLM_PROVIDERS_OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

Note on PROFILE_ID:

- PROFILE_ID is the canonical deployment/profile identifier and selects the active profile overlay from config/profiles/<profile_id>.
- Metrics and analytics attribution use this same identifier value.

### Optional Variables

```bash
# LLM API Keys (if using remote providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_CAPABILITY_PROBE_ENABLED=true
LLM_CAPABILITY_PROBE_ON_STARTUP=true

# LangGraph (internal service)
LANGGRAPH_SERVER_HOST=0.0.0.0
LANGGRAPH_SERVER_PORT=8000

# FastAPI Adapter
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8001

# Next.js Frontend
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_CHAT_WORKSPACE_ENABLED=false

# Monitoring
PROMETHEUS_PORT=9090

# Chat attachments and workspace document paths
WORKSPACES_PATH=./data/workspaces
CHAT_ATTACHMENTS_ENABLED=true
CHAT_ATTACHMENTS_ROOT=/tmp/steuermann-ai/chat-workspaces
CHAT_ATTACHMENTS_MAX_FILE_BYTES=524288
CHAT_ATTACHMENTS_RETENTION_HOURS=168

# Legacy conversation-scoped workspace actions (opt-in)
CHAT_WORKSPACE_ENABLED=false
CHAT_WORKSPACE_ROOT=
CHAT_WORKSPACE_RETENTION_HOURS=24

# Debug
DEBUG=false
LOG_LEVEL=INFO
```

LLM capability probing defaults to enabled. Use `LLM_CAPABILITY_PROBE_ENABLED=false` to disable probing globally, or `LLM_CAPABILITY_PROBE_ON_STARTUP=false` to keep probing enabled but skip the automatic startup probe.

**Chat/Workspace notes:**

- `CHAT_ATTACHMENTS_MAX_FILE_BYTES` is enforced by upload validators.
- `CHAT_ATTACHMENTS_ROOT` is the base path for conversation attachment storage and the default base for workspace roots.
- `WORKSPACES_PATH` controls the host mount path used by Docker for persistent workspace files.
- `CHAT_WORKSPACE_ENABLED` gates the legacy conversation-scoped workspace action endpoints; persistent `/api/workspace/documents` APIs are available independently.

### Variable Precedence

1. **Environment variables** (highest priority)
2. **Profile YAML overlays**
3. **Base YAML defaults** (lowest priority)

Example:

```yaml
# config/core.yaml (base defaults — deployment-global only)
tokens:
  default_budget: 10000

# config/profiles/medical-ai/core.yaml (profile overlay)
tokens:
  default_budget: 15000    # Overrides base

llm:
  roles:
    chat:
      temperature: 0.2     # Profile-specific temperature override
```

---

## Configuration Validation

### Automatic Validation

All configurations are validated using Pydantic at startup:

```python
from universal_agentic_framework.config.loader import load_core_config

config = load_core_config(config_dir=Path("config"))
# Raises ValidationError if invalid
```

### Manual Validation

```bash
# Validate all configs via the canonical CLI
poetry run steuermann config validate --format json

# Fail if advisory warnings are present
poetry run steuermann config validate --strict --format json

# Validate CLI contract parity against runtime/config surface
poetry run steuermann config contract-check --format json

# Check docs conformance and emit categorized drift report
poetry run steuermann docs check --format json

# Preview a profile-safe change without writing
poetry run steuermann config set --profile starter --key core.llm.roles.chat.temperature --value 0.3 --format json

# Persist a profile-safe change and run post-write validation
poetry run steuermann config set --profile starter --key core.llm.roles.chat.temperature --value 0.3 --apply --confirm APPLY --format json

# Preview an unset operation without writing
poetry run steuermann config unset --profile starter --key core.llm.roles.chat.temperature --format json

# Persist an unset operation with post-write validation and rollback safeguards
poetry run steuermann config unset --profile starter --key core.llm.roles.chat.temperature --apply --confirm APPLY --format json
```

When `--apply` is used without `--confirm APPLY`, interactive terminals prompt for confirmation.
Non-interactive execution (automation/CI) remains strict and requires the explicit `--confirm APPLY` flag.

### Drift Report Domains

`poetry run steuermann docs check --format json` now emits `drift_report.items[]` with a `domain` field.

- `docs` - drift in README/docs command or precedence guidance.
- `contract` - general contract parity drift.
- `bundle-compat` - drift in `profile_bundle_compatibility` defaults.
- `other` - non-standard checks outside the known categories.

The same payload includes `drift_report.by_domain` counters for quick filtering in automation.

### Common Validation Errors

**Temperature out of range:**

```yaml
llm:
  roles:
    chat:
      temperature: 3.0 # ❌ Must be 0.0-2.0
```

**Missing required field:**

```yaml
profile:
  name: "my-profile"
  # ❌ Missing required 'language' field
```

**Invalid model string:**

```yaml
llm:
  roles:
    chat:
      model: "gpt2" # ❌ Model string missing provider/ prefix (e.g. "openai/gpt-4o")
```

**Embedding dimension mismatch:**

```yaml
memory:
  embeddings:
    model: "all-mpnet-base-v2" # 768 dimensions
    dimension: 384 # ❌ Mismatch! Must be 768 for this model
```

---

## Configuration Examples

### Minimal Configuration (English, Local LLM)

Split across the two files that the loader merges:

```yaml
# config/core.yaml (deployment-global — database/memory/checkpointing only)
database:
  url: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"

memory:
  vector_store:
    type: "mem0"
    host: "qdrant"
    port: 6333
    collection_prefix: "simple-assistant"
  embeddings:
    dimension: 768
  mem0:
    search_limit: 10
```

```yaml
# config/profiles/simple-assistant/core.yaml (profile overlay)
profile:
  name: "simple-assistant"
  language: "en"

llm:
  roles:
    chat:
      provider_id: "ollama"
      api_base: "${LLM_PROVIDERS_OLLAMA_API_BASE}"
      model: "ollama/llama-3.1-8b"
      temperature: 0.7
      max_tokens: 4096
      timeout: 300
    embedding:
      provider_id: "ollama"
      api_base: "${LLM_PROVIDERS_OLLAMA_API_BASE}"
      model: "ollama/nomic-embed-text"
      temperature: 0.0
      timeout: 120
    vision:
      provider_id: "ollama"
      api_base: "${LLM_PROVIDERS_OLLAMA_API_BASE}"
      model: "ollama/llama-3.1-8b"
      temperature: 0.2
      max_tokens: 2048
      timeout: 300
    auxiliary:
      provider_id: "ollama"
      api_base: "${LLM_PROVIDERS_OLLAMA_API_BASE}"
      model: "ollama/llama-3.1-8b"
      temperature: 0.2
      max_tokens: 16384
      timeout: 300
  router:
    routing_strategy: simple-shuffle
    num_retries: 3
    retry_after: 1

tokens:
  default_budget: 5000

rag:
  enabled: true
  collection_name: "simple-assistant"
  top_k: 5
  pill_score_threshold: 0.6
```

### Production Configuration (German Medical AI)

```yaml
# config/profiles/medical-ai-de/core.yaml
profile:
  name: "medical-ai-de"
  language: "de"
  locale: "de_DE"
  timezone: "Europe/Berlin"
  supported_languages: ["de", "en"]

llm:
  roles:
    chat:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/mistral/mistral-7b-instruct"
      temperature: 0.2          # Lower for medical accuracy
      max_tokens: 32768
      timeout: 600
    embedding:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/text-embedding-granite-embedding-278m-multilingual"
      temperature: 0.0
      timeout: 300
    vision:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/mistral/mistral-7b-instruct"
      temperature: 0.2
      max_tokens: 2048
      timeout: 600
    auxiliary:
      provider_id: "lmstudio"
      api_base: "${LLM_PROVIDERS_LMSTUDIO_API_BASE}"
      model: "openai/mistral/mistral-7b-instruct"
      temperature: 0.2
      max_tokens: 32768
      timeout: 600
  router:
    routing_strategy: simple-shuffle
    num_retries: 3
    retry_after: 1
    fallbacks:
      - {"openai/mistral/mistral-7b-instruct": ["openai/gpt-4o-mini"]}

memory:
  embeddings:
    dimension: 768
  mem0:
    infer_enabled: true
  retention:
    session_memory_days: 30   # shorter retention for compliance
    user_memory_days: 90

rag:
  enabled: true
  collection_name: "medical-ai-de"
  top_k: 5
  pill_score_threshold: 0.6

tokens:
  default_budget: 15000
  per_turn_budget_ratio: 0.4
```

---

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common failure modes and diagnostic commands.

---

## Best Practices

### Security

- ✅ Never commit secrets to git
- ✅ Use environment variables for passwords
- ✅ Rotate API keys regularly
- ✅ Use `.env.example` as template

### Performance

- ✅ Use lower temperature for deterministic tasks
- ✅ Configure reasonable timeouts
- ✅ Optimize chunk sizes for your documents

### Maintainability

- ✅ Document why you override base configs
- ✅ Use descriptive profile names
- ✅ Keep configs DRY (use variables)
- ✅ Version control all config files

### Multi-Language

- ✅ Keep one primary profile language via `profile.language`
- ✅ Optionally expose multiple UI languages via `profile.supported_languages`
- ✅ Use language-specific chat models where helpful
- ✅ Prefer multilingual embeddings for mixed-language retrieval
- ✅ Accept ingested documents in any language and tag detected language metadata

---

## Docker Network Configuration

By default, only the Next.js frontend (3000) and FastAPI (8001) are bound to the host. Internal services — Qdrant (6333), Prometheus (9090), PostgreSQL (5432), and Redis (6379) — are only reachable within the Docker network.

To expose them for local development (e.g. to use the Qdrant dashboard or query Prometheus directly), copy the included override template:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
docker compose up -d
```

Docker Compose merges `docker-compose.override.yml` automatically. Delete the file to return to production-safe isolation.

### Remapping ports

Use the same override file to remap ports if the defaults conflict with other local services:

```yaml
# docker-compose.override.yml
services:
  nextjs:
    ports:
      - "9000:3000" # remap Next.js to 9000
  fastapi:
    ports:
      - "9001:8001" # remap FastAPI to 9001
```

If you remap ports, update the corresponding `.env` variables: `NEXT_PUBLIC_API_BASE` (FastAPI), `QDRANT_PORT`, `POSTGRES_PORT`.

### Raspberry Pi 5 / ARM64

If Qdrant fails with `Unsupported system page size`, add this to your override:

```yaml
services:
  qdrant:
    image: haktansuren/qdrant-pi5-fixed-jemalloc
```

---

## See Also

- [Profile Setup Guide](profile_creation.md) - Creating new profiles
- [Tool Development Guide](tool_development_guide.md) - Creating custom tools
- [Technical Architecture](technical_architecture.md) - System design overview
