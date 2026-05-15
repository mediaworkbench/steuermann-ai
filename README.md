# Steuermann - A Universal Agentic AI Orchestration Platform

<div align="center">

<img src="docs/steuermann.png" alt="Steuermann" width="800" />

**Build domain-specific, multi-agent AI applications — entirely on your own infrastructure.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%89%A51.1-green.svg)](https://github.com/langchain-ai/langgraph)
[![Next.js](https://img.shields.io/badge/Next.js-%E2%89%A516-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Status](https://img.shields.io/badge/status-experimental%20beta-orange.svg)](#status)

</div>

Steuermann (german for steersman) is a **self-hosted framework** for building agentic AI systems that run **on-prem-first**. It orchestrates multi-agent workflows through LangGraph, retrieves domain knowledge via RAG, remembers context across conversations, and presents everything through a modern React frontend. Provider connectivity is profile-driven, so deployments can run local-only endpoints or opt into explicitly configured external providers.

It is designed as a **reusable template**: one codebase, many deployments. Each deployment adapts to a specific domain (medical, financial, operational, analytical) through declarative **profile overlays** — configuration files that customize behavior without touching a line of framework code.

> **Experimental Beta** — Steuermann is under active development. Core architecture, APIs, configuration schemas, and feature surface may change between releases. **This is not yet production ready and many additional features will be implemented in the coming releases!**

## Why Steuermann?

Most agentic AI frameworks require cloud-hosted LLMs, lack proper UI integration, or force you into a single-domain mold. Steuermann takes a different approach:

- **Your infrastructure, your data.** Designed on-prem-first with profile-owned provider configuration for LM Studio, Ollama, OpenRouter, and other OpenAI-compatible endpoints. No data leaves your network unless you explicitly configure an external provider.
- **Not just a library — a complete application.** Ships with a production frontend (chat, settings, metrics, analytics), a FastAPI backend, and Docker Compose orchestration. `docker compose up` gives you a working AI assistant.
- **Domain-agnostic by design.** The same framework powers a medical knowledge assistant, a financial analysis tool, or an internal operations chatbot. Profiles customize everything through YAML — no forks, no rewrites.
- **Multi-agent, not multi-hack.** LangGraph owns the control flow. CrewAI crews (Research, Analytics, Code Generation, Planning) are invoked as graph nodes and return structured results. Clean separation, predictable behavior.
- **Memory that matters.** Semantic memory with importance scoring, co-occurrence linking, and knowledge graphs — not just a vector store dump. The system learns which memories are relevant and surfaces them intelligently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Next.js Frontend (Port 3000)                               │
│  Chat · Settings · Metrics Dashboard · Analytics            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FastAPI Adapter (Port 8001)                                │
│  Auth · Settings API · Metrics Proxy · Chat Streaming       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LangGraph Orchestrator (Port 8000, internal)               │
│  Graph Execution · Tool Routing · Memory · RAG · Crews      │
└────┬─────────┬───────────┬───────────┬──────────────────────┘
     │         │           │           │
  Qdrant   PostgreSQL   Redis      External LLM Provider(s)
  Vectors  Checkpoints  Cache      LM Studio / Ollama / OpenRouter
```

| Service        | Port            | Role                                                            |
| -------------- | --------------- | --------------------------------------------------------------- |
| **Next.js**    | 3000            | React frontend — chat, settings, metrics, analytics             |
| **FastAPI**    | 8001            | Backend adapter — auth, settings, metrics proxy, streaming chat |
| **LangGraph**  | 8000 (internal) | Orchestration engine — graph execution, tool routing, memory    |
| **PostgreSQL** | 5432 (internal) | Conversations, user data, checkpoints, audit logs               |
| **Qdrant**     | 6333 (internal) | Vector database — RAG embeddings and Mem0 internal storage      |
| **Redis**      | 6379 (internal) | Response caching, message broker                                |
| **Prometheus** | 9090 (internal) | Metrics collection and alerting                                 |
| **External LLM Provider(s)** | External endpoint(s) | Configured provider APIs (for example LM Studio, Ollama, OpenRouter, other OpenAI-compatible endpoints) |

By default only the frontend (3000) and FastAPI (8001) are bound to the host. Internal services (Qdrant, Prometheus, PostgreSQL, Redis) are accessible only within the Docker network. See [step 3](#3-optional-expose-internal-services-for-local-development) below to expose them during development.

---

## Features

### Multi-Agent Orchestration

LangGraph is the single source of truth for control flow. It decides what happens and when. CrewAI provides the collaborative reasoning, but only as workers invoked from graph nodes — they never make routing decisions.

- **4 specialized crews** ship out of the box:
  - **Research Crew** — web research, RAG retrieval, and synthesis across multiple agents (Searcher → Analyst → Writer)
  - **Analytics Crew** — data analysis with pattern recognition, statistical validation, and business insight generation
  - **Code Generation Crew** — software development pipeline with Architect, Developer, and QA Engineer agents
  - **Planning Crew** — project planning with requirements analysis, task decomposition, and risk assessment
- **Configurable process types**: sequential, hierarchical, or consensus-based agent collaboration
- **Crew result caching** with configurable TTL to avoid redundant multi-agent executions
- **Timeout and retry logic** with structured error handling per crew

> CrewAI crews are currently disabled by default via `multi_agent_crews: false` in `config/features.yaml`. They are fully implemented and tested but pending fine-tuning for production workloads.

### Advanced Memory System

Memory is not an afterthought — it is an explicit, first-class part of the execution graph. Memory nodes load relevant context before the LLM responds and persist new memories after.

- **Mem0 OSS** embedded backend with Qdrant-backed vector storage — no external memory service required
- **Semantic search** via Mem0's retrieval pipeline with configurable similarity thresholds
- **Importance scoring** — multi-factor ranking based on relevance, recency (exponential decay), access frequency (logarithmic), and explicit user feedback
- **User rating feedback loop** — memories rated after retrieval are tracked via Prometheus counters; feedback coverage visible in the Metrics Trends dashboard
- **Co-occurrence linking** — automatically builds a knowledge graph by tracking which memories are retrieved together, enabling context expansion and related-memory discovery
- **Memory summarization** — compresses and synthesizes older memories to maintain quality without unbounded growth
- **Explicit lifecycle** — memory load and update operations are dedicated graph nodes, not hidden side effects
- **Configurable extraction mode** — `memory.mem0.infer_enabled` can switch between full Mem0 extraction (`true`) and verbatim persistence fallback (`false`)
- **Auxiliary-role extraction binding** — memory extraction uses the `llm.roles.auxiliary` model path, so extraction capacity is controlled via auxiliary role model/context settings
- **Full CRUD API** — `/api/memories` endpoints with list, detail, delete, stats, and rate; `/memories` frontend page for user-facing memory management

### RAG & Knowledge Ingestion

A containerized ingestion service watches your document directories and automatically indexes everything into Qdrant for retrieval-augmented generation.

- **Supported formats**: PDF, DOCX, Markdown (.md, .markdown), Plain Text (.txt)
- **Recursive directory discovery** — drop files into any subfolder and they are found automatically
- **Watch mode** with three layers of reliability:
  - Watchdog filesystem monitoring for instant detection
  - Periodic fallback sweep every 30 seconds for missed events
  - Initial startup sweep to ingest all existing documents
- **Incremental processing** — file hashing skips unchanged documents; changed files have old chunks replaced automatically
- **Language detection** — every chunk is tagged with `detected_language`, `language_confidence`, and `target_language` metadata. All languages are accepted
- **Auto-deletion** — removing a source file automatically purges its chunks from Qdrant
- **Configurable chunking** with overlap, batch embedding, and concurrent file processing
- **Phase timing metrics** — per-file performance breakdown (parse, chunk, embed, upsert)

### Extensible Tool System

Tools are discovered dynamically from YAML manifests and can be LangChain-native Python classes or remote MCP (Model Context Protocol) servers.

- **Built-in tools**:
  - **DateTime** — current time, timezone conversions
  - **Calculator** — math operations, unit conversions, statistics
  - **File Operations** — sandboxed file read/write/list with permission controls
  - **Workspace File Operations** — workspace-specific file handling with intent detection
  - **Web Search** — DuckDuckGo search via MCP server
  - **Webpage Extraction** — fetch and parse web page content via MCP
- **MCP server integration** — connect any MCP-compatible tool server using the official SDK with streamable HTTP transport
- **Semantic tool routing** — queries are scored against tool descriptions using similarity matching with intent detection, so the right tools are selected without brittle keyword rules
- **Three tool-calling modes**: `native` (model function calling), `structured` (JSON schema in prompt), and `react` (Thought → Action → Observation loop)
- **Automatic mode downgrade** — detects if a model doesn't support native tool calling and automatically falls back to structured mode without breaking the conversation flow
- **Multi-provider LLM support** — role-based provider chains with automatic fallback and LiteLLM router policy, owned by `config/profiles/<profile_id>/core.yaml`. LLM capability probing detects tool-calling support at startup and on model changes
- **Tool-calling mode enforcement** — mode validation ensures consistency across all tool routing layers (prefilter, routing decision, and Layer 2 invocation nodes)
- **Tool sandboxing** with permission-based access control
- **Per-tool rate limiting** with sliding window enforcement
- **Profile-level tool configuration** — enable, disable, or reconfigure tools per deployment profile

### Multilingual Support

The framework is language-aware at every layer — from LLM model selection to prompt engineering to embedding generation.

- **Per-language LLM models** — configure different models for different languages (e.g., `llama-3.1-8b` for English, `llama-3.1-8b-german` for German)
- **Multilingual embeddings** — default model (`text-embedding-granite-278m-multilingual`, 768 dimensions) supports all major languages
- **Per-language prompt files** — system prompts, synthesis templates, and language enforcement rules stored in `config/prompts/{lang}.yaml` with profile-level overrides
- **Frontend i18n** — the UI adapts to the selected language via structured message bundles
- **Language auto-detection** during ingestion with confidence scoring and metadata tagging
- **Region-aware search** — automatic country/region inference for localized web search queries

### Modern Frontend

A production-oriented Next.js application — not a demo chat widget — with real settings management, analytics, and operational dashboards.

- **Chat interface** with streaming responses, Markdown rendering, source footnotes, and conversation history
- **Settings panel** — model selection, language preferences, tool toggles, RAG configuration
- **Metrics dashboard** at `/metrics` with two views:
  - **Real-Time** — requests, tokens, latency, active sessions, attachment stats, LLM call breakdown, live memory metrics panel (auto-refreshes every 10 seconds)
  - **Trends** — usage over time, token consumption, latency analysis, memory trends, retrieval feedback loop panel, summary cards, CSV export
- **Memory management** at `/memories` — browse, rate, and delete individual memories with full filtering and sorting
- **Workspace sidebar** for managing uploaded documents per conversation
- **Dark/light theme** with system preference detection
- **Profile-aware branding** — colors, logos, and labels adapt to the active deployment profile
- **Optional authentication** — session-based login via `AUTH_ENABLED=true` with scrypt password hashing

### Performance & Reliability

- **Multi-layer caching**: LLM response cache, memory query cache, and crew result cache backed by Redis (with graceful fallback to in-memory)
- **4 eviction policies**: LRU, LFU, FIFO, TTL — selectable per cache layer
- **Cache warming** — preload common queries on startup for immediate responsiveness
- **gzip compression** for cached large responses with automatic stats tracking
- **Conversation summarization** — when conversations grow beyond a configurable length, older messages are automatically summarized to reduce token usage (40–60% token reduction)
- **Circuit breakers** — prevent cascading failures when downstream services are unhealthy
- **Rate limiting** — per-user request throttling via SlowAPI with configurable limits
- **Qdrant snapshots** — vector database backup and restore support

### Monitoring & Observability

- **Prometheus metrics** for every layer: graph requests, node duration, token usage per model, LLM calls, memory operations, cache hit rates, attachment pipeline stats
- **Frontend metrics dashboard** — queries Prometheus via FastAPI, no direct Prometheus UI access needed
- **Structured JSON logging** via structlog with contextual fields (user_id, session_id, node, profile)
- **Custom alert rules** — Prometheus alerting configuration in `monitoring/alerts.yml`
- **Performance regression tests** — benchmarks for cache throughput and latency

### Profile-Based Deployment

One codebase serves unlimited deployment scenarios. Each domain gets a **profile** — a set of YAML overlays that customize behavior without code changes.

- **Profile structure**: `config/profiles/<profile_id>/` with `profile.yaml`, `core.yaml`, `features.yaml`, `agents.yaml`, `tools.yaml`, `ui.yaml`
- **What profiles control**: LLM provider and model selection, prompt templates, enabled tools, crew configuration, UI branding/theming, feature flags, knowledge collection references
- **Activation**: set `PROFILE_ID=medical-ai-de` in `.env` and restart
- **Plugin system**: profile-specific tools in `plugins/` are auto-discovered via the tool registry
- **Starter profile** ships as a working template to copy and customize

### Security Model

Steuermann is designed for **internal, trusted deployments** behind your network perimeter.

- **Optional session-based auth** with scrypt password hashing and configurable session secrets
- **Bearer token boundary** between Next.js and FastAPI (`CHAT_ACCESS_TOKEN`)
- **Network isolation by default** — Qdrant, Prometheus, PostgreSQL, and Redis are only accessible within the Docker network; only the frontend and API are exposed to the host
- **Per-user rate limiting** enabled by default
- **Tool sandboxing** with permission-based access control
- **No implicit data exfiltration** — all provider endpoints are explicit profile configuration; local-only operation is supported
- **CORS configuration** with allowlisted origins

> Public-facing, multi-tenant, and zero-trust deployments are out of scope for this release.

---

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- At least one reachable LLM provider endpoint configured for the active profile (for example [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or [OpenRouter](https://openrouter.ai/))

### 1. Clone and configure

```bash
git clone https://github.com/mediaworkbench/steuermann-ai.git
cd steuermann-ai
cp .env.example .env
# Edit .env — at minimum set POSTGRES_PASSWORD, PROFILE_ID, and the provider endpoint vars used by that profile
poetry install
poetry run steuermann setup doctor --format json
poetry run steuermann config validate --format json
poetry run steuermann config contract-check --format json
poetry run steuermann docs check --format json
```

Linux hosts (bind-mount permissions): set UID/GID in `.env` before first build so container users match your host account.

```bash
echo "APP_UID=$(id -u)" >> .env
echo "APP_GID=$(id -g)" >> .env
mkdir -p ./data/workspaces ./data/checkpoints ./data/rag-data
chown -R $(id -u):$(id -g) ./data/workspaces ./data/checkpoints ./data/rag-data
```

### 2. (Optional) Expose internal services for local development

By default, Qdrant (6333), Prometheus (9090), PostgreSQL (5432), and Redis (6379) are only reachable within the Docker network. To access them from your host — for example to use the Qdrant dashboard or query Prometheus directly — copy the override template:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
docker compose up -d
```

Docker Compose automatically merges `docker-compose.override.yml`, exposing the dev ports. Delete the file to return to production-safe network isolation.

You can also use the same override file to remap ports if the defaults conflict with other local services:

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

**If you remap ports**, update the corresponding `.env` variables:

- `NEXT_PUBLIC_API_BASE` if FastAPI moves to a non-standard port.
- `QDRANT_PORT` if Qdrant is remapped.
- `POSTGRES_PORT` if PostgreSQL is remapped.

Raspberry Pi 5 / ARM64: if Qdrant fails with `Unsupported system page size`, add this to your override:

```yaml
services:
  qdrant:
    image: haktansuren/qdrant-pi5-fixed-jemalloc
```

### 3. Start the stack

```bash
docker compose up -d
```

| URL                             | What you get                                              |
| ------------------------------- | --------------------------------------------------------- |
| http://localhost:3000           | Chat, settings, metrics dashboard                         |
| http://localhost:8001           | FastAPI API                                               |
| http://localhost:6333/dashboard | Qdrant dashboard (requires `docker-compose.override.yml`) |
| http://localhost:9090           | Prometheus (requires `docker-compose.override.yml`)       |

Under the hood, `docker compose up -d` also starts **PostgreSQL** (5432), **Redis** (6379), **DuckDuckGo MCP** (internal), and the **Ingestion** worker (watches `RAG_DATA_PATH` for documents). LangGraph (8000), Qdrant (6333), Prometheus (9090), PostgreSQL (5432), and Redis (6379) are internal-only by default and not exposed to the host unless you use the override file (see step 3).

### 4. (Optional) Enable authentication

Set these in `.env`:

```bash
AUTH_ENABLED=true
AUTH_USERNAME=admin
AUTH_PASSWORD_HASH='scrypt$...$...'   # see .env.example for generation command
AUTH_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
CHAT_ACCESS_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
```

Rebuild and visit http://localhost:3000/login:

```bash
docker compose up -d --build
```

### 5. (Optional) Ingest domain knowledge

```bash
# Prepare a document directory (adjust RAG_DATA_PATH in .env)
mkdir -p ./data/rag-data

# Drop PDF, DOCX, Markdown, or TXT files into the directory
cp /path/to/your/docs/*.pdf ./data/rag-data/

# The ingestion container watches the directory and indexes automatically
docker compose up -d ingestion
```

Watch mode detects new files in real time, performs periodic sweeps every 30 seconds, and removes chunks when source files are deleted. Configure ingestion language, thresholds, batching, and collection settings in `config/profiles/<profile_id>/core.yaml`; keep `.env` for mount paths and service wiring only.

For host-side development and local test execution with Poetry, see `docs/index.md`.

### 6. (Optional) Scaffold a new profile

```bash
poetry run steuermann profile scaffold --from starter --profile my-profile
poetry run steuermann config validate --profile my-profile --format json
```

---

## Configuration

Configuration follows a three-layer hierarchy: **Base → Profile Overlay → Environment Variables**.

```
config/
├── core.yaml              # LLM providers, embeddings, token budgets, RAG
├── agents.yaml            # CrewAI crew and agent definitions
├── tools.yaml             # Tool manifests and routing
├── features.yaml          # Feature flags (caching, crews, etc.)
├── prompts/
│   ├── en.yaml            # English system prompts
│   └── de.yaml            # German system prompts
└── profiles/
    └── starter/           # Default profile (copy to create your own)
        ├── profile.yaml   # Profile metadata
        ├── core.yaml      # LLM/memory/RAG overrides
        ├── features.yaml  # Feature flag overrides
        ├── agents.yaml    # Crew overrides
        ├── tools.yaml     # Tool overrides
        └── ui.yaml        # Branding and theme
```

Activate a profile by setting `PROFILE_ID` in `.env`. See [docs/configuration.md](docs/configuration.md) for the full schema reference.

---

## Operations CLI

The `steuermann` CLI ships as the single operational interface for validation, diagnostics, scaffolding, and documentation conformance. It is available after `poetry install`:

```bash
poetry run steuermann --help
```

| Command | Purpose |
|---|---|
| `steuermann profile active` | Show active profile id, directory, and metadata validity |
| `steuermann profile scaffold --from starter --profile <id>` | Create a new profile overlay from a template |
| `steuermann profile bundle export --profile <id> --out <file>` | Package a profile into a portable `.tar.gz` bundle |
| `steuermann profile bundle import --bundle <file> --profile <id>` | Import and validate a profile bundle |
| `steuermann config show [--section <s>]` | Render the fully merged effective configuration |
| `steuermann config explain --key <dot.path>` | Trace the source of a specific config key (base / overlay / env) |
| `steuermann config validate [--strict]` | Validate schema, required files, and env substitutions |
| `steuermann config set --profile <id> --key <k> --value <v>` | Dry-run set of a profile-safe key (add `--apply --confirm APPLY` to persist) |
| `steuermann config unset --profile <id> --key <k>` | Dry-run removal of a profile-safe key override |
| `steuermann config contract-check` | Verify CLI contract parity with the runtime loader |
| `steuermann setup doctor [--probe-endpoints]` | Preflight checks: env vars, endpoints, profile alignment |
| `steuermann docs check [--strict]` | Documentation conformance drift report (read-only) |
| `steuermann ingest ingest --source <dir> --collection <name>` | Index documents into Qdrant |
| `steuermann ingest watch --source <dir> --collection <name>` | Live watch mode with automatic re-indexing |
| `steuermann ingest validate --source <dir>` | Parse-only validation without writing to Qdrant |
| `steuermann ingest reindex --source <dir> --collection <name>` | Clear and rebuild a collection |

Every command supports `--format json` for CI-compatible machine-readable output with deterministic exit codes.

See [docs/cli.md](docs/cli.md) for the full CLI reference including argument details, guardrail behaviour, and workflow examples.

---

## Documentation

**Start here:** [docs/index.md](docs/index.md) — navigation hub for all documentation.

| Document                                                             | Description                                    |
| -------------------------------------------------------------------- | ---------------------------------------------- |
| [docs/index.md](docs/index.md)                                       | Complete documentation index                   |
| [docs/cli.md](docs/cli.md)                                           | Operations CLI reference (`steuermann`)        |
| [docs/technical_architecture.md](docs/technical_architecture.md)     | System design, service boundaries, data flow   |
| [docs/configuration.md](docs/configuration.md)                       | Configuration schema and reference             |
| [docs/profile_creation.md](docs/profile_creation.md)                 | Step-by-step guide to creating domain profiles |
| [docs/tool_development_guide.md](docs/tool_development_guide.md)     | Building custom tools and MCP integrations     |
| [docs/crewai_extension_guide.md](docs/crewai_extension_guide.md)     | Adding new CrewAI crews                        |
| [docs/ingestion.md](docs/ingestion.md)                               | RAG ingestion pipeline reference               |
| [docs/monitoring.md](docs/monitoring.md)                             | Prometheus metrics and dashboard guide         |
| [docs/performance_optimization.md](docs/performance_optimization.md) | Caching, load testing, benchmarking            |

---

## Technology Stack

| Layer           | Technology                                                  | Purpose                                          |
| --------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| Orchestration   | [LangGraph](https://github.com/langchain-ai/langgraph) ≥1.1 | Workflow engine, state management, checkpointing |
| Multi-Agent     | [CrewAI](https://github.com/joaomdmoura/crewAI) ≥1.14       | Role-based collaborative agent reasoning         |
| LLM Integration | [LangChain](https://github.com/langchain-ai/langchain) ≥1.2 + [langchain-litellm](https://github.com/langchain-ai/langchain/tree/master/libs/providers/litellm) 0.6 | Unified LLM interface via LiteLLM router |
| Backend         | [FastAPI](https://fastapi.tiangolo.com/) ≥0.135             | REST API, auth, metrics proxy                    |
| Frontend        | [Next.js](https://nextjs.org/) ≥16 + React 19               | Production UI with App Router                    |
| Vector Store    | [Qdrant](https://qdrant.tech/) ≥1.7                         | RAG embeddings and Mem0 internal vector store    |
| Database        | [PostgreSQL](https://www.postgresql.org/) ≥15               | Conversations, checkpoints, users, audit logs    |
| Memory          | [Mem0 OSS](https://github.com/mem0ai/mem0) ≥2.0             | Memory abstraction with embedded Qdrant backend  |
| Embeddings      | Multilingual embedding models                                | OpenAI-compatible API                            |
| Cache           | [Redis](https://redis.io/) ≥5                               | Response caching, session data                   |
| Monitoring      | [Prometheus](https://prometheus.io/)                        | Metrics collection and alerting                  |
| LLM Providers   | [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), [OpenRouter](https://openrouter.ai/) | External provider endpoints configured per profile |
| Tools           | [MCP SDK](https://modelcontextprotocol.io/)                 | Model Context Protocol integration               |

---

## Status

Steuermann is in **experimental beta**. The core orchestration, memory, RAG pipeline, frontend, and monitoring stack are stable and tested. The following areas are actively evolving:

- CrewAI crew fine-tuning for production workloads (currently disabled by default)
- Additional document parsers and ingestion formats
- Extended MCP tool ecosystem
- Multi-user workspace features

<div align="center">

**[Documentation](docs/index.md)** · **[CLI Reference](docs/cli.md)** · **[Architecture](docs/technical_architecture.md)** · **[Configuration](docs/configuration.md)** · **[Profile Guide](docs/profile_creation.md)**

</div>
