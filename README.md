# Steuermann — Profile-Driven Runtime for Multi-Agent AI Applications

**Build domain-specific, multi-agent AI systems — entirely on your own infrastructure.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%E2%89%A51.1-green.svg)](https://github.com/langchain-ai/langgraph)
[![Next.js](https://img.shields.io/badge/Next.js-%E2%89%A516-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
![Status](https://img.shields.io/badge/status-experimental%20beta-orange.svg)

Steuermann (German for steersman) is a **self-hosted runtime** for deploying and operating multi-agent AI systems **on-prem-first**. It combines graph-based orchestration, persistent memory, retrieval pipelines, tool routing, and operational infrastructure into a reusable deployment architecture.

One codebase, many deployments. Each domain deployment — medical, financial, operational, analytical — is driven by a **profile overlay**: YAML configuration files that customize every layer of behavior without touching runtime code.

> **Experimental Beta** — Steuermann is under active development. Core architecture, APIs, configuration schemas, and feature surface may change between releases. **This is not yet production ready and many additional features will be implemented in the coming releases!**

## Why Steuermann?

- **One runtime, any domain.** Profile overlays customize LLM providers, prompts, tools, crews, memory configuration, and UI branding per deployment — no forks, no rewrites.
- **Your infrastructure, your data.** Designed on-prem-first with profile-owned provider configuration for LM Studio, Ollama, OpenRouter, and other OpenAI-compatible endpoints. No data leaves your network unless you explicitly configure an external provider.
- **Not just a library — a complete application.** Ships with a frontend (chat, settings, metrics, analytics), a FastAPI backend, and Docker Compose orchestration. `docker compose up` gives you a working AI system.
- **Graph-owned control flow.** LangGraph is the single source of truth for execution. CrewAI crews are invoked as graph nodes — workers that return structured results, never orchestrators.
- **Memory that matters.** Persistent semantic memory with importance scoring, co-occurrence linking, and user feedback — not just a vector store.

---

## Architecture

```text
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

By default only the frontend (3000) and FastAPI (8001) are bound to the host. Internal services (Qdrant, Prometheus, PostgreSQL, Redis) are accessible only within the Docker network. See [Docker Network Configuration](docs/configuration.md#docker-network-configuration) for how to expose them during development.

---

## Features

### Deployment Profiles

Profiles are the architectural core of Steuermann — the mechanism that makes one codebase serve unlimited deployment scenarios. Each domain deployment is a self-contained YAML overlay that customizes LLM providers, prompts, tools, memory settings, crews, UI branding, and feature flags without touching the runtime. Activate a profile by setting `PROFILE_ID` in `.env`; no rebuild needed for config-only changes.

See [docs/profile_creation.md](docs/profile_creation.md) for a step-by-step guide.

### Graph-Orchestrated Execution

LangGraph owns the full execution graph — it decides what runs, in what order, and with what state. CrewAI crews (Research, Analytics, Code Generation, Planning) are invoked as graph nodes: workers that return structured results, never orchestrators. Crews are disabled by default and can be enabled per profile.

See [docs/technical_architecture.md](docs/technical_architecture.md) for the full graph execution flow.

### Memory Pipeline

Memory is an explicit, first-class part of the execution graph. Dedicated nodes load relevant context before the LLM responds and persist new memories after — no hidden side effects. Memories are ranked by relevance, recency, and user feedback, with co-occurrence linking to surface related context across conversations.

See [docs/technical_architecture.md](docs/technical_architecture.md) for memory pipeline details.

### RAG & Knowledge Ingestion

Drop documents into a watched directory and they are automatically chunked, embedded, and indexed into Qdrant. The ingestion service handles incremental updates, file deletions, and startup sweeps — no manual re-indexing needed. RAG retrieval is skipped automatically for queries where it adds no value.

Administrators can inspect what was indexed from the **RAG knowledge explorer** at `/admin/rag`: search a collection by keyword and review the matching chunks with their similarity scores, against a visible marker for the production retrieval cutoff — useful for evaluating chunking, embedding, and threshold tuning.

See [docs/ingestion.md](docs/ingestion.md) for ingestion configuration and CLI reference.

### Tool Runtime & MCP Integration

Tools are discovered from YAML manifests at startup. Each tool can be a LangChain-native Python class or a remote MCP server — the routing pipeline treats both identically. Semantic similarity matching selects the right tools for each query, and the system automatically adapts its calling strategy to match what the active model supports.

See [docs/tool_development_guide.md](docs/tool_development_guide.md) for building custom tools and MCP integrations.

### Operational Interface

A Next.js frontend built for operators. It ships a streaming chat interface with image attachment support, a settings panel for runtime configuration, a metrics dashboard with real-time and historical views, a memory management page, a RAG knowledge explorer (open to administrators and researchers) for searching the knowledge base by keyword and reviewing retrieved documents, an admin user-management page, and persistent workspace documents with version history and AI-driven save-back. Branding and theming adapt to the active profile.

### User Accounts & Roles

Authentication is DB-backed with three built-in roles: **user** (chat, own data, settings), **researcher** (everything a user can do plus the RAG knowledge explorer), and **administrator** (full access plus user management). Administrators provision accounts from an in-app users page — each new account gets an auto-generated temporary password and is required to set its own on first login. The first administrator is bootstrapped from environment configuration on startup. Passwords are hashed with argon2id and verified server-side. Every user's conversations, settings, memories, and workspace documents are isolated to their account; the RAG knowledge base is shared across all users.

### Performance & Reliability

Multi-layer Redis caching covers LLM responses, memory queries, and crew results — with graceful in-memory fallback. Conversations are automatically summarized as they grow to keep token usage bounded. Circuit breakers and per-user rate limiting protect downstream services.

See [docs/performance_optimization.md](docs/performance_optimization.md) for tuning guidance.

### Monitoring & Observability

Prometheus metrics cover every layer of the stack — graph execution, token usage, cache hit rates, memory operations. The frontend metrics dashboard queries Prometheus through FastAPI; no direct Prometheus access needed. All logs are structured JSON with per-request contextual fields.

See [docs/monitoring.md](docs/monitoring.md) for the full metrics reference.

### Multilingual Support

LLM models, system prompts, and embedding models are all configurable per language in the profile overlay. Language is detected automatically during ingestion and at the chat layer, so the right model and prompt are applied without user intervention.

### Security Model

Steuermann is designed for **internal, trusted deployments** behind your network perimeter. Internal services (Qdrant, PostgreSQL, Redis, Prometheus) are Docker-network-only by default. All provider endpoints are explicitly configured — no implicit data exfiltration.

Authentication is session-based (httpOnly JWT cookie) and DB-backed: credentials are verified on the FastAPI backend with argon2id. The backend is reachable only through the Next.js proxy, which authenticates the session and forwards a trusted identity + role to the backend (guarded by a shared `CHAT_ACCESS_TOKEN`); the backend independently enforces role checks and per-user data ownership as defense in depth. Authentication can be disabled for local development, in which case the app runs as a single bootstrap-admin user.

> Public-facing, multi-tenant, and zero-trust deployments are out of scope for this release, but will be a future feature.

---

## Built-in Tool Catalog

Steuermann ships a curated set of tools that route automatically — no explicit invocation needed. Each tool is enabled per profile and responds to natural language triggers in any supported language.

| Tool | What it does | Example trigger |
| --- | --- | --- |
| `calculator_tool` | Evaluates math expressions, unit conversions, statistics | "What is sqrt(144) / 3?" |
| `datetime_tool` | Current date/time, timezone conversion | "What time is it in Tokyo?" |
| `map_tool` | Geocode cities/regions/continents, measure distances, interactive map widget | "Where is Kyoto?", "How far is London from Madrid?" |
| `file_ops_tool` | Read and write files in a sandboxed workspace | "Save this summary to notes.md" |
| `web_search_mcp` | DuckDuckGo web search + webpage content extraction (MCP server) | "Search for the latest LangGraph release" |
| `analyze_image_tool` | Describe and analyze images via a vision LLM | "What's in this photo?" |
| `ocr_tool` | Extract text from screenshots and document images | "Read the text in this screenshot" |
| `analyze_document_tool` | Structured extraction from invoices, receipts, and forms | "Parse this invoice and list the line items" |
| `analyze_chart_tool` | Structured extraction from charts and graphs | "What does this bar chart show?" |
| `image_metadata_tool` | EXIF/GPS/file metadata — no vision LLM required | "When and where was this photo taken?" |
| `read_barcodes_tool` | Barcode and QR code decoding — no vision LLM required | "Scan this QR code" |

The four vision tools (`analyze_image_tool`, `ocr_tool`, `analyze_document_tool`, `analyze_chart_tool`) require a multimodal model configured under `llm.roles.vision` in the active profile. All other tools work with any text model.

See [docs/tools.md](docs/tools.md) for the full catalog with input schemas, configuration keys, intent triggers, and instructions for adding your own tools.

---

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- At least one reachable LLM provider endpoint configured for the active profile (for example [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or [OpenRouter](https://openrouter.ai/))

### 1. Clone and configure

Fastest path — one interactive command writes a valid `.env`, generates strong secrets +
an argon2 admin password, creates (and on Linux chowns) the data directories, and runs the
pre-flight checks:

```bash
git clone https://github.com/mediaworkbench/steuermann-ai.git
cd steuermann-ai
poetry install
poetry run steuermann setup init
```

The wizard walks you through provider/profile choice (LM Studio / Ollama / OpenRouter — endpoint,
API key, per-role models, and the embedding endpoint/model/dimension) and prints the generated
credentials once at the end. Customizing the reference `starter` profile scaffolds a fresh profile
copy and points `PROFILE_ID` at it, leaving `starter` pristine. See
[docs/cli.md](docs/cli.md#steuermann-setup-init) for all flags and behavior.

#### Manual setup (equivalent)

```bash
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

### 2. Start the stack

```bash
docker compose up -d
```

| URL                                  | What you get                                              |
| ------------------------------------ | --------------------------------------------------------- |
| <http://localhost:3000>              | Chat, settings, metrics dashboard                         |
| <http://localhost:8001>              | FastAPI API                                               |
| <http://localhost:6333/dashboard>    | Qdrant dashboard (requires `docker-compose.override.yml`) |
| <http://localhost:9090>              | Prometheus (requires `docker-compose.override.yml`)       |

Under the hood, `docker compose up -d` also starts **PostgreSQL** (5432), **Redis** (6379), **DuckDuckGo MCP** (internal), and the **Ingestion** worker (watches `RAG_DATA_PATH` for documents). LangGraph (8000), Qdrant (6333), Prometheus (9090), PostgreSQL (5432), and Redis (6379) are internal-only by default and not exposed to the host unless you use the override file.

### 3. (Optional) Enable authentication

With `AUTH_ENABLED=false` the app runs as a single local user (the dev bypass). To require
login and enable multi-user accounts, configure the **bootstrap administrator** in `.env`:

```bash
AUTH_ENABLED=true
AUTH_USERNAME=admin
AUTH_ADMIN_EMAIL=admin@example.com
# argon2id hash — wrap in single quotes so the '$' chars stay literal:
AUTH_PASSWORD_HASH='$(poetry run python -c "from argon2 import PasswordHasher; print(PasswordHasher().hash('"'"'your-password'"'"'))")'
AUTH_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
CHAT_ACCESS_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
```

Rebuild and visit <http://localhost:3000/login>:

```bash
docker compose up -d --build
```

The bootstrap administrator is seeded into the database on first start. Log in, then create
additional accounts (assigning **user**, **researcher**, or **administrator**) from the
in-app **Users** admin page — each gets a one-time temporary password and must change it on
first login.

### 4. (Optional) Ingest domain knowledge

```bash
# Prepare a document directory (adjust RAG_DATA_PATH in .env)
mkdir -p ./data/rag-data

# Drop PDF, DOCX, Markdown, or TXT files into the directory
cp /path/to/your/docs/*.pdf ./data/rag-data/

# The ingestion container watches the directory and indexes automatically
docker compose up -d ingestion
```

Watch mode detects new files in real time, performs periodic sweeps every 30 seconds, and removes chunks when source files are deleted. Configure ingestion language, thresholds, batching, and collection settings in `config/profiles/<profile_id>/core.yaml`; keep `.env` for mount paths and service wiring only.

For host-side development and local test execution with Poetry, see [docs/index.md](docs/index.md).

### 5. (Optional) Scaffold a new profile

```bash
poetry run steuermann profile scaffold --from starter --profile my-profile
poetry run steuermann config validate --profile my-profile --format json
```

---

## Configuration

Configuration follows a three-layer hierarchy: **Base → Profile Overlay → Environment Variables**.

```text
config/
├── core.yaml              # Base infra only: database, memory vector store, checkpointing
├── features.yaml          # Deployment-global feature flags
├── contracts/             # Config contract schemas (validation)
└── profiles/
    └── starter/           # Default profile (copy to create your own)
        ├── profile.yaml   # Profile metadata (top-level `profile:` key)
        ├── core.yaml      # LLM roles, embeddings, memory, RAG, tokens, ingestion
        ├── features.yaml  # Feature flag overrides
        ├── agents.yaml    # CrewAI crew and agent definitions
        ├── tools.yaml     # Tool registration and routing
        ├── ui.yaml        # Branding and theme
        └── prompts/
            ├── en.yaml    # English system prompts
            └── de.yaml    # German system prompts
```

Everything except base infra lives in the profile overlay — `agents.yaml`, `tools.yaml`, and per-language `prompts/` exist **only** under `config/profiles/<profile_id>/`, not at the base level.

Activate a profile by setting `PROFILE_ID` in `.env`. See [docs/configuration.md](docs/configuration.md) for the full schema reference.

---

## Operations CLI

The `steuermann` CLI ships as the single operational interface for validation, diagnostics, scaffolding, and documentation conformance. It is available after `poetry install`:

```bash
poetry run steuermann --help
```

Commands cover first-time setup (`setup init`), profile lifecycle (`profile active`, `scaffold`, `bundle`), configuration inspection and validation (`config show`, `validate`, `contract-check`), host preflight checks (`setup doctor`, `setup check`), and document ingestion (`ingest`). Every command supports `--format json` for CI-compatible output.

See [docs/cli.md](docs/cli.md) for the full command reference.

---

## Documentation

**Start here:** [docs/index.md](docs/index.md) — navigation hub for all documentation.

| Document | Description |
| --- | --- |
| [docs/index.md](docs/index.md) | Complete documentation index |
| [docs/quickstart.md](docs/quickstart.md) | Zero-to-running setup walkthrough |
| [docs/cli.md](docs/cli.md) | Operations CLI reference (`steuermann`) |
| [docs/technical_architecture.md](docs/technical_architecture.md) | System design, service boundaries, data flow |
| [docs/deployment_guide.md](docs/deployment_guide.md) | Docker topology, ingestion, frontend, security, and upgrade paths |
| [docs/configuration.md](docs/configuration.md) | Configuration schema and reference |
| [docs/tools.md](docs/tools.md) | Built-in tool catalog: 11 tools across utility, vision, and image library categories |
| [docs/profile_creation.md](docs/profile_creation.md) | Step-by-step guide to creating domain profiles |
| [docs/tool_development_guide.md](docs/tool_development_guide.md) | Building custom tools and MCP integrations |
| [docs/crewai_extension_guide.md](docs/crewai_extension_guide.md) | Adding new CrewAI crews |
| [docs/ingestion.md](docs/ingestion.md) | RAG ingestion pipeline reference |
| [docs/monitoring.md](docs/monitoring.md) | Prometheus metrics and dashboard guide |
| [docs/performance_optimization.md](docs/performance_optimization.md) | Caching, token budgets, and conversation compression |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common failure modes and diagnostic commands |

---

## Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) ≥1.1 | Workflow engine, state management, checkpointing |
| Multi-Agent | [CrewAI](https://github.com/joaomdmoura/crewAI) ≥1.14 | Role-based collaborative agent reasoning |
| LLM Integration | LangChain ≥1.2 + LiteLLM router | Unified LLM interface via LiteLLM router |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) ≥0.135 | REST API, auth, metrics proxy |
| Frontend | [Next.js](https://nextjs.org/) ≥16 + React 19 | Production UI with App Router |
| Vector Store | [Qdrant](https://qdrant.tech/) ≥1.7 | RAG embeddings and Mem0 internal vector store |
| Database | [PostgreSQL](https://www.postgresql.org/) ≥15 | Conversations, checkpoints, users, audit logs |
| Memory | [Mem0 OSS](https://github.com/mem0ai/mem0) ≥2.0 | Memory abstraction with embedded Qdrant backend |
| Embeddings | Multilingual embedding models | OpenAI-compatible API |
| Cache | [Redis](https://redis.io/) ≥5 | Response caching, session data |
| Monitoring | [Prometheus](https://prometheus.io/) | Metrics collection and alerting |
| LLM Providers | [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), [OpenRouter](https://openrouter.ai/) | External provider endpoints configured per profile |
| Tools | [MCP SDK](https://modelcontextprotocol.io/) | Model Context Protocol integration |

---

**[Documentation](docs/index.md)** · **[CLI Reference](docs/cli.md)** · **[Architecture](docs/technical_architecture.md)** · **[Configuration](docs/configuration.md)** · **[Profile Guide](docs/profile_creation.md)**
