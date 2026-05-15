# Documentation Index

Steuermann is a domain-agnostic, on-premise agentic AI template built around LangGraph orchestration, profile overlays, and local infrastructure.

---

## Start Here

- **[README.md](../README.md)** - Project overview, architecture snapshot, feature set, and quick start
- **[status.md](status.md)** - Current runtime snapshot, notable changes, and documentation sync checklist

---

## Core Concepts

- **Template**: The shared codebase in this repository.
- **Profile overlay**: Domain-specific configuration under `config/profiles/<profile_id>/` plus optional plugins.
- **Protected core**: Domain behavior should be customized through profile overlays, prompt overrides, and plugins rather than direct edits in `universal_agentic_framework/core/`.

---

## Architecture And Configuration

- **[technical_architecture.md](technical_architecture.md)** - Service boundaries, execution flow, data flow, and extension points
- **[configuration.md](configuration.md)** - Runtime configuration files, prompt layout, environment overrides, and profile-specific settings

---

## Feature Guides

- **[cli.md](cli.md)** - Operations CLI reference: all `steuermann` commands, arguments, guardrail behaviour, and workflow examples
- **[ingestion.md](ingestion.md)** - Document ingestion, watch mode, recursive discovery, and collection alignment
- **[monitoring.md](monitoring.md)** - Metrics dashboard, Prometheus queries, and operational checks
- **[performance_optimization.md](performance_optimization.md)** - Short tuning guide for caching, token budgets, and conversation compression
- **[crewai_extension_guide.md](crewai_extension_guide.md)** - Advanced and experimental CrewAI extension patterns
- **[tool_development_guide.md](tool_development_guide.md)** - Tool manifests, routing descriptions, and custom tool implementation

---

## Profile Work

- **[profile_creation.md](profile_creation.md)** - Create a new profile overlay from the starter profile and validate it locally

---

## Runtime Snapshot

| Service          | Port                  | Role                                       |
| ---------------- | --------------------- | ------------------------------------------ |
| Next.js frontend | 3000 (host-exposed)   | Chat UI, settings, metrics dashboard       |
| FastAPI adapter  | 8001 (host-exposed)   | Auth, settings, metrics proxy, chat relay  |
| LangGraph        | 8000 (internal only)  | Orchestration engine                       |
| Prometheus       | 9090 (internal only)  | Metrics collection                         |
| PostgreSQL       | 5432 (internal only)  | Conversations, checkpoints, users          |
| Qdrant           | 6333 (internal only)  | RAG vector store and Mem0 internal storage |
| Redis            | 6379 (internal only)  | Response caching                           |

Use `docker-compose.override.yml` (copy from `.example`) to expose internal ports for local development.

## Local Development (Host)

The commands below are for host-side development and test execution. They are not required for Docker-only production usage.

```bash
poetry install
poetry run pytest
poetry run steuermann --help
poetry run steuermann docs check --format json
docker compose up -d
```
