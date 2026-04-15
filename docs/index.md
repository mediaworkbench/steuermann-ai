# Documentation Index

Steuermann is a domain-agnostic, on-premise agentic AI template built around LangGraph orchestration, profile overlays, and local infrastructure.

---

## Start Here

- **[README.md](../README.md)** - Project overview, architecture snapshot, feature set, and quick start
- **[technical_architecture.md](technical_architecture.md)** - Current runtime architecture snapshot and documentation sync checklist

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

| Service          | Port          | Role                                      |
| ---------------- | ------------- | ----------------------------------------- |
| Next.js frontend | 3000          | Chat UI, settings, metrics dashboard      |
| FastAPI adapter  | 8001          | Auth, settings, metrics proxy, chat relay |
| LangGraph        | 8000 internal | Orchestration engine                      |
| Prometheus       | 9090 internal | Metrics collection                        |
| PostgreSQL       | 5432 internal | Conversations, checkpoints, users         |
| Qdrant           | 6333 internal | Vector storage for RAG and memory         |

## Development Baseline

```bash
poetry install
poetry run pytest
docker compose up -d
```
