# Changelog

## [0.2.2] — provider-model-hardening

- **fix** Model validation in `_validate_preferred_model` now derives provider prefix from the requested model ID, not the active profile's default — prevents `openrouter/...` being silently re-prefixed as `openai/...`
- **fix** Settings `POST /api/settings/user/{user_id}` preserves raw preferred model values; only normalizes IDs that already carry a recognized provider prefix
- **feat** `openrouter` recognised as a valid provider prefix throughout chat and settings validation layers
- **fix** LangGraph response handler normalizes list/dict-shaped content blocks (e.g. OpenRouter structured output) to plain text, preventing "LLM returned unexpected list" runtime errors
- **improve** `.env` and `.env.example` restructured with an explicit LLM provider section containing annotated examples for local Ollama, local LM Studio, and OpenRouter.ai
- **feat** `LLM_CAPABILITY_PROBE_ENABLED` and `LLM_CAPABILITY_PROBE_ON_STARTUP` env vars documented in `.env.example`, `.env`, and `docs/configuration.md`
- **test** Direct unit tests for `normalize_model_id` and `parse_model_id` with three-part `openai/<org>/<model>` IDs

## [0.2.1] — improved-config-flow

- **feat** Unified `steuermann` CLI with 16 commands: `profile` (active, scaffold, bundle export/import), `config` (show, explain, validate, set, unset, contract-check), `setup doctor`, `docs check`, `ingest` (ingest, watch, validate, reindex)
- **break** Profile commands now use profile IDs instead of paths: `profile scaffold --from starter --profile X` (was `--to config/profiles/X --profile-id X`)
- **feat** Profile bundling: export/import `.tar.gz` bundles with schema/framework/key compatibility validation
- **feat** Profile-safe config mutations with guardrails: `config set/unset` with dry-run, `--apply --confirm APPLY`, rollback, TTY fallback
- **feat** Configuration contract registry (`config/contracts/cli_contract.yaml`) ensures CLI/docs/code stay in sync
- **feat** `config validate` checks schema, files, env placeholders, and contract parity
- **feat** `setup doctor` preflight checks for env vars, endpoints, profile alignment with optional probing
- **feat** `docs check` validates documentation conformance and categorizes drift by domain (docs, contract, bundle-compat)
- **fix** Profile scaffold no longer writes bundle metadata to profile.yaml (moved to bundle_manifest.yaml only)
- **fix** Python 3.14 compatibility: tarfile.extractall() now uses filter="data" parameter
- **fix** YAML loading in ingest.py now has error handling with clear messages for malformed/missing files
- **fix** Ingestion logging: replaced print() with structured logger calls
- **improve** .env and .env.example aligned with APP_UID, APP_GID, CHECKPOINTER_POSTGRES_DSN, WEB_SEARCH_MCP_URL
- **improve** .gitignore updated to exclude custom profiles (keep starter template tracked)
- **improve** Bundle manifest includes explicit manifest_version field for future migrations
- **docs** New comprehensive docs/cli.md (400+ lines) with command reference, workflows, guardrails
- **docs** Updated profile_creation.md, configuration.md (LM Studio/Ollama guidance), ingestion.md (28 refs), README.md, copilot-instructions.md
- **test** Added 1096 CLI tests covering all 16 steuermann commands (871 passed, 5 skipped)

## [0.2] — refactor-memory-layer

- **perf** Singleton-cache `Mem0MemoryBackend` in factory — eliminates redundant Qdrant index round-trips per request
- **fix** Switch Mem0 LLM provider to `lmstudio` to avoid LM Studio `json_object` 400 errors; `infer=True` now works natively
- **feat** Structured multi-message exchange passed to Mem0 `upsert` for richer inference context
- **feat** Graceful `infer=False` verbatim fallback when Mem0 returns silent empty response
- **feat** Memory retrieval quality feedback loop — Prometheus counters + `/api/analytics/memory-retrieval-quality` endpoint + `RetrievalFeedbackPanel` in frontend
- **feat** User ratings persist across cache resets
- **feat** `MemoryTrendsChart` and `MemoryMetricsPanel` components added to metrics dashboard
- **feat** Dedicated `/memories` page with listing, stats, and rating UI
- **feat** Memory management API — list, retrieve, delete with user access control
- **feat** Tool prefiltering enforces web search intent override
- **refactor** Replace Qdrant backend with Mem0 OSS embedded backend; add integration tests

## [0.1] — unify-llm-handling-architecture

- **feat** Disable multi-agent crews by default in config
- **fix** Increase LangGraph request timeout; improve Redis health check
- **feat** All LangChain `BaseTool` subclasses auto-wrapped as CrewAI tools (not just `MCPServerTool`)
- **feat** Crew results injected into LLM system prompt as `=== RESEARCH FINDINGS ===`
- **feat** Date anchoring in system prompt (`[Today: YYYY-MM-DD]`) and crew task context
- **refactor** Unified LLM integration via LangChain/LiteLLM; update model aliases and configs

## [0.0] — initial

- **feat** Initial framework: LangGraph orchestration, FastAPI adapter, Next.js frontend, Docker Compose stack
- **feat** Linux compatibility — `APP_UID`/`APP_GID` support in Dockerfiles
