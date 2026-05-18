# Changelog

## [0.2.6] — tool-system-refactor-and-quality

- **fix** `file_ops_tool` disabled in `config/tools.yaml` — `sandbox_dir: ""` resolved to `/app` in Docker, giving the LLM read/write access to the entire application codebase; `WorkspaceFileOpsTool` (instantiated per-conversation in `backend/routers/chat.py`) is the correct production path for file operations
- **fix** `datetime_tool.convert_timezone` now accepts optional `time` (e.g. `"15:00"`) and `from_timezone` (e.g. `"Europe/Berlin"`) params — previously it silently duplicated `current_time` (both computed `datetime.now(ZoneInfo(tz))` and returned the same result); now supports real conversions like "what time is 3pm Berlin in New York?"
- **fix** `requested_web_results` from intent detection now injected as `max_results` into `web_search_mcp` tool calls in all three calling modes (native, structured, react) when the LLM did not specify it — previously the value was computed but never forwarded, always defaulting to 10 results
- **fix** `tool_name_map` hardcoding removed from `registry._get_description()` — now reads `default_tool` from the matching `config/tools.yaml` entry; adding a new multi-tool MCP entry no longer requires a registry code change
- **fix** Stale `entry_point: "src.main:app"` and `docker:` section (wrong image `web-search-mcp:latest` on port 9100) removed from `universal_agentic_framework/tools/web_search/tool.yaml`; actual deployment uses `mcp/duckduckgo` on port 8000 via `docker-compose.yml`
- **ops** Docker healthcheck added to `duckduckgo-mcp` service (`curl -sf http://localhost:8000/mcp`); LangGraph `depends_on` condition upgraded from `service_started` to `service_healthy`
- **refactor** `mentions_file_ops` intent detection removed from `intent_detection.py` and `node_prefilter_tools` (boosted a disabled tool); `file_ops_tool` references removed from `utility_tool_names` set and tool-result fallback list in `graph_builder.py`
- **test** `test_datetime_tool.py` updated for new `convert_timezone` output format (`"Time conversion:"` instead of `"Current time in ..."`); added tests for explicit `time` + `from_timezone` conversion and invalid time-string handling; added `DateTimeInput` schema test for new fields
- **docs** Updated intent boost table in `docs/tool_development_guide.md`, `docs/technical_architecture.md`, and `CLAUDE.md` (removed `file_ops_tool` row); updated Quick Reference tools table; updated `README.md` built-in tools list
- **refactor** Removed dead `node_route_tools` function (~155 lines) from `graph_builder.py` — it was defined but never registered with `add_node()` and had been superseded by the three-layer tool system (Layer 1 prefilter → Layer 2 LLM-driven calling → Layer 3 schema validation)
- **refactor** Removed helper infrastructure that was only used by the dead `node_route_tools`: `run_forced_tool`, `execute_semantic_scored_tools`, `build_semantic_tool_kwargs`, `prepare_scored_tools_with_forced_execution` (~300 lines across `semantic_execution.py` and `tool_preparation.py`)
- **refactor** Deleted `universal_agentic_framework/tools/sandbox.py` (254 lines) and `universal_agentic_framework/tools/rate_limiter.py` (242 lines) — both were fully implemented but never integrated into any execution path; sandbox/rate-limit enforcement can be added at integration time if ever needed
- **fix** `node_call_tools_native` no longer re-runs `detect_tool_routing_intents()` — it now reads `state["prefilter_intents"]` populated by `node_prefilter_tools` in Layer 1, eliminating a redundant embedding + regex pass per request
- **fix** `url_in_query` regex tightened: requires full URL scheme (`https?://`), explicit `www.` prefix, or a path segment on a bare domain — version strings (`v1.2`), file extensions (`.py`, `.json`), and email domains no longer trigger URL extraction
- **fix** `mentions_web_search` trigger narrowed: bare `search` and `find` removed; now requires explicit phrasing (`search the web`, `search for`, `look up`, `google`) to avoid false positives on `find the bug in my code` or generic `search` usage
- **improve** Tool YAML descriptions enriched for `calculator_tool`, `datetime_tool`, and `file_ops_tool` — added natural-language trigger synonyms in EN/DE/FR to improve cosine similarity scoring at Layer 1
- **test** Removed ~350 lines of dead tests: `TestSemanticKwargsBuilder`, `TestCalculatorExpressionExtraction`, `TestForcedToolExecutionHelper` (tested deleted functions); `TestToolSandbox`, `TestSlidingWindow`, `TestToolRateLimiter` (tested deleted files); `test_helper_namespace_exports.py` (tested deleted namespace distinction)
- **test** Fixed 3 URL-fallback tests in `test_semantic_tool_routing.py` to supply `prefilter_intents` in state, matching the updated `node_call_tools_native` contract
- **docs** Removed "Security: Sandbox & Rate Limiting" section from `docs/tool_development_guide.md` (referenced deleted `sandbox.py` and `rate_limiter.py` with dead code examples)

## [0.2.5] — probe-hardening-session-continuity

- **fix** `resolve_effective_tool_calling_mode()` now requires an exact `(provider_id, model_name)` match from probe results; previously fell back to any probe for the same provider when no exact match existed, silently applying the wrong model's capabilities
- **feat** New reason code `probe_model_not_found_forced_structured` emitted when probe data exists for the provider but not the specific model — distinguishes "no probe at all" from "wrong model probed"
- **feat** `LLMCapabilityProbeRunner.reprobe_for_model(model_name)` added — reprobes only providers whose configured model matches the given name; avoids a full reprobe on every model-change settings save
- **fix** `_trigger_reprobe_on_model_change()` in `backend/routers/settings.py` now calls the curated `reprobe_for_model()` instead of a full `run()`, reducing probe latency on settings updates
- **fix** `reprobe_model()` broken provider registry lookup fixed — was calling `llm_cfg.providers.get_registry()` which does not exist in the current flat role-based config schema
- **fix** Probe metadata noise removed — unprobed fields (`supports_streaming`, `supports_json_mode`, confidence, origin) were being emitted as `null`/`"unknown"` and displayed in the Settings UI; now only `probe_kind` and `max_output_tokens` are included
- **fix** `supports_tool_schema` removed from Settings capabilities detail panel — it was always identical to `supports_bind_tools` (both tested by the same `bind_tools()` call)
- **feat** `count_tokens_for_model(model_name, text)` added to `universal_agentic_framework/llm/budget.py` using `litellm.token_counter()` with tiktoken model-aware counting; falls back to `estimate_tokens()` character approximation on exception
- **fix** `respond_node` and `summarize_node` in `graph_builder.py` now use `count_tokens_for_model()` for input-token accounting instead of the character-estimate approximation
- **feat** "Memories used" UI folded into MetricsPanel — the always-expanded `MemoryUsedList` above each assistant message is removed; memory count now appears as a chip in the collapsed MetricsPanel summary line and as an expandable section inside the panel body; no new CSS required
- **fix** `POST /api/chat` now forwards `conversation_id` to LangGraph as `session_id` — the LangGraph checkpointer uses `session_id` as `thread_id`, so all turns within the same `conversation_id` now share graph state and conversation history; previously every request started a fresh thread regardless of `conversation_id`
- **fix** E2E memory inference test (`tests/test_live_memory_inference_e2e.py`) hardened: stale-memory cleanup now covers both `SHORT_TOKEN` and `LONG_TOKEN` prefixes; conversation isolation uses dedicated UUIDs (`conv_store` for storage turns, `conv_recall` for cross-session recall); storage instruction framing improved for reliable Mem0 fact extraction; short-recall sleep increased to 2 s
- **test** `tests/test_graph_builder_fallback.py` updated for simplified `invoke_with_model_fallback()` — replaced obsolete candidate-expansion-loop tests with error classification (generic error → `error_type="error"`) and LiteLLM context-window classification tests
- **test** `tests/test_reprobe_triggers.py` updated to verify curated `reprobe_for_model()` behavior — mock exposes the method and assertions confirm curated reprobe is called (not a full run) on model change

## [0.2.4] — active-profile-provider-cutover

- **break** Ingestion runtime settings now resolve from the active profile only; `.env`/Compose keep only deployment wiring, and `rag.collection_name` is now the single collection owner
- **break** Runtime provider/model ownership moved fully to `config/profiles/<profile_id>/core.yaml`; `PROFILE_ID` is now required and `base` is no longer a runnable profile id
- **break** Legacy `providers.primary` / `providers.fallback` assumptions removed from runtime consumers and test fixtures; role-based provider chains are now the only supported contract
- **feat** `LLMFactory.get_router_model()` now forwards profile-owned LiteLLM router policy from `llm.router` (retry, routing strategy, default parallelism)
- **fix** `backend/routers/chat.py` provider endpoint resolution now uses only the active profile's named provider registry; the final fallback to legacy `providers.primary` was removed
- **fix** `backend/routers/settings.py`, ingestion CLI/runtime defaults, tool-calling mode resolution, crews, memory backend construction, and model-resolution helpers now resolve behavior from the active profile and named provider roles instead of legacy aliases or env-owned ingestion knobs
- **fix** `.env` and `.env.example` now quote space-containing values so they can be sourced cleanly by POSIX shells and `zsh`
- **docs** Updated `README.md`, `docs/configuration.md`, `docs/ingestion.md`, `docs/technical_architecture.md`, `docs/status.md`, and `.github/ARCHITECTURE.md` for active-profile-only provider/model/ingestion configuration
- **test** Completed remaining legacy test cleanup, added direct chat endpoint resolution coverage, and validated the final cutover with focused regressions plus live stack smoke checks
- **feat** Added profile-level Mem0 extraction toggle `memory.mem0.infer_enabled` and wired it through schema, loader allowlist, memory factory, and backend behavior
- **fix** Mem0 extraction path now supports bounded infer payload compaction and robust fallback while preserving upsert continuity
- **ops** Re-enabled Mem0 infer for starter profile after increasing chat/auxiliary max tokens to `32768`
- **fix** CLI contract parity restored by adding `memory.mem0` to `config/contracts/cli_contract.yaml` `profile_safety.allowed_core_prefixes`
- **fix** `universal_agentic_framework/orchestration/helpers/model_resolution.py` typing import now includes `List` for fallback attempt annotations
- **fix** Frontend type-contract regression resolved in `frontend/src/hooks/__tests__/useProfile.test.tsx` by adding required `model_roles` mock field
- **docs** Synced README, status snapshot, and architecture/configuration docs for Mem0 infer toggle, auxiliary-model wiring, and contract parity
- **feat** Added message-quality telemetry pipeline: Prometheus counter for assistant thumbs feedback, analytics endpoint `/api/analytics/message-quality`, frontend `MessageQualityPanel`, and EN/DE i18n wiring in Metrics Trends
- **ops** Phase 8 verification completed: authenticated API smoke checks, message feedback write-path validation, Prometheus metric verification, docs drift check, and no-cache container rebuild/health validation
- **refactor** Mem0 adapter de-customization slices: enforce filters-based Mem0 search/list/delete API, switch to canonical OSS `get/delete/update` signatures, and remove redundant adapter `_text_cache`
- **refactor** Continued Mem0 adapter de-customization: removed `_owner_cache` and derive ownership from canonical Mem0 item fields/metadata (`user_id`) for lookup/delete/rating flows
- **refactor** Continued Mem0 adapter de-customization: removed `_metadata_cache` and `_rating_overrides`; metadata/rating consistency now relies on canonical Mem0 metadata normalization and canonical `update(memory_id, data=..., metadata=...)` persistence only
- **ops** Phase 3.5 memory-layer de-customization marked complete after final cache-layer removal, full regression pass, docs drift check, and no-cache backend image rebuild/recreate
- **test** Added regression coverage to lock the current Mem0 OSS adapter contract and validated full suite (`950 passed, 5 skipped`)
- **test** Added repeatable live stack E2E memory inference test (`tests/test_live_memory_inference_e2e.py`) covering short-term recall, long-term persistence (`/api/memories`), and long-term recall via `/api/chat`
- **docs** Updated README, monitoring, status, and architecture docs for message-quality telemetry and latest Mem0 adapter contract cleanup

## [0.2.3] — provider-endpoint-consolidation

- **break** `LLM_ENDPOINT` removed entirely; replaced by per-provider env vars `LLM_PROVIDERS_LMSTUDIO_API_BASE`, `LLM_PROVIDERS_OLLAMA_API_BASE`, `LLM_PROVIDERS_OPENROUTER_API_BASE` — update `.env` accordingly
- **fix** `langchain-litellm` added to FastAPI dependency group — capability probes now run successfully at startup (root cause: tools were discovered but never executed because probe results were unavailable)
- **fix** `server.py` state construction now forwards `llm_capability_probes` from the LangGraph request payload — mode reason advances from `configured_native_no_probe` to `configured_native_probe_ok`
- **fix** Fallback deduplication in `model_resolution.py` changed from `(provider, model, source)` to `(provider, model)` — prevents the same candidate being retried twice
- **feat** `config/core.yaml` provider `api_base` values interpolate from new provider-specific env vars rather than a single `LLM_ENDPOINT`
- **fix** `config/profiles/starter/core.yaml` aligned with base config: provider `api_base` now resolves from `LLM_PROVIDERS_*_API_BASE` env vars and LM Studio model IDs use canonical `openai/...` prefix
- **feat** `backend/routers/chat.py` resolves provider endpoint strictly from active provider config — no legacy fallback
- **feat** `backend/routers/settings.py` `/api/models` endpoint resolves from primary provider config — no legacy fallback
- **feat** `docker-compose.yml` injects `LLM_PROVIDERS_*_API_BASE` vars into all affected services; `WEB_SEARCH_MCP_URL` parameterised via env var
- **improve** `steuermann setup doctor` checks for provider-specific endpoint vars and probes each configured endpoint individually (was single `LLM_ENDPOINT` check)
- **improve** `.env.example` and `docs/configuration.md` updated to reflect new provider-specific env var naming; `LLM_ENDPOINT` references removed
- **improve** `docs/monitoring.md`, `docs/technical_architecture.md`, and `docs/status.md` aligned with provider-specific endpoint env vars
- **test** Updated endpoint-related fixtures/assertions in `tests/conftest.py`, `tests/test_config_loader.py`, `tests/test_langgraph_builder.py`, `tests/test_tool_invocation.py`, `tests/test_docker_compose_ingestion_env.py`, and `tests/test_steuermann_cli.py`
- **feat** Tool-calling policy moved to model-level config via `model_tool_calling` map per provider; provider-level `tool_calling` removed from runtime decision path
- **feat** Probe-authoritative mode resolution with freshness enforcement: stale/missing/invalid probe timestamps force `structured`; fresh successful probe is required for `native`
- **feat** New settings API endpoint `GET /api/llm/capabilities` exposing per-model desired mode, effective mode, probe status, and capability metadata
- **feat** Frontend Settings page now displays model capability status table with native/structured/react legend badges and an inline refresh action
- **feat** Added "Copy diagnostics" action in Settings capability panel to export probe TTL and per-model capability rows as tab-delimited clipboard output
- **feat** Capabilities table now supports per-model expandable details (configured mode, API base, bind/schema flags, mismatch flag, probe error, raw metadata)
- **feat** Settings now supports role-based model preferences (`preferred_models`) with provider-locked selectors per configured role (chat/embedding/vision/auxiliary)
- **feat** `/api/system-config` now includes `model_roles` entries (role, fixed provider, default model, role-scoped available models, optional model load error)
- **fix** User settings persistence now stores `preferred_models` JSON alongside legacy `preferred_model` and keeps chat preference synchronized for runtime compatibility
- **feat** Added configurable probe freshness env var `LLM_CAPABILITY_PROBE_TTL_SECONDS` (default `3600`) to `.env` and `.env.example`
- **fix** Chat router now forwards latest capability probe rows per provider+model (not collapsed per provider), enabling correct model-level mode resolution
- **docs** Updated `docs/configuration.md`, `docs/tool_development_guide.md`, and `docs/technical_architecture.md` for model-level tool-calling and probe freshness behavior

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
