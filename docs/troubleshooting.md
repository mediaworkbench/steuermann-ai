# Troubleshooting

Common failure modes, their symptoms, and how to fix them. For a full diagnostic command reference, see the [Diagnostic Commands](#diagnostic-commands) table at the bottom.

---

## 1. Stack Not Starting

**Symptom:** `docker compose up -d` exits early or services stay in "starting" / "unhealthy" state.

**Diagnosis:**
```bash
docker compose ps            # which services are unhealthy?
docker compose logs postgres  # check the first failure in the dependency chain
docker compose logs qdrant
docker compose logs redis
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| `POSTGRES_PASSWORD` not set in `.env` | Add `POSTGRES_PASSWORD=<value>` to `.env` and rebuild |
| `CHECKPOINTER_POSTGRES_DSN` password mismatch | Ensure the DSN password matches `POSTGRES_PASSWORD` |
| Port already in use on host | Another service holds 3000 or 8001; stop it or change the host port in `docker-compose.override.yml` |
| Insufficient disk space | Docker build and Qdrant storage both need headroom; free at least 5 GB |
| Missing `.env` file | Copy `.env.example` to `.env` and fill in required values |

**Health-check dependency order:** `postgres → qdrant → redis → duckduckgo-mcp → langgraph → fastapi → nextjs`. A failure at any point in this chain prevents the services that depend on it from starting.

---

## 2. Model / LLM Connection Failures

**Symptom:** LangGraph starts but chat returns errors like `ConnectionError`, `timeout`, or `model not found`.

**Diagnosis:**
```bash
poetry run steuermann setup doctor --probe-endpoints
docker compose logs langgraph | grep -i "error\|exception\|connection"
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| LM Studio / Ollama not running on host | Start the LLM server before starting the Docker stack |
| `host.docker.internal` not resolving (Linux) | Add `extra_hosts: ["host.docker.internal:host-gateway"]` to the service in `docker-compose.override.yml` |
| Wrong `api_base` in profile overlay | Check `config/profiles/<profile_id>/core.yaml` → `llm.roles.chat.api_base`; it must be reachable from inside the container |
| Model not loaded in LM Studio | Load the model before the stack; LM Studio returns 404 or empty model list if nothing is loaded |
| `OPENAI_API_KEY` not set | Set to `lm-studio` for local providers; OpenRouter needs the real API key |

**Reprobe after fixing:** `POST /api/llm/reprobe` (via Settings page) or restart `langgraph` to trigger the startup probe.

---

## 3. Tool Not Being Selected

**Symptom:** The model answers without using a tool that should have been selected (e.g. ignores a URL, doesn't call `calculator_tool` for a math question).

**Diagnosis:** Check the LangGraph logs for the `prefilter_tools` node output:
```bash
docker compose logs langgraph | grep -i "prefilter\|similarity\|intent\|tool_selection"
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| Query similarity below 0.55 threshold | The query doesn't match the tool description closely enough; rephrase or lower `similarity_threshold` in the profile overlay |
| Spread-gate failure | The top tool score is high but the second-best is too close; the gate suppresses selection; lower `min_spread` in `tool_routing` config |
| Intent boost not triggering | The boost keywords must appear in the message; check the [Quick Reference table](tools.md#quick-reference) for the exact trigger patterns |
| `tool_calling_mode` downgraded to `structured` | If the capability probe is missing or stale, native mode is disabled; use `POST /api/llm/reprobe` to refresh |
| Tool disabled in profile | Check `config/profiles/<profile_id>/tools.yaml` — the tool may be set to `enabled: false` |

---

## 4. Vision Tool Not Working

**Symptom:** Image is in the message or attachment but `analyze_image_tool`, `ocr_tool`, `analyze_document_tool`, or `analyze_chart_tool` is not called, or the tool is called but returns an error.

**Diagnosis:**
```bash
docker compose logs langgraph | grep -i "vision\|analyze_image\|ocr"
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| `llm.roles.vision` not in profile overlay | Add a `vision` role block to `config/profiles/<profile_id>/core.yaml`; without it, the runtime falls back to the `chat` role which may not accept image input |
| Attachment path not resolving | The attachment must be uploaded via the chat attachment system; direct file paths outside the workspace root are rejected |
| Image larger than 10 MB | Resize or compress before uploading |
| Model does not support vision | Verify the loaded model supports image input; check `/api/llm/capabilities` → `supports_vision` |
| Tool not in `_VISION_LLM_TOOLS` set | Only the four vision tools (`analyze_image_tool`, `ocr_tool`, `analyze_document_tool`, `analyze_chart_tool`) route to the vision LLM role; `image_metadata_tool` and `read_barcodes_tool` do not |

---

## 5. Memory Not Persisting

**Symptom:** Memories are not saved between sessions, or `GET /api/memories` returns empty results after a conversation.

**Diagnosis:**
```bash
docker compose logs langgraph | grep -i "memory\|mem0\|update_memory"
docker compose logs qdrant
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| `CHECKPOINTER_POSTGRES_DSN` not set | Set in `.env`; without it, the `PostgresSaver` cannot start and session state is lost |
| No `conversation_id` in request | Ephemeral requests (no `conversation_id`) don't write checkpoint rows; pass a `conversation_id` via the UI or API |
| Qdrant collection not initialised | On first run, Mem0 creates the collection automatically; if Qdrant was restarted with data loss, send one chat message to trigger re-creation |
| `memory.mem0.infer_enabled` is false | Memory extraction runs only when `infer_enabled: true` in the profile overlay |
| Embedding dimension mismatch | If you changed the embedding model, recreate Qdrant collections; dimension mismatch causes silent write failures |

---

## 6. RAG Not Returning Results

**Symptom:** Responses don't reflect document content that was ingested, or ingestion reports success but retrieval returns nothing.

**Diagnosis:**
```bash
docker compose logs langgraph | grep -i "retrieve_knowledge\|rag\|qdrant\|score_threshold"
docker compose logs qdrant
```

As an admin, the fastest interactive check is the **RAG knowledge explorer** at `/admin/rag`:
search your keyword and see whether the expected chunks come back, at what score, and whether they
fall above or below the production cutoff line. No matches points to a collection mismatch or
missing ingestion; matches that all sit *below* the cutoff points to a `pill_score_threshold` that is
too high (or a since-changed embedding model).

**Common causes:**

| Cause | Fix |
| --- | --- |
| `rag.collection_name` mismatch | The profile overlay's `rag.collection_name` must exactly match the collection used during ingestion; check both |
| `pill_score_threshold` too high | Lower `rag.pill_score_threshold` in the profile overlay (default 0.72); chunks with lower similarity are filtered out |
| Qdrant not healthy | Check `docker compose logs qdrant`; collection list: `curl http://localhost:6333/collections` |
| RAG disabled in config | Two profile-level switches gate retrieval: `rag.enabled` in `config/profiles/<id>/core.yaml` and the `rag_retrieval` flag in `config/profiles/<id>/features.yaml` — set either to `false` and the retrieve node is skipped (base `config/features.yaml` does **not** hold this flag) |
| Embedding model changed since ingestion | Re-ingest after changing the embedding model; existing vectors are not compatible |
| Documents not ingested | Verify ingestion ran: `docker compose logs ingestion` or run `poetry run steuermann ingest ingest --source ./data/rag-data --collection <name>` |

---

## 7. Redis Cache Not Working

**Symptom:** Repeated identical queries are not faster; logs show cache misses every time or Redis connection errors.

**Diagnosis:**
```bash
docker compose logs langgraph | grep -i "redis\|cache\|connection"
docker compose logs redis
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| `REDIS_URL` not set | Set `REDIS_URL=redis://redis:6379/0` in `.env` |
| Redis container not healthy | `docker compose ps redis`; restart if needed |
| In-memory fallback active | The framework falls back to in-memory cache when Redis is unreachable — check startup logs for `"Redis unavailable, using in-memory cache"` |
| Transient connection errors | Isolated `closed transport` errors are usually transient; only investigate repeated bursts |

> **Note:** Cache TTLs (`CACHE_LLM_TTL`, `CACHE_MEMORY_TTL`, `CACHE_SUMMARY_TTL`) are currently hardcoded in `universal_agentic_framework/caching/manager.py` (LLM: 86400 s, memory: 3600 s). To change them, update the code defaults and rebuild the `langgraph` image.

---

## 8. Configuration Validation Errors

**Symptom:** `steuermann config validate` fails, or the LangGraph service refuses to start with a config error.

**Diagnosis:**
```bash
poetry run steuermann config validate --profile <profile_id> --format json
poetry run steuermann config contract-check --format json
```

**Common causes:**

| Cause | Fix |
| --- | --- |
| Required field missing from profile overlay | The validator output names the missing field and its expected location |
| Disallowed profile override attempted | Some keys are protected and can only be set at the base level; the contract-check output lists which keys are disallowed in profile overlays |
| Unresolved environment variable substitution | A `$VAR_NAME` reference in a YAML file has no corresponding env var set; add the value to `.env` |
| Wrong schema (provider-registry vs flat per-role) | The runtime uses the flat per-role schema (`llm.roles.chat.api_base`); the provider-registry schema (`llm.providers.<name>`) is not active — see `CLAUDE.md` for the authoritative schema |
| `base` profile activated | `base` is repository defaults only; set `PROFILE_ID` to a runnable profile (e.g. `starter`) |

---

## Diagnostic Commands

| Command | What it checks |
| --- | --- |
| `poetry run steuermann setup doctor --probe-endpoints` | LLM, Qdrant, Postgres, Redis endpoint reachability |
| `poetry run steuermann config validate --profile <id> --format json` | Schema validation for the active profile overlay |
| `poetry run steuermann config contract-check --format json` | Contract parity (allowed overrides, required fields) |
| `poetry run steuermann config show --profile <id>` | Merged config after overlay application |
| `poetry run steuermann config explain --key <key>` | Where a config value comes from (base, overlay, env) |
| `docker compose logs -f langgraph` | Full graph execution trace including tool selection |
| `docker compose logs -f fastapi` | HTTP request logs, circuit-breaker events, workspace intent |
| `docker compose ps` | Health status of all services |
| `curl http://localhost:6333/collections` | Qdrant collection list (requires exposing port) |
| `curl http://localhost:8001/api/llm/capabilities` | Active LLM capability probe results |

---

## Related Documentation

- **[quickstart.md](quickstart.md)** — getting the stack running from a fresh clone
- **[configuration.md](configuration.md)** — full configuration reference
- **[deployment_guide.md](deployment_guide.md)** — Docker topology, security, and service wiring
- **[monitoring.md](monitoring.md)** — Prometheus metrics and operational alerts
