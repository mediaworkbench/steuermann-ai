# Quickstart Guide

Get from a fresh clone to a working conversation in about 15 minutes.

---

## Prerequisites

| Requirement | Minimum version | Notes |
| --- | --- | --- |
| Docker + Docker Compose | Docker Engine 24+ | Compose plugin (V2) required; `docker compose` not `docker-compose` |
| Poetry | 1.7+ | Python package manager — `pip install poetry` if absent |
| Python | 3.11+ | Used by Poetry for CLI tools and local test runs |
| LLM endpoint | — | One of: LM Studio (local), Ollama (local), OpenRouter (cloud) |

The full stack runs in Docker. Python and Poetry are only needed on the host for the `steuermann` CLI and `pytest`.

---

## 1. Clone and Install

```bash
git clone <your-repo-url> steuermann-ai
cd steuermann-ai
poetry install
```

`poetry install` installs the `steuermann` CLI and all Python dependencies into an isolated virtual environment.

---

## 2. Configure Environment Variables

### Fastest path: `setup init`

One interactive command does the whole step — it writes a valid `.env`, generates strong
secrets (`POSTGRES_PASSWORD`, `AUTH_SESSION_SECRET`, `CHAT_ACCESS_TOKEN`) and an argon2id admin
password hash, constructs `CHECKPOINTER_POSTGRES_DSN`, creates (and on Linux chowns) the data
directories, and runs the pre-flight checks:

```bash
poetry run steuermann setup init
```

It walks you through provider/profile choice (LM Studio / Ollama / OpenRouter — endpoint, API key,
per-role models, and the embedding endpoint/model/dimension). If you customize the reference
`starter` profile it scaffolds a fresh profile copy and points `PROFILE_ID` at it (steps 3 and the
`AUTH_ENABLED=true` part of authentication are handled for you). The generated credentials are
printed once at the end — save them. See [cli.md](cli.md#steuermann-setup-init) for flags and
behavior. With the wizard you can skip to **step 4 (Start the Stack)**.

### Manual setup (fallback)

Copy the example file and fill in the required values:

```bash
cp .env.example .env
```

**Required variables** — the stack will not start without these:

| Variable | Example | Description |
| --- | --- | --- |
| `PROFILE_ID` | `starter` | Active profile overlay to load |
| `POSTGRES_PASSWORD` | `changeme` | PostgreSQL database password |
| `CHECKPOINTER_POSTGRES_DSN` | `postgresql://framework:changeme@postgres:5432/framework` | Full DSN for the LangGraph checkpointer |
| `EMBEDDING_SERVER` | `http://host.docker.internal:1234/v1` | OpenAI-compatible embedding endpoint |
| `LLM_PROVIDERS_LMSTUDIO_API_BASE` | `http://host.docker.internal:1234/v1` | LM Studio endpoint (if using LM Studio) |
| `LLM_PROVIDERS_OLLAMA_API_BASE` | `http://host.docker.internal:11434/v1` | Ollama endpoint (if using Ollama) |
| `OPENAI_API_KEY` | `lm-studio` | Placeholder key — set to `lm-studio` for local providers |

> **Linux hosts:** `host.docker.internal` is not automatically available. Add `extra_hosts: ["host.docker.internal:host-gateway"]` to the relevant services in `docker-compose.override.yml`, or use your host IP directly.

**Optional but recommended for first run:**

```bash
AUTH_ENABLED=false                      # Keep disabled for local development
NEXT_PUBLIC_AUTH_USER_ROLE=administrator  # Grants admin UI in dev without login
REDIS_URL=redis://redis:6379/0  # Already set in .env.example
```

### Provider-specific endpoint setup

**LM Studio** (recommended for local):
```bash
LLM_PROVIDERS_LMSTUDIO_API_BASE=http://host.docker.internal:1234/v1
EMBEDDING_SERVER=http://host.docker.internal:1234/v1
OPENAI_API_KEY=lm-studio
```

**Ollama**:
```bash
LLM_PROVIDERS_OLLAMA_API_BASE=http://host.docker.internal:11434/v1
EMBEDDING_SERVER=http://host.docker.internal:11434/v1
OPENAI_API_KEY=lm-studio
```

**OpenRouter** (cloud):
```bash
LLM_PROVIDERS_OPENROUTER_API_BASE=https://openrouter.ai/api/v1
LLM_PROVIDERS_OPENROUTER_API_KEY=<your-openrouter-api-key>
OPENAI_API_KEY=<your-openrouter-api-key>
```

---

## 3. Scaffold Your First Profile

The `starter` profile is the reference profile and is already included. For a custom profile:

```bash
poetry run steuermann profile scaffold --from starter --profile my-profile
```

This creates `config/profiles/my-profile/` as a copy of the starter overlay. Then validate it:

```bash
poetry run steuermann config validate --profile my-profile --format json
```

All fields should pass. Fix any errors before starting the stack.

---

## 4. Start the Stack

```bash
docker compose up -d
```

Services start in dependency order. The full health-check chain:

```
postgres → qdrant → redis → duckduckgo-mcp → langgraph → fastapi → nextjs
```

Startup takes ~30–60 seconds on first run (image pulls + LangGraph initialization). On subsequent starts it is faster.

Watch the orchestration service until it is ready:

```bash
docker compose logs -f langgraph
```

Look for a line like `INFO: Uvicorn running on http://0.0.0.0:8000` before proceeding.

---

## 5. Validate the Stack

Check that configuration is valid and all endpoints are reachable:

```bash
# Validate config against schema and contracts
poetry run steuermann config validate --profile starter --format json

# Probe all service endpoints (LLM, Qdrant, Postgres, Redis)
poetry run steuermann setup doctor --probe-endpoints
```

A healthy output from `setup doctor` shows green checkmarks for each service. If any probe fails, check the corresponding service logs:

```bash
docker compose logs postgres
docker compose logs qdrant
docker compose logs redis
```

---

## 6. First Conversation Test

Open the chat UI:

```
http://localhost:3000
```

Send a test message — e.g. `"What time is it?"` (triggers the `datetime_tool`) or `"Calculate 42 * 7"` (triggers `calculator_tool`).

If the response includes a tool result, routing is working. If it hangs or returns an error, check:

```bash
docker compose logs -f langgraph
docker compose logs -f fastapi
```

The LangGraph logs show the full graph execution, including which tools were selected and whether the LLM endpoint responded.

---

## 7. Next Steps

| Topic | Document |
| --- | --- |
| Full configuration reference | [configuration.md](configuration.md) |
| Creating a domain-specific profile | [profile_creation.md](profile_creation.md) |
| Available tools and routing | [tools.md](tools.md) |
| Docker topology, security, ingestion | [deployment_guide.md](deployment_guide.md) |
| CLI command reference | [cli.md](cli.md) |
| Metrics and observability | [monitoring.md](monitoring.md) |
| Troubleshooting | [troubleshooting.md](troubleshooting.md) |
