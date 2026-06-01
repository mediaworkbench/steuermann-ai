# Deployment Guide

Operational reference for running, deploying, securing, and maintaining the Steuermann stack.

For core execution architecture (LangGraph graph, memory, LLM integration, tool selection), see [Technical Architecture](technical_architecture.md).

---

## **1. Ingestion Pipeline Architecture**

### **1.1 Service Design**

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

### **1.2 Ingestion Configuration**

Ingestion settings live under the `ingestion:` key in the active profile overlay (`config/profiles/<profile_id>/core.yaml`). There is no separate `ingestion.yaml` file.

```yaml
# config/profiles/starter/core.yaml
ingestion:
  source_path: $RAG_DATA_PATH          # Directory to ingest (env var or --source CLI override)
  language: "de"                        # Primary language for ingestion
  language_threshold: 0.8              # Minimum language detection confidence
  embedding_batch_size: 32             # Documents embedded per batch
  upsert_batch_size: 128               # Vectors upserted per Qdrant batch
  file_concurrency: 1                  # Parallel file processing workers
  incremental_mode: true               # Skip unchanged files (hash-based)
  phase_timing: true                   # Emit per-phase timing logs
  reingest_timeout_seconds: 1800       # Timeout for full reingest via API trigger

rag:
  collection_name: "framework"         # Must match collection used at ingest AND retrieval time
```

**Note:** Language validation accepts all documents and tags chunks with `detected_language`, `language_confidence`, and `target_language` metadata fields. All languages are accepted; none are rejected.

### **1.3 Ingestion CLI**

All ingestion commands go through the unified `steuermann` CLI. Source, collection, and language all default to values from the active profile's `core.yaml` and can be overridden per-run.

```bash
# One-shot ingestion (uses core.yaml defaults for source/collection)
poetry run steuermann ingest ingest

# Override source and collection for a single run
poetry run steuermann ingest ingest \
  --source /data/rag-data \
  --collection medical-ai-de-procedures \
  --language de

# Watch mode (auto-ingest new files, checks every 30 seconds)
poetry run steuermann ingest watch \
  --source /data/rag-data \
  --collection medical-ai-de-procedures

# Clear and reindex a collection
poetry run steuermann ingest reindex \
  --source /data/rag-data \
  --collection medical-ai-de-procedures

# Validate documents without ingesting (dry run)
poetry run steuermann ingest validate \
  --source /data/rag-data \
  --verbose
```

### **1.4 Language Detection & Tagging**

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

### **1.5 Collection Metadata Schema**

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

## **2. Frontend Architecture: Next.js + FastAPI + LangGraph**

### **2.1 Production Architecture**

The production stack separates concerns across three containers:

```text
┌─────────────────────────────────────┐
│  Next.js Frontend (Port 3000)       │
│  - Chat interface + conversations   │
│  - User profile & settings          │
│  - Analytics & metrics dashboards   │
│  - Role-based admin surface         │
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

### **2.2 Chat Interface**

The frontend provides 5 pages, 17 components, and 5 hooks — all fully wired to backend APIs.

**Pages:**

| Route | Purpose | Minimum role | Backend APIs |
| --- | --- | --- | --- |
| `/` | Chat with conversation persistence | user | `/api/chat`, `/api/conversations/*` |
| `/memories` | Personal memory management | user | `/api/memories/*` |
| `/settings` | Personal preferences: language, sound, tools, RAG on/off + top-K, chat model | user | `/api/settings/*`, `/api/models`, `/api/system-config` |
| `/admin` | LLM diagnostics, RAG collection/threshold, system model roles, re-ingest, danger zone | **administrator** | `/api/settings/*`, `/api/llm/*`, `/api/ingestion/*`, `/api/admin/*` |
| `/metrics` | Real-time system metrics and historical analytics trends | **administrator** | `/api/metrics`, `/api/analytics/*` |

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

### **2.3 UI Customization**

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

## **3. Monitoring & Observability**

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

## **4. Configuration Management**

Configuration uses hierarchical loading: **Base → Profile Overlay → Environment Variables**. All configs are validated by Pydantic schemas.

**Key files:**

- `config/core.yaml` — deployment-global defaults (database, memory, checkpointing)
- `config/profiles/<profile_id>/core.yaml` — runtime LLM providers, roles, router policy, RAG, token budgets, ingestion
- `config/profiles/<id>/agents.yaml` — CrewAI crew definitions
- `config/profiles/<id>/tools.yaml` — Tool enable/disable and overrides
- `config/features.yaml` — Feature flags with dependency validation

Profile customization is declarative — profiles override configs and register components, while direct core edits are avoided unless the shared template itself is being improved.

→ **Full schema reference and loading details:** [Configuration](configuration.md)

---

## **5. Docker Deployment Architecture**

### **5.1 Service Composition**

**Service Topology** (source of truth: `docker-compose.yml` at the repo root):

The stack runs 9 services on the `steuermann-network` Docker network:

| Service | Image / Build | Exposed port | Role |
| --- | --- | --- | --- |
| `langgraph` | `docker/Dockerfile.langgraph` | `8000` (internal only) | LangGraph orchestration engine |
| `fastapi` | `docker/Dockerfile.fastapi` | `8001:8001` (host) | FastAPI adapter — auth, metrics proxy, chat proxy |
| `nextjs` | `docker/Dockerfile.nextjs` | `3000:3000` (host) | Next.js frontend (chat, settings, metrics) |
| `ingestion` | `docker/Dockerfile.ingestion` | — | Qdrant document ingestion watcher |
| `postgres` | `postgres:15.5-alpine` | `5432:5432` | Conversation storage + LangGraph checkpoints |
| `redis` | `redis:7.2-alpine` | `6379:6379` | LLM response cache, memory cache, summary cache |
| `qdrant` | `qdrant/qdrant:latest` | `6333` (internal) | Vector store for RAG and Mem0 memory |
| `prometheus` | `prom/prometheus:v2.48.1` | `9090` (internal) | Metrics scraping (LangGraph `/metrics` every 15 s) |
| `duckduckgo-mcp` | `mcp/duckduckgo` | `8000` (internal) | DuckDuckGo web-search MCP server |

**Key wiring details:**

- `langgraph` connects to `postgres`, `qdrant`, `redis`, and `duckduckgo-mcp` (health-checked before start)
- `fastapi` connects to `langgraph` (health-checked) and `prometheus` for metrics proxying
- `nextjs` connects to `fastapi` (health-checked)
- `ingestion` connects to `qdrant` and the profile config; mounts `./data/rag-data` read-only
- LangGraph and ingestion both mount `./config` read-only; workspaces mount via `WORKSPACES_PATH`
- No `plugins/` volume — there is no plugins directory; tool registration is via `config/profiles/<id>/tools.yaml`
- Postgres connection uses split env vars (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`), not a single `DATABASE_URL`
- `CHECKPOINTER_POSTGRES_DSN` is set separately (defaults to `postgresql://framework:<pw>@postgres:5432/framework`)

### **5.2 Profile-Specific Overrides**

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

### **5.3 Host Network Configuration**

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

## **6. Security & Access Control**

**Service isolation:** LangGraph runs on internal-only network. FastAPI bridges internal/external. Next.js is user-facing only.

**Secrets:** Environment variables (`.env`, not committed). Docker Secrets supported for production.

**Authentication (opt-in):** Next.js login with signed HttpOnly session cookies, trusted proxy forwarding to FastAPI, and rate limiting (slowapi). Disabled by default via `AUTH_ENABLED=false`.

**Role-based access control:** A single `AUTH_USER_ROLE` env var (`user` | `administrator`, default `user`) sets the role embedded in the JWT at login time. The middleware (`proxy.ts`) blocks non-administrator sessions from `/admin` and `/metrics`. When `AUTH_ENABLED=false` (local dev), `NEXT_PUBLIC_AUTH_USER_ROLE` controls the client-side assumed role without requiring a login. For single-operator deployments, set both to `administrator`.

**Chat attachments and workspace documents (current implementation):**

- Conversation uploads are handled by `POST /api/conversations/{conversation_id}/attachments`.
- Text uploads are validated as UTF-8; accepted text formats: `.txt`, `.md`, `.markdown`, `.json`, `.yaml`, `.yml`, `.csv`, `.html`, `.xml`. Image uploads accepted as MIME types `image/jpeg`, `image/png`, `image/gif`, `image/webp` (extensions `.jpg/.jpeg/.png/.gif/.webp`); `extract_text()` is bypassed for images — `extracted_text` is stored as `""`.
- Image files are stored on disk under the workspace volume (`${WORKSPACES_PATH:-./data/workspaces}`); the same volume is mounted in both the `fastapi` and `langgraph` containers so the orchestrator can read uploaded files.
- `stored_path` is forwarded with each attachment from `chat.py` to LangGraph state; `build_attachment_context_block()` renders image attachments as file-path references in the system prompt, and `node_call_tools_structured` includes this block in its isolated `SystemMessage` so the model receives a real path rather than a placeholder.
- Conversation attachment rows are references to those canonical workspace documents (same document ID is reused).
- Persistent workspace document APIs: `POST/GET/PUT/PATCH/DELETE /api/workspace/documents`, `GET .../download`, `GET/POST .../versions`, `GET .../versions/{ver}`, `POST .../versions/{ver}/restore`.
- Version history: every `PUT` update auto-snapshots the current content to `workspace_document_versions` before overwriting; previous versions are listable and restorable via the API.
- Chat requests support both `attachment_ids` and `document_ids`; resolved workspace document content is injected into LangGraph state as `workspace_documents`.
- LangGraph injects both attachment context and workspace document context into prompt construction. In writeback mode the full document content (not truncated) is injected as a `HumanMessage` via `workspace_writeback_document` state field.
- Save-back intent is detected via a hybrid LLM classifier (`_classify_workspace_intent_llm`) — language-agnostic, fires only when relevant documents/attachments are in context, falls back to EN+DE regex. The classifier uses a direct `httpx` POST to the auxiliary provider (not `ChatLiteLLM.ainvoke()` which drops `api_base` in async context). When intent is confirmed and exactly one document is in context, the model produces a structured `SUMMARY:` / `DOCUMENT:` two-section response — the document content is stored as the new version and the summary is shown in the chat confirmation message (`workspace_document_writeback` in response metadata).

**Legacy workspace editing path (still present, opt-in):**

- Conversation-scoped workspace actions (`copy_to_workspace`, `read_workspace_file`, `write_workspace_file`, `write_revised_copy`) remain available behind `CHAT_WORKSPACE_ENABLED=true` and explicit edit intent.
- Filesystem operations are confined to configured roots with normalized path checks.

**Audit logging:** All data access logged to PostgreSQL `audit_log` table with user_id, action, resource_id, timestamp.

→ **Full details:** [SECURITY.md](../.github/SECURITY.md)

---

## **7. Development Workflow**

**Local setup:** `poetry install`, configure `.env`, `docker compose up -d`, `cd frontend && npm run dev`.

**Testing:** Unit tests (config, LLM factory, memory) + integration tests (full graph with mocked LLMs). Target: 70%+ coverage. Run via `poetry run pytest`.

**CI/CD:** This project does not use GitHub Actions or external CI. Run tests locally and rely on human code review.

→ **Full workflow (commands, PROJECT_LOG, testing strategy):** See [copilot-instructions.md](../.github/copilot-instructions.md)

---

## **8. Performance Considerations**

- **Token tracking:** `tokens_used`, `input_tokens`, `output_tokens` accumulated in state for Prometheus metrics and SSE metadata; automatic summarization when context grows beyond `max_tokens * 0.75`
- **Caching:** LLM response caching (optional), session-scoped memory query caching, Qdrant collection caching
- **Concurrency:** Container-based — each session gets independent graph execution in LangGraph service, horizontal scaling via additional containers

→ **Full details:** [Performance Optimization](performance_optimization.md)

---

## **9. Disaster Recovery & Backup**

### **9.1 Data Persistence**

**What needs backup:**

1. PostgreSQL (checkpoints, artifacts, audit logs)
2. Qdrant (vector collections)
3. Configuration files
4. Knowledge source files (if stored locally)

**What doesn't need backup:**

- Docker images (reproducible)
- LangGraph code (in git)
- Temporary caches

### **9.2 Backup Strategy**

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

### **9.3 Disaster Recovery Plan**

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

## **10. Migration & Upgrade Path**

### **10.1 Repository Upgrade Process**

Use a repository-first upgrade flow:

1. Review [README.md](../README.md) and the [Technical Architecture](technical_architecture.md) document for architecture or runtime changes.
2. Pull the target repository revision.
3. Compare profile overlays against updated defaults in `config/*.yaml` and `config/profiles/starter/`.
4. Rebuild the stack with `docker compose up -d --build`.
5. Run the relevant regression slice before promoting the revision.

### **10.2 Typical Compatibility Checks**

Focus on these surfaces after an upgrade:

- Configuration keys in `config/profiles/<profile_id>/core.yaml`, `config/features.yaml`, `config/profiles/<id>/tools.yaml`, and `config/profiles/<id>/agents.yaml`
- Prompt override file layout under `config/profiles/<id>/prompts/` and `config/profiles/<profile_id>/prompts/`
- Profile metadata surfaced through the FastAPI and chat metadata contracts
- Monitoring dashboards and metrics label expectations
- Ingestion and retrieval alignment for `rag.collection_name`, embeddings, and chunking settings

### **10.3 Breaking-Change Discipline**

When a revision changes a compatibility surface:

- update the starter profile if it is the new baseline
- update the affected docs pages in the same change
- rerun the targeted regression tests for the touched subsystem
- document the change in the relevant docs page and README release notes section

### **10.4 Minimum Validation**

```bash
poetry run pytest -q
docker compose up -d --build
```

### **10.5 Example Breaking Changes**

```markdown
# Example: profile compatibility review

1. Configuration key renamed or removed
2. Prompt file structure changed
3. Metrics label or dashboard assumptions changed
4. Chat metadata contract changed
```

### **10.6 Example Node Compatibility Change**

- `load_memory()` → `load_memory_node(state)`
- Action: update custom nodes or wrappers that still depend on the old call shape
