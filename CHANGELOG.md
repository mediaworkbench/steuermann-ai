# Changelog

## [0.5.1] — components updates

- chore: model IDs are now the provider's own IDs (e.g. `google/gemma-4-e4b`) without LiteLLM's `openai/` prefix; stored model preferences were reset.
- chore: replaced LiteLLM with a small OpenAI-compatible client (langchain-openai); every provider and gateway is reached over its OpenAI-compatible API, images are about 250 MB smaller, and the `llm.router` section is replaced by `llm.max_retries`.
- chore: upgraded the frontend's charts (Recharts 3), tables (TanStack Table 9), icons (Lucide 1) and accessibility test tooling; decorative icons are now hidden from screen readers.
- security: the web search server (DuckDuckGo MCP 0.7) no longer fetches internal or private addresses, so a model-chosen URL can't reach the app's own services; search results keep full URLs.
- chore: upgraded the MCP client (mcp 2); web search and page fetching keep working with existing MCP servers.
- chore: upgraded the Redis client (redis-py 8) and the image library (Pillow 13).
- security: upgraded the map library (MapLibre 6) and math rendering (KaTeX 0.19), clearing all known frontend dependency vulnerabilities.
- fix: the dashed distance line on maps is drawn again — theme colors are now converted to a format the map library understands.
- chore: upgraded the infrastructure to PostgreSQL 18, Redis 8.10 and Qdrant 1.20, the services to Python 3.13 and Node 24 LTS, and FastAPI to 0.162 / uvicorn 0.54. Existing deployments need a one-time Postgres dump/restore (see the deployment guide).
- chore: updated dependencies within their current major versions (LangGraph, LangChain, Mem0, Pydantic, Next.js, React, Tailwind) and refreshed the Python, Node and Prometheus images.

## [0.5.0] — security fixes and bugs

- fix: large attachments and workspace documents are no longer cut to their first ~2,400 characters — the budget now scales with the model's context window, and oversized files keep their beginning and end (so the newest entries of logs and exports stay visible), clearly marked as an excerpt.
- fix: document size is now estimated per content type, so number-heavy JSON and CSV files no longer overflow the model's context; file content is also sent once instead of twice.
- improvement: the spreadsheet tool now also computes over JSON data files (lists of records), so totals and counts work over the whole file even when only an excerpt fits the prompt.
- fix: the "Retry" model reprobe and the reprobe after switching models failed with a server error, so the model's real context window was never refreshed.
- chore: removed CrewAI and its never-enabled crews, together with the `agents.yaml` profile file, the crew feature flags and the crew-only result cache; multi-step workflows belong in LangGraph.
- fix: streamed answers now match what the model's history holds — the error message when the model provider fails mid-chat (previously a blank answer), filtered untrusted links, and retried answers now show up live and after reload.
- fix: a server-side failure mid-reply no longer saves the partial answer into a workspace document, and the turn is kept marked as interrupted instead of as a finished answer.
- fix: the chat circuit breaker can no longer get stuck rejecting every message after its recovery probe was stopped or answered with a client error.
- fix: background work after each reply (memory update, history compression, trace write-back) can no longer be dropped by the server's garbage collector.
- security: the web proxy no longer forwards client-supplied login rate-limit headers, which let a signed-in user bypass the login brute-force limits.
- security: vision tools only fetch images from public addresses (each redirect re-checked) and stop downloading at the size limit, so a URL can no longer reach internal services or exhaust memory.
- security: tools that open local files only accept files attached to the current conversation, so they can no longer read another user's uploads.
- security: removed the model-facing `file_ops_tool` — it was meant to be disabled, but came back when tools stopped having an on/off flag; it could read and overwrite the app's own source and config, shared by all users. Workspace documents cover the use case.
- improvement: preferred-model validation caches the provider's model list for a minute instead of querying it on every message.
- fix: a reasoning model's inline thinking is no longer kept in the chat history sent back to the model (or fed into memory), and a reply that comes back empty or as thinking only now gets the retry/fallback instead of a blank answer.
- fix: a tool entry that still carries the removed `enabled` flag is now rejected with a clear error instead of silently loading the tool.
- improvement: removed the memory-cache graph steps, which never took effect (memory was always re-queried); cache keys now use the full prompt/query, so long prompts sharing a prefix no longer collide.
- fix: the Redis cache keeps a separate connection per event loop, so cache calls from worker threads no longer break or leak the main connection; a user can no longer be "dreamed" twice back to back.
- fix: restoring a document version now detects concurrent edits like a normal save, text downloads always match the saved version, re-attaching a document no longer duplicates it, and deleting a conversation removes its workspace files.
- improvement: most conversation, workspace and memory endpoints no longer block the server's event loop, and the active-sessions metric now counts recently active users instead of everyone since startup.
- chore: removed unused feature flags (UI toggles, importance scoring) and unused helpers; consolidated the auth-enabled check.
- fix (UI): the provider-health poll stops after logout, the answer-evidence drawer no longer steals focus, citation links with spaces or parentheses render correctly, the Inspector lists the learned-preferences step, the workspace panel no longer shows an older answer's evidence as the latest, and the stream parser tolerates more SSE formats.
- fix: long conversations now load their newest messages (the view and the "recent document" lookup used to read the oldest ones), and an explicitly attached document no longer pulls an older one into the turn.
- fix: conversations whose first turns ran while the model provider was offline now still get an automatic title.
- fix: stopping a reply or an upstream error no longer loses the turn — the question and any partial answer are saved (marked as stopped); streaming failures now also trip the chat circuit breaker.
- fix: Regenerate and Edit-and-resend now replace the old turn on the server and in the model's history, instead of only hiding it in the UI.
- fix: sending in one chat while a reply is still streaming in another no longer kills that reply — the new message queues behind it, and a queued follow-up stays in the chat it was typed in.
- fix: clearing a document's text can now be saved, unsaved document edits are saved when leaving the chat page, and stale warnings no longer reappear.
- fix: upload size limits now also apply to document edits and reject oversized files early; the database pool waits briefly for a free connection instead of failing under load.
- fix: heartbeat task cooldowns now hold for their full duration (a task could previously run every other beat).
- fix: chat checkpoint storage no longer grows with every turn; conversation compression keeps the earlier summary instead of dropping it.
- fix: memory no longer stores duplicates when restating a known fact or a "No user facts" placeholder; memory analytics no longer show stale values with memory off, and turning memory off also skips the background fact extraction.
- fix: the dreaming engine no longer turns every preference into a new belief (or marks a cycle done) when the embedding provider is down, resumes capped batches instead of skipping the rest, catches contradictions of older same-run preferences, and ignores JSON inside a reasoning model's thinking.
- fix: knowledge-base results are always ordered by relevance, re-ingesting a file no longer removes it from search when embedding fails.
- security: re-indexing the whole knowledge base (`POST /api/ingestion/reingest-all`) now requires an administrator; any signed-in user could previously wipe and rebuild the shared RAG collection.
- security: login rate limiting is now per client and per username instead of one bucket shared by everyone behind the web proxy (which let five bad attempts lock every user out); a throttled login now says so instead of "invalid password".
- security: closed an open redirect after login (`/login?next=/\evil.com` could send users off-site).
- privacy: "reset my data" / admin reset now also clear the Dreaming Engine state (audit log with deleted-memory text, conflicts, learned rules, run history) and the model-side chat history checkpoints; deleting a conversation now removes its checkpoint history too.
- fix: memory-review undo and conflict resolution are now atomic and report failures — a double-click no longer applies an action twice, an outage no longer silently "succeeds", and undoing a preference anchor now actually reverts it.
- security: Postgres and Redis are no longer published on the host by default (use the override file, now bound to localhost), `PROFILE_ID` must be set explicitly instead of defaulting to the unrunnable `base`, and the checkpointer DSN now uses the configured database password.
- security: constant-time shared-token comparison, stricter workspace path confinement, and the browser no longer sends a (never-configured) shared-token header that could have leaked the secret into the JS bundle.

## [0.4.9] — cognitive memory architecture

- groundwork: cognitive-memory foundation — long-term memories now carry a tier tag (episodic/semantic), a confidence score, and a last-accessed timestamp, with legacy memories normalizing automatically; behaviour is unchanged until the new feature flags are enabled.
- groundwork: added the per-user data stores for learned preferences, memory-conflict resolution, and a reversible audit log that back the upcoming Dreaming Engine; not yet wired into any user-facing flow.
- groundwork: optional blended memory retrieval that surfaces consolidated "semantic" memories ahead of raw episodic ones, and records when a memory is actually used so stale ones can later be forgotten; off by default, identical to today until enabled.
- groundwork: a per-user background "dreaming" process that, once enabled, forgets old never-used memories and flags contradictions between new and established facts for you to resolve — privacy-isolated to one user at a time and degrading gracefully when the model provider is offline; off by default.
- groundwork: the dreaming process now also consolidates recurring memories into a single higher-level "semantic" memory it synthesizes, keeping the originals as provenance; runs on a configurable cadence with per-run caps, off by default.
- groundwork: the dreaming process can now learn formatting and style preferences from how you interact and, once you approve them, fold them into the assistant's persona; core-logic/safety rules are never auto-learned. Off by default; nothing reaches the prompt without explicit approval.
- feature: a Memory Review page (in the user menu) to resolve memory contradictions, approve or reject learned preferences, and undo recent memory changes within a 7-day window; plus an admin Dreaming Metrics dashboard showing anonymized, aggregate engine health (no user content).
- fix: the background memory consolidation and preference-learning steps now use a larger model token budget (and strip inline reasoning), so reasoning-style local models reliably return a result instead of an empty one. Found during end-to-end validation against a live model.
- feature: the cognitive memory + dreaming engine is now **enabled in the starter profile** and validated end-to-end against a live model — chat tags and consolidates memories, the background engine forgets/reconciles/consolidates on its schedule, and the Memory Review + admin pages drive it. Disable per profile via the `cognitive_memory_enabled` / `dreaming_engine_enabled` / `procedural_overrides_enabled` flags.
- feature: admins can now set each heartbeat task's cooldown (how often a per-user task like dreaming runs for a given user) directly on the `/admin/heartbeat` page — applied by the scheduler within ~30 seconds, no rebuild; setting a task back to its configured value clears the override.
- improvement: the admin heartbeat run log now shows the last 24 hours of beats (instead of a fixed count) and paginates 25 rows per page.
- docs: README, architecture, and configuration references updated to document the cognitive memory + dreaming engine and the runtime-configurable heartbeat controls; internal review pass hardened the dreaming engine's provider-reachability handling for the (non-default) concurrent-ticks configuration.
- fix: the dreaming engine no longer logs an error each beat for a user who has no memories yet (or a fresh deployment before anyone has chatted) — a missing memory store is now treated as "nothing to do" and the run records OK instead of failing.
- fix: contradiction detection now works reliably with reasoning-style local models — the contradiction check was using too small a token budget, so the model ran out of room while "thinking" and returned no verdict, silently producing zero conflicts; raised to match the other engine steps.

## [0.4.8] — intent-detection fixes, tool-routing polish, auth hardening & heartbeat per-user fan-out

- feature: heartbeat tasks can now run per user — a task marked `per_user` fans out once per active user each beat, drained by a bounded worker pool so a large user set never blocks the beat; global system tasks still run once. Run history is per-user and pruned on a retention window.
- feature: admin heartbeat inspector on its own `/admin/heartbeat` page (linked from the sidebar) — see the configured tasks (scope, cooldown, last run) and the run log (last 50 beats, filterable by task, user, and status) with error detail on expand.
- security: `POST /api/admin/reset-all-databases` now requires an administrator — it previously inherited only the shared-secret guard, so any authenticated user could wipe every user's data through the proxy.
- security: login and change-password are now rate-limited (5/min and 10/min, IP-keyed for unauthenticated callers) to blunt password brute-forcing.
- security: the API perimeter now fails closed — with authentication enabled but `CHAT_ACCESS_TOKEN` unset, requests are rejected (503) and a startup CRITICAL is logged, instead of leaving identity headers spoofable.
- feature: real logout and session revocation via a per-user token version — logging out, changing a password, or an admin changing a user's role/status/password immediately invalidates that user's existing sessions (no waiting for the 7-day token to expire).
- feature: optional `SESSION_EPOCH` — when set, a redeploy that changes it forces all sessions to re-login (an opt-in lever for "rebuild logs everyone out"; secret rotation remains the no-config alternative).
- fix: a greeting-prefixed substantive question (e.g. "Hi, what's our refund policy?") no longer skips knowledge retrieval; the greeting RAG short-circuit now only fires for messages that are essentially just a greeting.
- fix: a missing webpage URL is now backfilled from the user's message in all three tool-calling modes (native, structured, react), not only native — webpage extraction is consistent regardless of the model's calling style.
- improvement: the score-spread routing gate now compares the top tool to the runner-up instead of the average, so a single clearly-best tool is no longer discarded when the other candidates are bunched together.
- improvement: the structured-mode score at which a tool call is forced is now configurable per profile (`tool_routing.force_tool_use_score`, default 0.75).
- chore: removed dead code in the intent layer (an unused calculator-expression helper, a redundant image-flag alias, and a vestigial tool `enabled` flag).

## [0.4.7] — composer UX polish, heartbeat, smarter conversation titles & weather tool

- feature: weather tool — current conditions, two-place temperature comparison, and multi-day forecast via Open-Meteo, shown as an inline weather widget; units (°C/°F) configurable per profile; available to all roles and admin-gateable like other tools.
- feature: smarter conversation titles — a concise title is generated by the auxiliary model after the first exchange and upgraded once after a few turns using the fuller context; manual renames are never overwritten.
- feature: heartbeat — a virtual cron embedded in the orchestration service that wakes on a schedule and runs background tasks through an observe→reason→act tick (no-op scaffolding for now); every beat is recorded for observability.
- feature: admins can set the heartbeat beat rate (minutes) on the admin page; the scheduler applies the change within ~30 seconds.
- feature: memory toggle (Brain icon) in chat composer toolbar — disables both `load_memory` and `update_memory` nodes; persisted to user settings so it survives reload, mirroring the RAG toggle; does not affect the system-level `long_term_memory` feature flag.
- fix: the LangGraph edge rebuilt graph state from a fixed key allowlist that had drifted, silently dropping `allowed_tools`, `memory_enabled`, and `workspace_writeback_document` from every request — role tool gating failed open, the new memory toggle was ignored, and AI writeback never injected its prompt. All three are now forwarded.
- fix: tools icon in composer turns blue when one or more tools are active, matching the RAG toggle visual pattern.
- fix: toast notifications now appear at bottom-right.

## [0.4.6] — one-command first-time setup + idempotent RAG ingestion

- fix: RAG ingestion now uses deterministic, source-relative chunk IDs so re-running ingest or restarting the watcher never produces duplicate vectors; watcher and reindex share the same `/data/rag-data` mount path.
- feature: `steuermann setup init` — interactive wizard that writes `.env`, generates secrets (postgres password, session secret, access token, argon2id admin hash), configures an LLM provider, scaffolds a profile copy if needed, and runs pre-flight validation.
- fix: `.env` value parser now correctly handles double-quoted values with inline comments and single-quoted values containing `$` signs.

- feature: `steuermann setup init` — interactive wizard that writes `.env`, generates secrets (postgres password, session secret, access token, argon2id admin hash), configures an LLM provider, scaffolds a profile copy if needed, and runs pre-flight validation.
- fix: `.env` value parser now correctly handles double-quoted values with inline comments and single-quoted values containing `$` signs.

## [0.4.5] — multi-user accounts & role-based access

- feature: real multi-user authentication with three roles (user / researcher / administrator), argon2id login, and a bootstrapped admin seeded from env config.
- feature: admin user management — create, edit role/status, reset passwords, delete accounts with guardrails against self-demotion and removing the last admin.
- feature: per-user data isolation for conversations, settings, memories, and workspace documents; rate limiting now buckets per authenticated user.
- feature: role-based tool access — admins set a per-role allowlist, users toggle within that set, enforced server-side in the graph's load_tools node.

---

## [0.4.4] — inspector: full traceability of post-response nodes

- feature: Inspector now shows post-response nodes (compress / summarize / update_memory / cache_stats) with real timing and status, persisted per turn and surviving reload.
- feature: "Running in background…" placeholder for the latest answer; auto-polls until the drain completes without a manual reload.

## [0.4.3] — orchestration audit: graph fixes, performance, modularization

- fix: version history panel never appeared unless the editor was already open for that document.
- fix: `/invoke` 500'd on every request (sync call on async Postgres checkpointer); switched to `ainvoke`.
- fix: ephemeral requests (no `session_id`) raised a missing-configurable error; now run on a throwaway thread cleaned up after the run.
- fix: stale per-turn channels (tool results, crew results, RAG cache) leaked across turns; all per-turn state now resets on each request.
- fix: analytics crew output prefix mismatch caused results to leak back into conversation history.
- fix: summarizer failure wrote the extraction prompt into long-term memory instead of yielding empty.
- fix: native tool batch retry re-executed tools that already succeeded in a prior attempt.
- fix: URL anti-hallucination filter false-positived on user-pasted URLs and trailing-punctuation variants.
- perf: config loading cached (~102× faster per load); tool discovery cached (53.9 ms → 0.001 ms warm); performance nodes made async-native.
- refactor: respond node modularized into sub-packages; crew nodes reduced 54% via declarative registry; logging standardized on structlog kwargs style.

## [0.4.2] — workspace editing, writeback hardening, compression repair

- feature: per-user appearance settings — light / dark / system theme preference and metrics panel visibility toggle.
- fix: conversation compression repaired (four bugs: wrong LLM method, silent truncation on failure, summary digest dropped from prompt, manual compact used random UUID that broke checkpoint ordering).
- feature: auto-compression threshold now uses actual context window (0.75×) instead of the output cap; fill measured from provider-reported prompt tokens.
- feature: context-window ring shows per-model window size; switching models updates the denominator live.
- feature: `csv_analyze_tool` with German delimiter detection, numeric coercion, and intent boost when a CSV workspace doc is present.
- feature: dirty tracking for the document editor; auto-saves before sending so the model always reads the persisted version.
- feature: optimistic locking (409 on version conflict) for workspace document edits and AI writeback.
- feature: version origin badges (user / AI / restored) in the document history panel.
- feature: compact writeback SSE — chat bubble shows a summary while the document saves silently; `writeback` event carries the exact persisted content.
- fix: images blocked from edit, restore, and writeback paths.

## [0.4.1] — workspace split-view, answer-scoped panel, evidence drawer

- feature: workspace panel is answer-scoped — clicking a chip pins Knowledge / Memory / Outputs / Inspector to that answer; auto-resets to latest on a new turn.
- feature: MetricsPanel is now technical-only (timing / tokens / model); provenance lives in chips and panel tabs.
- feature: EvidenceChips is the single inline provenance summary under every assistant answer, replacing the old separate badge rows.
- feature: Outputs tab shows tool invocations as expandable cards with sanitized args and truncated results.
- feature: read-only evidence drawer on `/chats` showing the latest answer's evidence for any past conversation.
- feature: active document split-view pane as the sole editing surface; open document remembered per conversation.
- feature: documents list virtualizes past 50 rows; workspace toggle moved into the chat composer.

## [0.4.0] — shadcn/ui migration

- refactor: replaced all Material Symbols icons with lucide-react; migrated to shadcn/ui primitives (20 components); removed redundant Radix UI wrappers.
- feature: token contract for profile `ui.yaml` validation in both Python and TypeScript, checked by `steuermann config validate` and at runtime.
- feature: automated accessibility gates — `eslint-plugin-jsx-a11y` static analysis and `jest-axe` runtime assertions on all interactive primitives.
- fix: build failure on Linux from case-sensitive PascalCase component filenames (only affected case-sensitive filesystems).

---

## [0.3.9] — design-system foundation

- refactor: established shared UI primitives with Radix UI and Tailwind; replaced hardcoded hex colors with semantic CSS variable tokens for full light/dark parity.
- feature: Geist font; map and chart colors resolve dynamically to the active theme and profile.

## [0.3.8] — workspace panel evolution

- feature: workspace sidebar refactored into a modular tabbed panel (Documents / Knowledge / Memory / Outputs / Inspector).
- feature: unified evidence source (`deriveAnswerEvidence`) feeds all panel tabs and the inline EvidenceChips row from a single derivation.
- feature: Inspector tab shows graph execution trace with per-node status and timing, persisted per message and surviving reload.

## [0.3.7] — frontend improvements

- feature: LaTeX math rendering (KaTeX) and syntax-highlighted code blocks with copy button in assistant answers.
- feature: `normalizeMath` preprocessor protects code spans and escapes currency `$`-amounts before math parsing.
- feature: RAG knowledge explorer on `/admin/rag` — search chunks by keyword, see scores relative to the production threshold.
- feature: tokens/sec metric in MetricsPanel; `response_time_ms` now persisted so it survives a page reload.
- feature: in-flight stream survives conversation switching; returns live or shows the completed answer on switch-back.
- feature: `/chats` pagination and search are server-side with `pg_trgm` GIN indexes; search covers all conversations, not just the loaded slice.
- feature: queued follow-up messages — auto-fires on normal completion, stays queued on Stop or error.
- fix: streaming reasoning bar now spans full width from the first token.

## [0.3.6] — role-based frontend, map tool, data reset

- feature: role-based frontend — user surface (chat / memories / settings) and administrator surface (diagnostics / admin / metrics).
- feature: `map_tool` with Nominatim geocoding and an inline MapLibre widget; three operations: locate, distance, multi-pin.
- feature: `POST /api/user/reset-my-data` — per-category data deletion (conversations, workspace, memories).
- feature: image thumbnails in the workspace sidebar open a fullscreen lightbox; "Attach" button always visible and creates a conversation on the fly if none exists.
- fix: several `steuermann config` CLI flags had wrong example key paths or incorrectly included `base` in default profile lists; `config set --apply` now preserves key order and non-ASCII characters.
- fix: MapLibre v4 null comparison errors in style layer filter expressions.

## [0.3.5] — vision tools expansion

- feature: workspace sidebar accepts image uploads with lazy JPEG thumbnails.
- feature: five new vision tools — `ocr_tool`, `analyze_document_tool`, `analyze_chart_tool`, `image_metadata_tool`, `read_barcodes_tool`; shared vision helpers extracted to avoid duplication.

## [0.3.4] — auxiliary model expansion

- feature: RAG query rewriting with multi-variant expansion via the auxiliary model; enabled by default in the starter profile.
- feature: conversation auto-titling after the first exchange using the auxiliary model.
- feature: structured tool retry re-prompts now use the auxiliary model instead of the full chat model.

## [0.3.3] — vision model integration

- feature: `analyze_image_tool` — calls `llm.roles.vision` via direct httpx; accepts image URLs and uploaded attachment paths.
- fix: image upload returned 400 (binary decode); file path not forwarded to LangGraph; workspaces volume missing from the langgraph container.

## [0.3.2] — auxiliary model routing

- feature: summarization and RAG query rewriting now use the auxiliary model, freeing the chat model for user-facing generation.
- feature: auxiliary and vision LLM roles are now optional; missing roles skip gracefully with no fallback to chat.

---

## [0.3.1] — postgres checkpointing, reasoning UI

- feature: always-on Postgres checkpointing; SQLite path and `enabled` flag removed.
- fix: `GraphState.loaded_tools` (BaseTool instances) caused silent checkpoint write failures; excluded from serialization.
- feature: streaming chain-of-thought UI with collapsible ReasoningBox; `<think>` tags and native `reasoning_content` field both supported.
- fix: metrics panel fields lost after navigation; thumbs feedback not persisted for in-session messages (persistedId backfill added).
- fix: context ring denominator was `max_tokens` (output cap) instead of the model's actual context window.

## [0.3.0] — streaming chat composer

- feature: context-window ring indicator in the composer showing real-time fill percentage.
- feature: chat composer redesigned — auto-growing textarea, tools popover, model selector, RAG toggle.
- feature: early `metadata` + `[DONE]` emission after the respond node; summarize/memory nodes drain in the background.
- feature: auto-scroll pauses when user scrolls up; floating "scroll to bottom" button with unread count badge.
- fix: RAG `collection_name` warning false-positive when the key was correctly omitted from the profile.

## [0.2.9] — adaptive RAG and knowledge base toggle

- feature: intent-based RAG short-circuit for greetings, math, and datetime queries (saves ~50–80 ms per trivial turn).
- feature: per-session knowledge base toggle in the composer; `pill_score_threshold` separates display cutoff from the retrieval floor.
- fix: web search not forced in structured mode when explicitly requested; mandatory footer + retry loop added.
- refactor: `node_retrieve_knowledge` extracted to its own module; pure RAG helpers extracted to a separate utility module.

## [0.2.8] — workspace writeback quality, admin reset

- fix: workspace writeback classifier used async `ChatLiteLLM` which silently dropped `api_base`; switched to direct httpx.
- feature: writeback uses `SUMMARY:` / `DOCUMENT:` two-section format; the chat bubble shows the change summary.
- fix: version history endpoint caused 500 (raw datetime objects not accepted by Pydantic response model).
- feature: `POST /api/admin/reset-all-databases` — truncates user-data tables, deletes Qdrant collections, wipes files.

## [0.2.7] — workspace tool gold standard

- feature: hybrid LLM workspace intent classifier with EN+DE regex fallback; fires only when workspace documents are present.
- feature: version history for workspace documents — content snapshotted on every save, restorable via API.
- feature: full document content injected in writeback mode, bypassing the normal context truncation limit.

## [0.2.6] — tool system refactor and quality

- fix: `file_ops_tool` disabled — `sandbox_dir: ""` resolved to `/app` in Docker, exposing the entire app codebase.
- fix: `datetime_tool` convert_timezone now supports real cross-timezone conversions with an explicit `from_timezone` parameter.
- refactor: removed dead `node_route_tools` and ~450 lines of unreachable helper infrastructure that had been superseded by the three-layer tool system.

## [0.2.5] — probe hardening, session continuity

- fix: probe matching now requires exact `(provider_id, model_name)`; stale probes for the wrong model no longer activate native tool-calling mode.
- feature: `reprobe_for_model()` reprobes only matching providers on model change (not a full reprobe of all providers).
- fix: `POST /api/chat` now forwards `conversation_id` as `session_id` to the LangGraph checkpointer so turns share graph state.
- feature: "Memories used" count moved into MetricsPanel; removed the always-expanded memory list above every assistant message.

## [0.2.4] — active profile provider cutover

- break: all runtime provider, model, and ingestion config now lives in the active profile overlay; `PROFILE_ID` is required; `base` is not a runnable profile.
- feature: message-quality telemetry pipeline — Prometheus counter, analytics endpoint, and frontend quality panel.
- refactor: Mem0 adapter de-customized to canonical OSS API (filters-based search/list/delete, canonical get/delete/update signatures).

## [0.2.3] — provider endpoint consolidation

- break: `LLM_ENDPOINT` removed; replaced by per-provider env vars (`LLM_PROVIDERS_LMSTUDIO_API_BASE`, `LLM_PROVIDERS_OLLAMA_API_BASE`, `LLM_PROVIDERS_OPENROUTER_API_BASE`).
- feature: tool-calling policy moved to model-level config via `model_tool_calling` map; probe freshness enforced (stale or missing probes force structured mode).
- feature: `GET /api/llm/capabilities` exposes per-model effective mode, probe status, and capability metadata.

## [0.2.2] — provider model hardening

- fix: model validation derived the provider prefix from the active profile's default model instead of the requested model's own ID.
- feature: `LLM_CAPABILITY_PROBE_ENABLED` and `LLM_CAPABILITY_PROBE_ON_STARTUP` env vars added and documented.

## [0.2.1] — improved config flow

- feature: unified `steuermann` CLI with 16 commands across profile, config, setup, docs, and ingest groups.
- feature: profile bundling — export/import `.tar.gz` archives with schema and framework compatibility validation.
- feature: profile-safe `config set` / `config unset` with dry-run mode, `--apply --confirm APPLY`, and automatic rollback on validation failure.

---

## [0.2] — memory layer refactor

- perf: singleton-cache `Mem0MemoryBackend` eliminates redundant Qdrant round-trips per request.
- fix: Mem0 provider set to `lmstudio` to fix `json_object` 400 errors; `infer=True` now works natively.
- feature: `/memories` page with listing, stats, and per-memory rating UI; memory retrieval quality feedback loop with Prometheus counters.

## [0.1] — unify LLM handling

- feature: unified LLM integration via LangChain/LiteLLM; all `BaseTool` subclasses auto-wrapped as CrewAI tools.
- feature: crew results injected into system prompt as `=== RESEARCH FINDINGS ===`; date anchoring added to system prompt.

## [0.0] — initial

- feature: initial framework — LangGraph orchestration, FastAPI adapter, Next.js frontend, Docker Compose stack, Linux `APP_UID`/`APP_GID` support.
