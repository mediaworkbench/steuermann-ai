# Performance Optimization Guide

This guide is intentionally short. It focuses on the runtime levers that matter in the current repository instead of hard-coded benchmark claims.

---

## Current Scope

Performance work in Steuermann currently centers on:

- Redis-backed caching with graceful fallback when Redis is unavailable
- Conversation compression and summarization to control token growth
- Token budgeting across conversation, turn, and node scope
- Prometheus metrics and targeted regression tests

This is an active tuning area. Exact gains depend on workload, prompt stability, model latency, and how aggressively profile overlays enable tools, crews, and retrieval.

---

## Main Levers

### 1. Cache Stability

Use Redis for the production cache path and treat the in-memory backend as a fallback, not the baseline.

Practical defaults:

- Keep Redis healthy before tuning anything else.
- Start with conservative TTLs and raise them only for stable prompts or repeated retrieval queries.
- Monitor cache errors and connection churn in the LangGraph logs before assuming a logic problem.

### 2. Token Control

The most reliable cost and latency reduction usually comes from token discipline:

- Lower retrieval fan-out before lowering model quality.
- Keep `response_reserve_ratio` intact so downstream nodes still have budget.
- Use conversation compression to bound long sessions instead of letting context grow indefinitely.
- Tune per-node budgets only after looking at real node-level token metrics.

### 3. Retrieval Cost

RAG cost grows with chunk count, `top_k`, payload size, and embedding latency.

Start here:

- Keep `rag.top_k` small unless the profile genuinely needs broad recall.
- Return only payload fields the runtime uses.
- Re-ingest after chunking changes so query behavior stays consistent.
- Keep ingestion and runtime embedding configuration aligned.

---

## Configuration Checklist

Review these settings first when the system feels slow:

```yaml
# config/core.yaml
rag:
  top_k: 5
  score_threshold: 0.6

tokens:
  default_budget: 10000
  per_turn_budget_ratio: 0.4
  response_reserve_ratio: 0.15
  enforce_per_node_hard_limit: true

memory:
  embeddings:
    batch_size: 32
```

```bash
# .env / container runtime
REDIS_URL=redis://redis:6379/0
CACHE_LLM_TTL=86400
CACHE_MEMORY_TTL=3600
CACHE_SUMMARY_TTL=604800
SUMMARIZE_MAX_TOKENS=4096
SUMMARIZE_MIN_MESSAGES=10
SUMMARIZE_KEEP_RECENT=5
```

---

## Operational Tuning Order

1. Confirm Redis, Qdrant, and the LLM endpoint are healthy.
2. Check `/metrics` before changing config so you know whether latency comes from LLM calls, retrieval, or cache misses.
3. Tune retrieval breadth and token budgets.
4. Tune cache TTLs and eviction behavior.
5. Re-test with the same workload slice.

Changing several levers at once usually hides the real bottleneck.

---

## Advanced And Experimental Areas

The following remain worth exploring, but should be treated as workload-specific tuning rather than defaults:

- More aggressive cache warming
- Profile-specific eviction strategy changes
- Conversation compression threshold changes for long-running sessions
- Crew-specific timeout and cache tuning
- Wider Prometheus-based performance regression coverage

---

## Troubleshooting

### High Latency With Low Cache Hits

- Check whether prompts are too dynamic to benefit from caching.
- Verify retrieval settings are not pulling unnecessary context.
- Confirm the slow path is not the upstream LLM endpoint.

### Cache Errors In Logs

- A closed transport or handler error usually indicates a transient Redis connection issue.
- If requests still complete, treat it as a resilience signal first, not an application correctness bug.
- Investigate repeated bursts, not isolated single events.

### Token Usage Keeps Growing

- Confirm summarization thresholds are reachable.
- Check whether attachments or retrieval payloads are dominating prompt size.
- Review node-level token metrics before tightening budgets blindly.

---

## Related Docs

- **[monitoring.md](monitoring.md)**
- **[configuration.md](configuration.md)**
- **[technical_architecture.md](technical_architecture.md)**
