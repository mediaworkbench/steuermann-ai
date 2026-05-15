# Monitoring Guide

Complete guide for monitoring the Steuermann.

---

## Overview

The framework includes monitoring capabilities:

- **Prometheus metrics** for real-time performance tracking
- **Frontend metrics dashboard** at
  [Metrics dashboard](http://localhost:3000/metrics)
  with real-time and trends views
- **Structured logging** via `docker compose logs`
- **Attachment pipeline observability**
  (injection, no-attachment requests, refusal-retry guardrail)

---

## Quick Start

### 1. Start Services

```bash
# Start all services including monitoring
docker compose up -d

# Verify services are healthy
docker compose ps
```

### 2. Access Monitoring

- **Frontend Metrics**: [Metrics dashboard](http://localhost:3000/metrics)
  - Stats: total requests, tokens used, average latency, active sessions
  - Attachment stats: injected attachment contexts, no-attachment requests,
    retry triggers, retry success rate
  - Real-Time tab: token usage chart by profile/model,
    request status breakdown, memory ops, LLM call table
  - Trends tab: usage trends, token consumption, latency analysis, exportable CSV
  - Auto-refreshes every 10 seconds for real-time data

- **Prometheus**: [Prometheus](http://localhost:9090)
  - Direct access to metrics and PromQL queries
  - Scrapes LangGraph `/metrics` endpoint every 15s

---

### Frontend Metrics Dashboard

**URL**: [Dashboard](http://localhost:3000/metrics)

The Next.js frontend includes a built-in metrics dashboard that
queries Prometheus via the FastAPI adapter.

**Data flow**: Frontend → FastAPI `/api/metrics` → Prometheus PromQL →
LangGraph `/metrics`

**Panels**:

- **Real-Time**: Stats cards, attachment cards, token usage,
  request status, memory operations, LLM calls, retry guardrails
- **Trends**: Usage trends, token consumption, latency analysis,
  summary cards, CSV export

**Auto-refresh**:

- Real-Time tab: every 10 seconds
- Trends tab: optional 60 second refresh

---

## Metrics Reference

Note: Prometheus metric label keys still use the legacy label name
`fork_name`. The value should be the active `PROFILE_ID`.

### Graph Execution Metrics

```promql
# Total graph invocations by status
langgraph_requests_total{fork_name="starter", status="success"}

# Request duration histogram
langgraph_request_duration_seconds{fork_name="starter"}

# Active sessions gauge
langgraph_active_sessions{fork_name="starter"}
```

### Token Metrics

```promql
# Total tokens consumed
langgraph_tokens_used_total{fork_name="starter", model="gemma3:4b", node="respond"}

# Tokens per request
sum(rate(langgraph_tokens_used_total[5m])) / sum(rate(langgraph_requests_total{status="success"}[5m]))
```

### Node Metrics

```promql
# Node execution duration
langgraph_node_duration_seconds{fork_name="starter", node="respond"}

# Node execution rate
rate(langgraph_node_duration_seconds_count[5m])
```

### LLM Metrics

```promql
# LLM calls by provider and status
langgraph_llm_calls_total{fork_name="starter", provider="ollama", status="success"}
```

### Attachment Metrics

```promql
# Total turns where attachment context was injected into the prompt
langgraph_attachments_injected_total{fork_name="starter"}

# Total turns without attachments
langgraph_attachments_none_total{fork_name="starter"}

# Total times the refusal guardrail retry was triggered
langgraph_attachment_refusal_retries_total{fork_name="starter"}

# Total successful retries after refusal detection
langgraph_attachment_refusal_retries_success_total{fork_name="starter"}

# Retry success rate (guardrail effectiveness)
langgraph_attachment_refusal_retries_success_total / langgraph_attachment_refusal_retries_total
```

### Memory Metrics

```promql
# Memory operations by type
langgraph_memory_operations_total{fork_name="starter", operation="load"}

# Memory importance ranking duration
langgraph_memory_importance_ranking_duration_seconds{fork_name="starter"}

# Co-occurrence graph statistics
langgraph_memory_co_occurrence_graph_nodes{fork_name="starter"}
langgraph_memory_co_occurrence_graph_edges{fork_name="starter"}

# Memory quality score distribution
langgraph_memory_quality_score{fork_name="starter"}

# Related memories retrieved
langgraph_memory_related_retrieved{fork_name="starter"}

# Memory age in days
langgraph_memory_age_days{fork_name="starter"}
```

### Memory Retrieval Quality / Feedback Loop Metrics

These counters instrument the relationship between what the agent retrieves and how users subsequently rate it.

```promql
# Total memories served in chat context, by prior-rating state and rating bucket
# Labels: fork_name, rated (yes/no), rating_bucket (none/low/mid/high)
langgraph_memory_retrieval_signal_total{fork_name="starter"}

# Memories rated by the user shortly after being retrieved (feedback loop fired)
langgraph_memory_rated_after_retrieval_total{fork_name="starter"}

# Derived: feedback coverage (what fraction of retrieved memories get rated)
sum(langgraph_memory_rated_after_retrieval_total) /
  sum(langgraph_memory_retrieval_signal_total)

# Derived: prior-rating coverage at retrieval time
sum(langgraph_memory_retrieval_signal_total{rated="yes"}) /
  sum(langgraph_memory_retrieval_signal_total)

# Breakdown by rating bucket at retrieval
sum by (rating_bucket) (langgraph_memory_retrieval_signal_total)
```

**Dashboard**: These are surfaced in the **Metrics → Trends** tab under "Retrieval Feedback Loop".

**Single-surface rating model**: As of Phase 4, memory star ratings are collected exclusively from the **Memories page** (`/memories`). The chat view shows retrieved memories as read-only context with the current rating displayed as text; the interactive rating widget is intentionally absent from chat. This means `langgraph_memory_rated_after_retrieval_total` is incremented only when a user navigates to the Memories page and rates a memory that was recently retrieved in a conversation — which is the correct, unambiguous signal for retrieval quality feedback. Thumbs feedback in chat remains a separate message-quality signal and is not counted here.

### Message Quality Metrics

These metrics instrument thumbs up/down feedback on assistant messages separately from memory ratings.

```promql
# Total assistant-message feedback actions in chat
# Labels: fork_name, value (up/down/removed)
langgraph_message_feedback_total{fork_name="starter"}
```

**Dashboard**: These are surfaced in the **Metrics → Trends** tab under "Message Quality" via the backend analytics endpoint `/api/analytics/message-quality`.

**Signal boundary**: Message thumbs in chat reflect response quality for assistant messages. They do not modify memory ratings directly and are intentionally separate from retrieval-quality coverage metrics.

### Memory Health Thresholds

The following thresholds are recommended for operational alerting and manual review. The `/metrics` dashboard surface shows these as contextual data; configure Prometheus alert rules in `monitoring/alerts.yml` for automated firing.

| Signal | Healthy | Warning | Critical |
| --- | --- | --- | --- |
| Memory load success rate | ≥ 95% | 90–95% | < 90% |
| Memory load error rate | < 5% | 5–10% | > 10% |
| Average importance score | ≥ 0.5 | 0.3–0.5 | < 0.3 |
| Rated coverage (memories with user_rating) | ≥ 30% | 10–30% | < 10% |
| Feedback coverage (rated after retrieval) | ≥ 20% | 5–20% | < 5% |
| Prior rating coverage at retrieval time | ≥ 40% | 20–40% | < 20% |
| P95 importance ranking latency | < 50 ms | 50–100 ms | > 100 ms |

### Memory Retrieval Failure Triage

If `memory_load_failed` events appear in LangGraph logs or the memory load error rate crosses threshold:

1. **Check Qdrant availability** — `curl http://localhost:6333/healthz` from inside the `langgraph` container. If unavailable, restart the `qdrant` service.
2. **Check embedding model** — Verify the configured embedding endpoint (`memory.embeddings.remote_endpoint` in `core.yaml`) is reachable. Run a quick smoke: `curl -X POST <endpoint>/v1/embeddings -d '{"model":"...", "input":"test"}'`.
3. **Check Mem0 initialization** — Look for `api_key` or `llm.config.api_base` errors in the langgraph startup logs. At least one `LLM_PROVIDERS_*_API_BASE` env var must be set and reachable at LangGraph startup.
4. **Inspect null-score events** — If memories return with missing scores (visible as importance `0.0` in the dashboard), the Qdrant vector has no score attached. This is handled defensively (`score = float(item.get("score") or 1.0)`) but indicates Mem0 returned an unexpected shape.
5. **Check ownership filtering** — If a user sees 0 memories but Qdrant has data, the `filters={"user_id": ...}` in `Mem0MemoryBackend.load()` may not match. Confirm `AUTH_USERNAME` or the authenticated user ID matches the `user_id` stored during memory upsert.
6. **Rating persistence failures** — A `WARNING` log line containing `"mem0_rating_persist_signature_mismatch"` or `"mem0_rating_persist_failed"` indicates the adapter could not persist the canonical Mem0 `update(memory_id, data=..., metadata=...)` call. Ratings remain in the adapter's in-process fallback state for the current process lifetime, but the underlying Mem0 record was not updated. Investigate the running Mem0 SDK version and update API behavior.

---

## Example Queries

### Find Slow Requests

```promql
# 95th percentile response time
histogram_quantile(0.95,
  sum(rate(langgraph_request_duration_seconds_bucket[5m])) by (le))
```

### Calculate Error Rate

```promql
# Error rate as percentage
sum(rate(langgraph_requests_total{status="error"}[5m])) /
  sum(rate(langgraph_requests_total[5m])) * 100
```

### Token Cost Estimation

```promql
# Estimated daily cost at $0.50 per million tokens
(sum(increase(langgraph_tokens_used_total[24h])) / 1000000) * 0.5
```

### Identify Expensive Nodes

```promql
# Token usage by node (last hour)
sum by (node) (increase(langgraph_tokens_used_total[1h]))
```

---

## Structured Logging

### Log Format

All logs are output in structured JSON format (when `json_logs=True`):

```json
{
  "event": "graph_execution_completed",
  "timestamp": "2026-01-20T10:30:45.123Z",
  "level": "info",
  "profile_id": "starter",
  "user_id": "anon-123",
  "session_id": "sess_abc",
  "tokens_used": 1523
}
```

### Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages (LLM fallback, etc.)
- `ERROR`: Error messages with stack traces

### Viewing Logs

```bash
# LangGraph orchestration logs
docker compose logs -f langgraph

# FastAPI adapter logs
docker compose logs -f fastapi

# Next.js frontend logs
docker compose logs -f nextjs

# Filter by level
docker compose logs langgraph | grep '"level":"error"'

# Follow specific user
docker compose logs fastapi | grep 'user_id":"user_123"'
```

---

## Alerting

Alert rules are defined in `monitoring/alerts.yml` and evaluated by
Prometheus locally. Alerts are visible in the Prometheus UI at
[Prometheus alerts](http://localhost:9090/alerts).

To add notification routing (Slack, email, etc.), add an alertmanager
service to `docker-compose.yml` with a receiver configuration.

### Defined Alert Rules

| Alert | Condition | Severity |
| --- | --- | --- |
| `LangGraphTargetDown` | LangGraph unreachable for 2+ minutes | critical |
| `HighErrorRate` | Error rate >10% over 5 minutes | warning |
| `P95LatencyHigh` | p95 latency >5s for 5 minutes | warning |
| `NoSuccessfulRequests` | All requests fail for 10 minutes | critical |
| `DailyTokenUsageHigh` | >1M tokens in 24 hours | warning |
| `MemoryLoadErrorRateHigh` | Memory load error rate >10% over 5 minutes | warning |
| `MemoryQualityScoreLow` | Average importance score <0.3 | warning |

Check active alerts: [Prometheus alerts](http://localhost:9090/alerts)

---

## Performance Tuning

### Reduce Metric Cardinality

```python
# Avoid high-cardinality labels
# BAD: include labels (profile_id, model, user_id)  # user_id creates many series
# GOOD: include labels (profile_id, model, node)
```

### Prometheus Retention

```yaml
# In docker-compose.yml, add to prometheus command:
--storage.tsdb.retention.time=30d  # Keep metrics for 30 days
--storage.tsdb.retention.size=10GB # Or limit by size
```

---

## Troubleshooting

### Metrics Not Appearing in Frontend

**Symptom**: Metrics page shows empty or errors

**Solution**:

1. Check Prometheus is scraping LangGraph:

    ```bash
    curl http://localhost:9090/api/v1/targets
    ```

2. Verify LangGraph metrics endpoint (internal):

    ```bash
    docker exec langgraph curl http://localhost:8000/metrics
    ```

3. Check FastAPI can reach Prometheus:

    ```bash
    docker compose logs fastapi | grep -i prometheus
    ```

---

## Custom Metrics

### Adding New Metrics in Code

```python
from universal_agentic_framework.monitoring.metrics import Counter, Histogram

# Define metric
custom_operations = Counter(
    'custom_operations_total',
    'Custom operation counter',
    ['operation_type']
)

# Use in code
def my_operation():
    custom_operations.labels(operation_type='my_op').inc()
```

### Exposing Metrics

Metrics are automatically exposed at
[http://localhost:8000/metrics](http://localhost:8000/metrics)
(internal) by the LangGraph server.

Prometheus is configured to scrape this endpoint every 15 seconds
via the internal Docker network.

No additional configuration needed.

---

## Best Practices

### Dashboard Design

- ✅ Use consistent time ranges across panels
- ✅ Include p50, p95, p99 for latency metrics
- ✅ Show both rates and totals for counters
- ✅ Use thresholds for visual alerts

### Metric Naming

- ✅ Follow Prometheus naming conventions
- ✅ Use `_total` suffix for counters
- ✅ Use `_seconds` for durations
- ✅ Include unit in metric name

### Logging

- ✅ Always include contextual information (user_id, session_id)
- ✅ Use structured logging (key-value pairs)
- ✅ Log errors with stack traces
- ✅ Use appropriate log levels

### Alerting Guidance

- ✅ Set alerts on actionable metrics only
- ✅ Define clear severity levels
- ✅ Include remediation instructions in annotations
- ✅ Test alerts in staging before production

---

## See Also

- [Technical Architecture](technical_architecture.md) - System design
- [Configuration Guide](configuration.md) - Config reference
- [Status & Roadmap](status.md) - Current state and future work

---

## Architecture Reference

### Components

**Application services:**

- LangGraph (`langgraph`, internal port 8000) — Prometheus metrics at
  `/metrics`, health at `/health`, `/health/live`, `/health/ready`
- FastAPI adapter (`fastapi`, port 8001) — health endpoints and
  Prometheus-backed metrics proxying

**Observability stack:**

- Prometheus (port 9090) — scrapes LangGraph metrics, evaluates alert
  rules from `monitoring/alerts.yml`

### Data Flow

**Metrics:** LangGraph updates counters/histograms in-process →
Prometheus scrapes `/metrics` every 15s →
FastAPI queries Prometheus → frontend `/metrics` page displays.

**Alerts:** Prometheus evaluates `framework_alerts` rules every 30s →
condition persists for `for` duration → alerts visible in Prometheus UI at
[Prometheus alerts](http://localhost:9090/alerts).

### Alert Rule Baseline

Defined in `monitoring/alerts.yml`:

| Rule | Severity | SRE Dimension |
| --- | --- | --- |
| `LangGraphTargetDown` | critical | Availability |
| `HighErrorRate` | warning | Availability |
| `P95LatencyHigh` | warning | Latency |
| `NoSuccessfulRequests` | critical | Availability |
| `DailyTokenUsageHigh` | warning | Saturation/cost |

### Health and Readiness Strategy

- **Liveness** (`/health/live`): process/container is alive
- **Readiness** (`/health/ready`): instance can serve traffic
  (FastAPI includes DB probe, LangGraph includes graph init check)
- **Legacy** (`/health`): backward-compatible endpoint

### Operations Checklist

```bash
# Service health
docker compose ps
curl -s http://localhost:8001/health/ready
docker exec langgraph curl -s http://localhost:8000/health/ready

# Prometheus and alerting
curl -s http://localhost:9090/api/v1/rules
curl -s http://localhost:9090/api/v1/alerts

# Tracing
curl -s http://localhost:16686/api/services
# Expected: framework-fastapi, framework-langgraph
```

### Configuration Files

| File | Purpose |
| --- | --- |
| `monitoring/prometheus.yml` | Prometheus scrape config |
| `monitoring/alerts.yml` | Alert rule definitions |
