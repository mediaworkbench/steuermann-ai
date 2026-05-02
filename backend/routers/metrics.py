from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query

from backend.prometheus_client import PrometheusClient, extract_value, extract_values_dict
from backend.single_user import require_api_access


router = APIRouter(prefix="/api", tags=["metrics"], dependencies=[Depends(require_api_access)])


def _daily_series_from_range_query(results: list[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate Prometheus range query values by UTC day (YYYY-MM-DD)."""
    daily: Dict[str, float] = {}
    for series in results:
        values = series.get("values") or []
        for item in values:
            if not isinstance(item, list) or len(item) < 2:
                continue
            try:
                ts = float(item[0])
                value = float(item[1])
            except (TypeError, ValueError):
                continue
            if value != value:  # NaN guard
                continue
            day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
            daily[day] = daily.get(day, 0.0) + value
    return daily


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Aggregate metrics from Prometheus."""
    client = PrometheusClient()

    metrics = {
        "requests": {},
        "tokens": {},
        "latency": {},
        "sessions": {},
        "memory_ops": {},
        "memory_ops_by_status": {},
        "llm_calls": {},
        "attachments": {},
        "attachment_retries": {},
        "profile_guardrails": {},
        "workspace": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Total requests by status
    requests_results = await client.query('sum by (status) (langgraph_requests_total)')
    for result in requests_results:
        status = result.get("metric", {}).get("status", "unknown")
        value = extract_value(result)
        if value is not None:
            metrics["requests"][status] = value

    # Total tokens used by fork and model
    tokens_results = await client.query('sum by (fork_name, model) (langgraph_tokens_used_total)')
    for result in tokens_results:
        labels = result.get("metric", {})
        fork = labels.get("fork_name", "unknown")
        model = labels.get("model", "unknown")
        value = extract_value(result)
        if value is not None:
            key = f"{fork}/{model}"
            metrics["tokens"][key] = value

    # Request latency (average). Histogram metrics must use _sum/_count.
    latency_results = await client.query(
        'sum(langgraph_request_duration_seconds_sum) / sum(langgraph_request_duration_seconds_count)'
    )
    for result in latency_results:
        value = extract_value(result)
        if value is not None:
            metrics["latency"]["avg_request_duration_seconds"] = value

    # Active sessions by fork
    sessions_results = await client.query('langgraph_active_sessions')
    for result in sessions_results:
        fork = result.get("metric", {}).get("fork_name", "unknown")
        value = extract_value(result)
        if value is not None:
            metrics["sessions"][fork] = value

    # Memory operations by type
    mem_ops_results = await client.query('sum by (operation) (langgraph_memory_operations_total)')
    for result in mem_ops_results:
        operation = result.get("metric", {}).get("operation", "unknown")
        value = extract_value(result)
        if value is not None:
            metrics["memory_ops"][operation] = value

    # Memory operations broken down by operation + status
    mem_ops_status_results = await client.query(
        'sum by (operation, status) (langgraph_memory_operations_total)'
    )
    mem_ops_by_status: dict = {}
    for result in mem_ops_status_results:
        labels = result.get("metric", {})
        operation = labels.get("operation", "unknown")
        status = labels.get("status", "unknown")
        value = extract_value(result)
        if value is not None:
            key = f"{operation}/{status}"
            mem_ops_by_status[key] = value
    metrics["memory_ops_by_status"] = mem_ops_by_status

    # Memory quality score (average of histogram sum/count)
    mem_quality_results = await client.query(
        'sum(langgraph_memory_quality_score_sum) / sum(langgraph_memory_quality_score_count)'
    )
    for result in mem_quality_results:
        value = extract_value(result)
        if value is not None:
            metrics["memory_ops"]["avg_quality_score"] = round(value, 4)

    # Co-occurrence graph size
    cooc_nodes_results = await client.query('sum(langgraph_memory_co_occurrence_graph_nodes)')
    for result in cooc_nodes_results:
        value = extract_value(result)
        if value is not None:
            metrics["memory_ops"]["co_occurrence_nodes"] = value

    cooc_edges_results = await client.query('sum(langgraph_memory_co_occurrence_graph_edges)')
    for result in cooc_edges_results:
        value = extract_value(result)
        if value is not None:
            metrics["memory_ops"]["co_occurrence_edges"] = value

    # Importance ranking duration (avg)
    importance_dur_results = await client.query(
        'sum(langgraph_memory_importance_ranking_duration_seconds_sum) / sum(langgraph_memory_importance_ranking_duration_seconds_count)'
    )
    for result in importance_dur_results:
        value = extract_value(result)
        if value is not None:
            metrics["memory_ops"]["avg_importance_ranking_ms"] = round(value * 1000, 2)

    # LLM calls by provider and model
    llm_results = await client.query('sum by (provider, model, status) (langgraph_llm_calls_total)')
    for result in llm_results:
        labels = result.get("metric", {})
        provider = labels.get("provider", "unknown")
        model = labels.get("model", "unknown")
        status = labels.get("status", "unknown")
        value = extract_value(result)
        if value is not None:
            key = f"{provider}/{model}/{status}"
            metrics["llm_calls"][key] = value

    # Attachment context metrics
    attachment_injected_results = await client.query('sum(langgraph_attachments_injected_total)')
    for result in attachment_injected_results:
        value = extract_value(result)
        if value is not None:
            metrics["attachments"]["injected_total"] = value

    attachment_none_results = await client.query('sum(langgraph_attachments_none_total)')
    for result in attachment_none_results:
        value = extract_value(result)
        if value is not None:
            metrics["attachments"]["none_total"] = value

    # Attachment refusal-retry guardrail metrics
    attachment_retry_results = await client.query('sum(langgraph_attachment_refusal_retries_total)')
    for result in attachment_retry_results:
        value = extract_value(result)
        if value is not None:
            metrics["attachment_retries"]["retry_total"] = value

    attachment_retry_success_results = await client.query('sum(langgraph_attachment_refusal_retries_success_total)')
    for result in attachment_retry_success_results:
        value = extract_value(result)
        if value is not None:
            metrics["attachment_retries"]["retry_success_total"] = value

    # Profile guardrail metrics
    profile_mismatch_results = await client.query('sum(langgraph_profile_id_mismatch_total)')
    for result in profile_mismatch_results:
        value = extract_value(result)
        if value is not None:
            metrics["profile_guardrails"]["profile_id_mismatch_total"] = value

    # Workspace operation metrics
    workspace_created_results = await client.query('sum(langgraph_workspace_created_total)')
    for result in workspace_created_results:
        value = extract_value(result)
        if value is not None:
            metrics["workspace"]["created_total"] = value

    workspace_write_allowed_results = await client.query('sum(langgraph_workspace_write_allowed_total)')
    for result in workspace_write_allowed_results:
        value = extract_value(result)
        if value is not None:
            metrics["workspace"]["write_allowed_total"] = value

    workspace_write_denied_results = await client.query('sum(langgraph_workspace_write_denied_total)')
    for result in workspace_write_denied_results:
        value = extract_value(result)
        if value is not None:
            metrics["workspace"]["write_denied_total"] = value

    workspace_intent_denied_results = await client.query(
        'sum by (operation, reason) (langgraph_workspace_intent_denied_total)'
    )
    for result in workspace_intent_denied_results:
        labels = result.get("metric", {})
        operation = labels.get("operation", "unknown")
        reason = labels.get("reason", "unknown")
        value = extract_value(result)
        if value is not None:
            key = f"{operation}/{reason}"
            metrics["workspace"]["intent_denied"] = metrics["workspace"].get("intent_denied", {})
            metrics["workspace"]["intent_denied"][key] = value

    workspace_revised_copy_results = await client.query('sum(langgraph_workspace_revised_copy_created_total)')
    for result in workspace_revised_copy_results:
        value = extract_value(result)
        if value is not None:
            metrics["workspace"]["revised_copy_created_total"] = value

    workspace_cleanup_deleted_results = await client.query('sum(langgraph_workspace_cleanup_deleted_total)')
    for result in workspace_cleanup_deleted_results:
        value = extract_value(result)
        if value is not None:
            metrics["workspace"]["cleanup_deleted_total"] = value

    return metrics


@router.get("/analytics/memory-trends")
async def get_memory_trends(
    days: int = Query(default=30, ge=1, le=365),
) -> Dict[str, Any]:
    """Prometheus-backed daily memory trends for the last N days."""
    client = PrometheusClient()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days - 1)
    start = start_dt.isoformat()
    end = end_dt.isoformat()
    step = "1d"

    load_results = await client.query_range(
        'sum(increase(langgraph_memory_operations_total{operation="load"}[1d]))',
        start=start,
        end=end,
        step=step,
    )
    update_results = await client.query_range(
        'sum(increase(langgraph_memory_operations_total{operation="update"}[1d]))',
        start=start,
        end=end,
        step=step,
    )
    error_results = await client.query_range(
        'sum(increase(langgraph_memory_operations_total{status="error"}[1d]))',
        start=start,
        end=end,
        step=step,
    )
    quality_results = await client.query_range(
        'sum(increase(langgraph_memory_quality_score_sum[1d])) / clamp_min(sum(increase(langgraph_memory_quality_score_count[1d])), 1)',
        start=start,
        end=end,
        step=step,
    )

    load_daily = _daily_series_from_range_query(load_results)
    update_daily = _daily_series_from_range_query(update_results)
    error_daily = _daily_series_from_range_query(error_results)
    quality_daily = _daily_series_from_range_query(quality_results)

    trends = []
    for offset in range(days):
        current_day = (start_dt + timedelta(days=offset)).date().isoformat()
        loads = load_daily.get(current_day, 0.0)
        updates = update_daily.get(current_day, 0.0)
        errors = error_daily.get(current_day, 0.0)
        total_ops = loads + updates
        error_rate = (errors / total_ops * 100.0) if total_ops > 0 else 0.0
        avg_quality_score = quality_daily.get(current_day, 0.0)
        trends.append(
            {
                "date": current_day,
                "loads": round(loads, 4),
                "updates": round(updates, 4),
                "errors": round(errors, 4),
                "error_rate": round(error_rate, 2),
                "avg_quality_score": round(avg_quality_score, 4),
            }
        )

    total_loads = sum(day["loads"] for day in trends)
    total_updates = sum(day["updates"] for day in trends)
    total_errors = sum(day["errors"] for day in trends)
    total_ops = total_loads + total_updates

    return {
        "period_days": days,
        "trends": trends,
        "totals": {
            "loads": round(total_loads, 4),
            "updates": round(total_updates, 4),
            "errors": round(total_errors, 4),
            "error_rate": round((total_errors / total_ops * 100.0), 2) if total_ops > 0 else 0.0,
        },
    }
