"""FastAPI HTTP server wrapper for LangGraph orchestration.

Exposes:
- /invoke: POST endpoint for graph execution
- /metrics: Prometheus metrics endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, REGISTRY, CONTENT_TYPE_LATEST
from fastapi.responses import Response, StreamingResponse

from universal_agentic_framework.config import get_active_profile_id, load_core_config
from universal_agentic_framework.orchestration.graph_builder import build_graph, GraphState
from universal_agentic_framework.orchestration.checkpointing import prune_checkpoints, setup_checkpointer
from universal_agentic_framework.orchestration.performance_nodes import compress_state
from universal_agentic_framework.monitoring.logging import configure_logging, get_logger, bind_context, clear_context
from universal_agentic_framework.monitoring.metrics import (
    track_graph_request,
    update_active_sessions,
    initialize_system_info,
)

# Configure logging
configure_logging(level="INFO", json_logs=False)
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Orchestration Server",
    description="HTTP wrapper for Steuermann graph execution",
    version="1.0.0"
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system metrics and build graph
CONFIG = load_core_config()
ACTIVE_PROFILE_ID = get_active_profile_id()
GRAPH = build_graph()
ACTIVE_SESSIONS: set[str] = set()
_invocation_count: int = 0

# Initialize system info metrics
initialize_system_info(version="1.0.0", environment="production")
logger.info("LangGraph server initialized", profile=CONFIG.profile.name)

# Pre-load and probe embedding provider — server must not start with a broken stack.
# Retries with exponential backoff (capped at 16 s per attempt) for ~2 min total.
# If the provider is still unreachable after all retries, startup fails and the
# container restarts (Docker restart policy handles recovery).
logger.info("Probing embedding provider at startup...")
_MAX_STARTUP_RETRIES = 15
for _attempt in range(_MAX_STARTUP_RETRIES):
    try:
        import time as _time
        from universal_agentic_framework.orchestration.helpers.embedding_provider import (
            get_routing_embedding_provider as _get_embed_provider,
        )
        _embedder, _embedding_model_name = _get_embed_provider(CONFIG)
        _embedder.encode("startup probe")
        logger.info("Embedding provider ready", model=_embedding_model_name, attempt=_attempt + 1)
        break
    except Exception as _e:
        _delay = min(2.0 ** _attempt, 16.0)
        if _attempt < _MAX_STARTUP_RETRIES - 1:
            logger.error(
                "Embedding provider not ready at startup — retrying",
                attempt=_attempt + 1,
                max=_MAX_STARTUP_RETRIES,
                retry_in=_delay,
                error=str(_e),
            )
            _time.sleep(_delay)
        else:
            logger.critical(
                "Embedding provider unreachable — aborting startup",
                error=str(_e),
                exc_info=True,
            )
            raise RuntimeError("Embedding provider unreachable at startup") from _e

logger.info("LangGraph server ready to accept requests")


@app.on_event("startup")
async def _startup() -> None:
    await setup_checkpointer(GRAPH.checkpointer)
    await prune_checkpoints(GRAPH.checkpointer)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "profile": CONFIG.profile.name}


@app.get("/health/live")
async def health_live() -> Dict[str, str]:
    """Liveness endpoint for process/container health."""
    return {"status": "ok", "check": "liveness", "profile": CONFIG.profile.name}


@app.get("/health/ready")
async def health_ready() -> Dict[str, Any]:
    """Readiness endpoint for serving traffic.

    Confirms core orchestration objects are initialized.
    """
    graph_ready = GRAPH is not None
    config_ready = CONFIG is not None
    ready = graph_ready and config_ready
    return {
        "status": "ok" if ready else "not_ready",
        "check": "readiness",
        "profile": CONFIG.profile.name,
        "graph_ready": graph_ready,
        "config_ready": config_ready,
    }


@app.post("/invoke")
async def invoke_graph(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the graph with provided state.
    
    Request body:
    {
        "messages": [{"role": "user", "content": "..."}, ...],
        "user_id": "user-123",
        "language": "en"  # optional, default: en
    }
    
    Returns:
    {
        "messages": [...],
        "loaded_memory": [...],
        "knowledge_context": [...],
        "tool_results": {...},
        "tokens_used": 0,
        "summary_text": "...",
        ...
    }
    """
    try:
        # Validate required fields
        if "messages" not in request:
            raise HTTPException(status_code=400, detail="Missing 'messages' field")
        if "user_id" not in request:
            raise HTTPException(status_code=400, detail="Missing 'user_id' field")
        
        user_id = request.get("user_id", "anonymous")
        language = request.get("language", "en")
        
        # Bind logging context
        bind_context(user_id=user_id)
        
        # Track session
        ACTIVE_SESSIONS.add(user_id)
        update_active_sessions(CONFIG.profile.name, len(ACTIVE_SESSIONS))
        
        profile_name = CONFIG.profile.name
        profile_id = ACTIVE_PROFILE_ID
        
        logger.info(
            "Graph invocation received",
            user_id=user_id,
            language=language,
            message_count=len(request.get("messages", [])),
            attachment_count=len(request.get("attachments", [])),
            workspace_document_count=len(request.get("workspace_documents", [])),
        )
        
        # Ephemeral sessions (no session_id) are not checkpointed
        session_id = request.get("session_id")

        # Load at the edge: merge accumulated messages from checkpoint with new user message
        existing_messages: list = []
        if session_id:
            ct = await GRAPH.checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
            if ct:
                existing_messages = ct.checkpoint.get("channel_values", {}).get("messages", [])

        # Prepare graph state
        state: GraphState = {
            "messages": existing_messages + request.get("messages", []),
            "user_id": user_id,
            "session_id": session_id,
            "language": language,
            "profile_name": profile_name,
            "profile_id": profile_id,
            "user_settings": request.get("user_settings", {}),  # Include user settings from request
            # Preserve adapter-provided probe snapshots so Layer 1 can resolve
            # tool-calling mode with probe-aware downgrade logic.
            "llm_capability_probes": request.get("llm_capability_probes", []),
            "attachments": request.get("attachments", []),
            "workspace_documents": request.get("workspace_documents", []),
            "workspace_writeback_requested": bool(request.get("workspace_writeback_requested", False)),
        }
        
        logger.debug("Graph state prepared", session_id=session_id, profile_name=profile_name)
        
        # Track request with metrics
        with track_graph_request(profile_name) as ctx:
            try:
                invoke_config = {"configurable": {"thread_id": session_id}} if session_id else {}

                # Execute graph (use invoke since HTTP is sync context)
                result = GRAPH.invoke(state, config=invoke_config)

                ctx["status"] = "success"
                global _invocation_count
                _invocation_count += 1
                if _invocation_count % 100 == 0:
                    asyncio.create_task(prune_checkpoints(GRAPH.checkpointer))
                logger.info(
                    "Graph execution completed successfully",
                    tokens_used=result.get("tokens_used", 0),
                    tools_executed=len(result.get("tool_results", {}))
                )
                
                # Return result
                return {
                    "messages": result.get("messages", []),
                    "loaded_memory": result.get("loaded_memory", []),
                    "memory_analytics": result.get("memory_analytics", {}),
                    "knowledge_context": result.get("knowledge_context", []),
                    "tool_results": result.get("tool_results", {}),
                    "tokens_used": result.get("tokens_used", 0),
                    "input_tokens": result.get("last_input_tokens", 0),
                    "output_tokens": result.get("last_output_tokens", 0),
                    "provider_used": result.get("provider_used", "unknown"),
                    "model_used": result.get("model_used", "unknown"),
                    "profile_id": result.get("profile_id", profile_id),
                    "summary_text": result.get("summary_text", ""),
                    "sources": result.get("sources", []),
                    "rag_attempted": result.get("rag_attempted", False),
                    "rag_doc_count": result.get("rag_doc_count", 0),
                }
                
            except Exception as e:
                ctx["status"] = "error"
                logger.error(
                    "Graph execution failed",
                    error=str(e),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Graph execution error: {str(e)}"
                )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in /invoke endpoint", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        clear_context()


_NODE_STATUS_LABELS: dict[str, str] = {
    "retrieve_knowledge": "Searching knowledge base...",
    "load_memory": "Loading memories...",
}

# Tool-calling nodes bypass LangChain's callback chain, so on_tool_start never
# fires. We synthesise tool_call events from on_chain_start / on_chain_end.
_TOOL_CALL_NODES: frozenset[str] = frozenset({
    "call_tools_native",
    "call_tools_structured",
    "call_tools_react",
})

# Full set of real graph node names, derived from the compiled graph so it stays
# in sync with the builder. Drives the Inspector `node_state` trace (every node,
# not just the user-facing _NODE_STATUS_LABELS subset). on_chain_* events also
# fire for non-node runnables; membership here filters those out. Two reflection
# sources are tried (their shapes vary across langgraph versions); if both fail
# we log rather than silently disabling the Inspector trace.
def _discover_graph_node_names() -> frozenset[str]:
    for accessor in (lambda: GRAPH.get_graph().nodes, lambda: GRAPH.nodes):
        try:
            names = frozenset(accessor()) - {"__start__", "__end__"}
            if names:
                return names
        except Exception:
            continue
    logger.warning("Inspector node trace disabled: could not enumerate graph node names")
    return frozenset()


_GRAPH_NODE_NAMES: frozenset[str] = _discover_graph_node_names()

_TOOL_LABELS: dict[str, str] = {
    "web_search_mcp": "Searching the web...",
    "duckduckgo_search": "Searching the web...",
    "extract_webpage_mcp": "Reading webpage...",
    "calculator_tool": "Calculating...",
    "datetime_tool": "Checking date & time...",
    "memory_search_tool": "Searching memories...",
    "file_read_tool": "Reading file...",
    "file_write_tool": "Writing file...",
    "workspace_file_ops_tool": "Editing workspace document...",
}


# Ordered longest-first to avoid matching <think> before <thinking>.
_THINK_TAG_PAIRS: list[tuple[str, str]] = [
    ("<thinking>", "</thinking>"),
    ("<reflection>", "</reflection>"),
    ("<think>", "</think>"),
]
_MAX_OPEN_TAG_LEN = max(len(o) for o, _ in _THINK_TAG_PAIRS)   # 10
_MAX_CLOSE_TAG_LEN = max(len(c) for _, c in _THINK_TAG_PAIRS)  # 11


def _tool_label(tool_name: str) -> str:
    """Return a human-readable label for a tool name, falling back to a generated one."""
    if tool_name in _TOOL_LABELS:
        return _TOOL_LABELS[tool_name]
    friendly = (
        tool_name
        .replace("_mcp", "")
        .replace("_tool", "")
        .replace("_", " ")
        .title()
    )
    return f"Using {friendly}..."


async def _drain_remaining(iterator: Any) -> None:
    """Silently consume remaining astream_events so post-processing nodes complete."""
    try:
        async for _ in iterator:
            pass
    except Exception:
        pass


@app.post("/stream")
async def stream_graph(request: Dict[str, Any]) -> StreamingResponse:
    """Stream graph execution events as Server-Sent Events.

    Emits token, tool_call, node, metadata, and error events.
    Terminates with ``data: [DONE]``.
    """
    if "messages" not in request:
        raise HTTPException(status_code=400, detail="Missing 'messages' field")
    if "user_id" not in request:
        raise HTTPException(status_code=400, detail="Missing 'user_id' field")

    user_id = request.get("user_id", "anonymous")
    language = request.get("language", "en")
    bind_context(user_id=user_id)
    ACTIVE_SESSIONS.add(user_id)
    update_active_sessions(CONFIG.profile.name, len(ACTIVE_SESSIONS))
    profile_name = CONFIG.profile.name
    profile_id = ACTIVE_PROFILE_ID

    # Ephemeral sessions (no session_id) are not checkpointed
    session_id = request.get("session_id")

    # Load at the edge: merge accumulated messages from checkpoint with new user message
    existing_messages: list = []
    if session_id:
        ct = await GRAPH.checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
        if ct:
            existing_messages = ct.checkpoint.get("channel_values", {}).get("messages", [])

    state: GraphState = {
        "messages": existing_messages + request.get("messages", []),
        "user_id": user_id,
        "session_id": session_id,
        "language": language,
        "profile_name": profile_name,
        "profile_id": profile_id,
        "user_settings": request.get("user_settings", {}),
        "llm_capability_probes": request.get("llm_capability_probes", []),
        "attachments": request.get("attachments", []),
        "workspace_documents": request.get("workspace_documents", []),
        "workspace_writeback_requested": bool(request.get("workspace_writeback_requested", False)),
    }
    invoke_config = {"configurable": {"thread_id": session_id}} if session_id else {}

    logger.info(
        "Graph stream invocation received",
        user_id=user_id,
        language=language,
        message_count=len(request.get("messages", [])),
    )

    async def _event_stream() -> AsyncGenerator[str, None]:
        _final_output: dict = {}
        _done_emitted = False
        _captured_input_tokens: int = 0
        _captured_output_tokens: int = 0
        _captured_map_data = None  # populated from tool-call node output, not respond node
        _in_thinking = False   # currently inside a <think> block
        _close_tag = ""        # matching close tag for the current block
        _pending_buf = ""      # guards against tags split across chunk boundaries
        _full_thinking = ""    # accumulated thinking text for metadata
        _node_start: dict[str, float] = {}  # node name -> perf_counter() at start
        _node_seq = 0          # monotonic sequence for the Inspector node trace
        _node_trace_list: list[dict[str, Any]] = []  # accumulated trace, persisted in metadata
        _events_iter = GRAPH.astream_events(state, config=invoke_config, version="v2")
        try:
            with track_graph_request(profile_name) as ctx:
                async for event in _events_iter:
                    kind: str = event.get("event", "")
                    name: str = event.get("name", "")
                    data: dict = event.get("data", {})

                    if kind == "on_chat_model_stream":
                        # Only stream tokens from the respond node — not from
                        # tool-selection nodes which emit raw JSON tool calls.
                        langgraph_node = event.get("metadata", {}).get("langgraph_node", "")
                        if langgraph_node and langgraph_node != "respond":
                            continue
                        chunk = data.get("chunk")
                        delta = ""
                        if chunk is not None:
                            # One-shot debug: log the first non-empty chunk shape so we can
                            # confirm where LM Studio puts reasoning content.
                            _ak_debug = getattr(chunk, "additional_kwargs", {}) or {}
                            _content_debug = getattr(chunk, "content", "")
                            if (_ak_debug.get("reasoning_content") or _content_debug) and not _full_thinking and not _pending_buf:
                                logger.debug(
                                    "First content chunk shape",
                                    has_content=bool(_content_debug),
                                    content_preview=str(_content_debug)[:60] if _content_debug else None,
                                    has_reasoning_content=bool(_ak_debug.get("reasoning_content")),
                                    reasoning_preview=str(_ak_debug.get("reasoning_content", ""))[:60] or None,
                                    additional_kwargs_keys=list(_ak_debug.keys()),
                                )

                            # Capture real usage from the final streaming chunk (LM Studio
                            # always reports usage here; this fires before on_chain_end).
                            _chunk_usage = getattr(chunk, "usage_metadata", None)
                            if isinstance(_chunk_usage, dict) and _chunk_usage:
                                _in = _chunk_usage.get("input_tokens", 0) or 0
                                _out = _chunk_usage.get("output_tokens", 0) or 0
                                if _in > 0:
                                    logger.debug("Captured usage from streaming chunk (on_chat_model_stream)", input_tokens=_in, output_tokens=_out)
                                    _captured_input_tokens = _in
                                if _out > 0:
                                    _captured_output_tokens = _out
                            # Native reasoning_content field (LM Studio / LiteLLM / Ollama)
                            # takes priority: emit directly without going through the tag parser.
                            _ak = getattr(chunk, "additional_kwargs", {}) or {}
                            _reasoning_delta = _ak.get("reasoning_content") or ""
                            if _reasoning_delta:
                                if not _in_thinking:
                                    _in_thinking = True
                                    _close_tag = ""
                                    yield "event: thinking_start\ndata: {}\n\n"
                                yield f"event: thinking\ndata: {json.dumps({'delta': _reasoning_delta})}\n\n"
                                _full_thinking += _reasoning_delta

                            content = getattr(chunk, "content", chunk)
                            if isinstance(content, str):
                                delta = content
                            elif isinstance(content, list):
                                delta = "".join(
                                    b.get("text", "") if isinstance(b, dict) else str(b)
                                    for b in content
                                )

                            # Close the native reasoning block once content starts arriving.
                            if delta and _in_thinking and not _close_tag:
                                _in_thinking = False
                                yield "event: thinking_end\ndata: {}\n\n"

                        if delta:
                            _pending_buf += delta
                            while True:
                                if not _in_thinking:
                                    best_idx, best_open, best_close = -1, "", ""
                                    for open_tag, close_tag in _THINK_TAG_PAIRS:
                                        idx = _pending_buf.find(open_tag)
                                        if idx != -1 and (best_idx == -1 or idx < best_idx):
                                            best_idx, best_open, best_close = idx, open_tag, close_tag
                                    if best_idx == -1:
                                        safe = max(0, len(_pending_buf) - (_MAX_OPEN_TAG_LEN - 1))
                                        if safe > 0:
                                            yield f"event: token\ndata: {json.dumps({'delta': _pending_buf[:safe]})}\n\n"
                                            _pending_buf = _pending_buf[safe:]
                                        break
                                    if best_idx > 0:
                                        yield f"event: token\ndata: {json.dumps({'delta': _pending_buf[:best_idx]})}\n\n"
                                    _pending_buf = _pending_buf[best_idx + len(best_open):]
                                    _in_thinking = True
                                    _close_tag = best_close
                                    yield "event: thinking_start\ndata: {}\n\n"
                                else:
                                    idx = _pending_buf.find(_close_tag)
                                    if idx == -1:
                                        safe = max(0, len(_pending_buf) - (_MAX_CLOSE_TAG_LEN - 1))
                                        if safe > 0:
                                            chunk_text = _pending_buf[:safe]
                                            yield f"event: thinking\ndata: {json.dumps({'delta': chunk_text})}\n\n"
                                            _full_thinking += chunk_text
                                            _pending_buf = _pending_buf[safe:]
                                        break
                                    if idx > 0:
                                        chunk_text = _pending_buf[:idx]
                                        yield f"event: thinking\ndata: {json.dumps({'delta': chunk_text})}\n\n"
                                        _full_thinking += chunk_text
                                    _pending_buf = _pending_buf[idx + len(_close_tag):]
                                    _in_thinking = False
                                    _close_tag = ""
                                    yield "event: thinking_end\ndata: {}\n\n"

                    elif kind == "on_chat_model_end":
                        # Override streaming chunk capture with the fully-merged result.
                        # Fires once per LLM call after all chunks accumulate — more
                        # reliable than the per-chunk path.
                        langgraph_node = event.get("metadata", {}).get("langgraph_node", "")
                        if langgraph_node == "respond":
                            _end_output = data.get("output")
                            _end_usage = None
                            if hasattr(_end_output, "usage_metadata"):
                                _end_usage = _end_output.usage_metadata
                            elif hasattr(_end_output, "message"):
                                _end_usage = getattr(_end_output.message, "usage_metadata", None)
                            if isinstance(_end_usage, dict) and _end_usage:
                                _in = _end_usage.get("input_tokens", 0) or 0
                                _out = _end_usage.get("output_tokens", 0) or 0
                                logger.debug("Captured usage from on_chat_model_end", input_tokens=_in, output_tokens=_out)
                                if _in > 0:
                                    _captured_input_tokens = _in
                                if _out > 0:
                                    _captured_output_tokens = _out

                    elif kind == "on_tool_start":
                        # Fallback path — fires if future refactor adds LangChain
                        # tool invocation with callbacks wired up.
                        yield f"event: tool_call\ndata: {json.dumps({'name': name, 'status': 'start', 'label': _tool_label(name)})}\n\n"

                    elif kind == "on_tool_end":
                        yield f"event: tool_call\ndata: {json.dumps({'name': name, 'status': 'end', 'label': _tool_label(name)})}\n\n"

                    elif kind == "on_chain_start":
                        if name in _GRAPH_NODE_NAMES:
                            _node_start[name] = time.perf_counter()
                        if name in _NODE_STATUS_LABELS:
                            # Suppress retrieve_knowledge indicator when the user has
                            # explicitly disabled RAG — the node still runs but exits
                            # immediately, so showing the label is misleading.
                            if name == "retrieve_knowledge":
                                _rag_enabled = (
                                    state.get("user_settings", {})
                                    .get("rag_config", {})
                                    .get("enabled", True)
                                )
                                if not _rag_enabled:
                                    continue
                            yield (
                                f"event: node\ndata: "
                                f"{json.dumps({'node': name, 'label': _NODE_STATUS_LABELS[name]})}\n\n"
                            )
                        elif name in _TOOL_CALL_NODES:
                            # Tool-calling nodes invoke tools directly in Python,
                            # so on_tool_start never fires. Emit start here instead.
                            yield f"event: tool_call\ndata: {json.dumps({'name': name, 'status': 'start', 'label': 'Using tools...'})}\n\n"

                    elif kind == "on_chain_end":
                        if name in _GRAPH_NODE_NAMES:
                            _node_seq += 1
                            _started = _node_start.pop(name, None)
                            _dur_ms = (
                                round((time.perf_counter() - _started) * 1000, 1)
                                if _started is not None
                                else None
                            )
                            _ns = {"node": name, "sequence": _node_seq, "duration_ms": _dur_ms, "status": "success"}
                            _node_trace_list.append(_ns)
                            yield f"event: node_state\ndata: {json.dumps(_ns)}\n\n"
                        output = data.get("output")
                        if isinstance(output, dict):
                            if "messages" in output:
                                _final_output = output
                                if name == "respond" and not _done_emitted:
                                    # Drain pending buffer before emitting metadata.
                                    if _in_thinking:
                                        if _pending_buf:
                                            yield f"event: thinking\ndata: {json.dumps({'delta': _pending_buf})}\n\n"
                                            _full_thinking += _pending_buf
                                            _pending_buf = ""
                                        yield "event: thinking_end\ndata: {}\n\n"
                                        _in_thinking = False
                                    elif _pending_buf:
                                        yield f"event: token\ndata: {json.dumps({'delta': _pending_buf})}\n\n"
                                        _pending_buf = ""
                                    # Emit metadata + [DONE] immediately after the respond
                                    # node so the client can render pills without waiting
                                    # for summarize + update_memory (~43 s).
                                    _map_data = _captured_map_data
                                    meta_payload = {
                                        "tokens_used": _final_output.get("tokens_used", 0),
                                        "input_tokens": _captured_input_tokens if _captured_input_tokens > 0 else _final_output.get("last_input_tokens", 0),
                                        "output_tokens": _captured_output_tokens if _captured_output_tokens > 0 else _final_output.get("last_output_tokens", 0),
                                        "provider_used": _final_output.get("provider_used", "unknown"),
                                        "model_used": _final_output.get("model_used", "unknown"),
                                        "profile_id": _final_output.get("profile_id", profile_id),
                                        "tool_results": _final_output.get("tool_results", {}),
                                        "sources": _final_output.get("sources", []),
                                        "rag_attempted": _final_output.get("rag_attempted", False),
                                        "rag_doc_count": _final_output.get("rag_doc_count", 0),
                                        "loaded_memory": _final_output.get("loaded_memory", []),
                                        "memory_analytics": _final_output.get("memory_analytics", {}),
                                        "thinking_content": _full_thinking if _full_thinking else None,
                                        "map_data": _map_data,
                                        "node_trace": _node_trace_list,
                                    }
                                    yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    _done_emitted = True
                                    global _invocation_count
                                    _invocation_count += 1
                                    if _invocation_count % 100 == 0:
                                        asyncio.create_task(prune_checkpoints(GRAPH.checkpointer))
                                    # Drain post-processing nodes (summarize, update_memory)
                                    # in the background so they complete without blocking.
                                    asyncio.create_task(_drain_remaining(_events_iter))
                                    break
                            # Emit specific tool label once we know which tool ran.
                            if name in _TOOL_CALL_NODES:
                                tool_results: dict = output.get("tool_results") or {}
                                tool_names = list(tool_results.keys())
                                t_name = tool_names[0] if tool_names else name
                                yield f"event: tool_call\ndata: {json.dumps({'name': t_name, 'status': 'end', 'label': _tool_label(t_name)})}\n\n"
                                _tool_exec = output.get("tool_execution_results") or {}
                                if "map_tool" in _tool_exec:
                                    _captured_map_data = _tool_exec["map_tool"].get("data")

                    elif kind == "on_chain_error":
                        # Best-effort per-node error status for the Inspector. A
                        # raising node fires on_chain_error (not on_chain_end).
                        if name in _GRAPH_NODE_NAMES:
                            _node_seq += 1
                            _started = _node_start.pop(name, None)
                            _dur_ms = (
                                round((time.perf_counter() - _started) * 1000, 1)
                                if _started is not None
                                else None
                            )
                            _ns = {"node": name, "sequence": _node_seq, "duration_ms": _dur_ms, "status": "error"}
                            _node_trace_list.append(_ns)
                            yield f"event: node_state\ndata: {json.dumps(_ns)}\n\n"

                ctx["status"] = "success"

            # Fallback: emit metadata if the respond node's on_chain_end never fired
            # (e.g. graph ended via a different path).
            if not _done_emitted and _final_output:
                if _in_thinking:
                    if _pending_buf:
                        yield f"event: thinking\ndata: {json.dumps({'delta': _pending_buf})}\n\n"
                        _full_thinking += _pending_buf
                        _pending_buf = ""
                    yield "event: thinking_end\ndata: {}\n\n"
                    _in_thinking = False
                elif _pending_buf:
                    yield f"event: token\ndata: {json.dumps({'delta': _pending_buf})}\n\n"
                    _pending_buf = ""
                _map_data = _captured_map_data
                meta_payload = {
                    "tokens_used": _final_output.get("tokens_used", 0),
                    "input_tokens": _captured_input_tokens if _captured_input_tokens > 0 else _final_output.get("last_input_tokens", 0),
                    "output_tokens": _captured_output_tokens if _captured_output_tokens > 0 else _final_output.get("last_output_tokens", 0),
                    "provider_used": _final_output.get("provider_used", "unknown"),
                    "model_used": _final_output.get("model_used", "unknown"),
                    "profile_id": _final_output.get("profile_id", profile_id),
                    "tool_results": _final_output.get("tool_results", {}),
                    "sources": _final_output.get("sources", []),
                    "rag_attempted": _final_output.get("rag_attempted", False),
                    "rag_doc_count": _final_output.get("rag_doc_count", 0),
                    "loaded_memory": _final_output.get("loaded_memory", []),
                    "memory_analytics": _final_output.get("memory_analytics", {}),
                    "thinking_content": _full_thinking if _full_thinking else None,
                    "map_data": _map_data,
                    "node_trace": _node_trace_list,
                }
                yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

        except Exception as exc:
            logger.error("Stream graph error", error=str(exc), exc_info=True)
            if not _done_emitted:
                yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
        finally:
            clear_context()
            if not _done_emitted:
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/compact")
async def compact_conversation(request: Dict[str, Any]) -> Dict[str, Any]:
    """Manually compress a conversation's LangGraph checkpoint.

    Loads the checkpoint for ``session_id``, runs force-compression (bypassing the
    auto-compact token threshold), and saves the compressed messages back.  The
    ConversationStore (UI message log) is intentionally left unchanged — only the
    checkpoint (what the LLM sees next turn) is updated.

    Returns ``status: "skipped"`` if the conversation is too short to compress
    (≤ keep_recent_count messages), or ``status: "ok"`` with ``estimated_tokens``.
    """
    import uuid
    from datetime import datetime, timezone

    session_id = request.get("session_id")
    user_id = request.get("user_id", "unknown")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    config = {"configurable": {"thread_id": session_id}}
    ct = await GRAPH.checkpointer.aget_tuple(config)
    if not ct:
        raise HTTPException(status_code=404, detail="No checkpoint found for this session")

    channel_values = ct.checkpoint.get("channel_values", {})
    state = {**channel_values, "user_id": user_id}

    compressed = await compress_state(state, force=True)

    if compressed.get("messages") == channel_values.get("messages"):
        return {"status": "skipped", "estimated_tokens": compressed.get("tokens_used", 0)}

    new_checkpoint_id = str(uuid.uuid4())
    new_checkpoint = {
        **ct.checkpoint,
        "id": new_checkpoint_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {
            **channel_values,
            "messages": compressed["messages"],
            "tokens_used": compressed.get("tokens_used", 0),
            "digest_chain": compressed.get("digest_chain", []),
        },
    }
    save_config = {
        "configurable": {
            "thread_id": session_id,
            "checkpoint_ns": ct.config["configurable"].get("checkpoint_ns", ""),
        }
    }
    new_versions = ct.checkpoint.get("channel_versions", {})
    await GRAPH.checkpointer.aput(save_config, new_checkpoint, ct.metadata or {}, new_versions)

    logger.info("Manual compact complete", session_id=session_id, estimated_tokens=compressed.get("tokens_used", 0))
    return {
        "status": "ok",
        "estimated_tokens": compressed.get("tokens_used", 0),
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint (scrapable by Prometheus)."""
    try:
        metrics_output = generate_latest(REGISTRY)
        return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e), exc_info=True)
        return Response(content="", media_type=CONTENT_TYPE_LATEST, status_code=500)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "universal_agentic_framework.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
