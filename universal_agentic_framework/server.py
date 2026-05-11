"""FastAPI HTTP server wrapper for LangGraph orchestration.

Exposes:
- /invoke: POST endpoint for graph execution
- /metrics: Prometheus metrics endpoint
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, REGISTRY, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from universal_agentic_framework.config import get_active_profile_id, load_core_config
from universal_agentic_framework.orchestration.graph_builder import build_graph, GraphState
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

# Initialize system info metrics
initialize_system_info(version="1.0.0", environment="production")
logger.info("LangGraph server initialized", fork=CONFIG.fork.name)

# Pre-load embedding model for semantic tool routing to avoid cold-start delays
logger.info("Pre-loading embedding model for faster first request...")
try:
    from universal_agentic_framework.orchestration import graph_builder

    _, embedding_model_name = graph_builder._get_routing_embedding_provider(CONFIG)
    logger.info(f"Embedding provider pre-loaded: {embedding_model_name}")
except Exception as e:
    logger.warning(f"Failed to pre-load embedding model: {e}, will load on first request")

logger.info("LangGraph server ready to accept requests")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "fork": CONFIG.fork.name}


@app.get("/health/live")
async def health_live() -> Dict[str, str]:
    """Liveness endpoint for process/container health."""
    return {"status": "ok", "check": "liveness", "fork": CONFIG.fork.name}


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
        "fork": CONFIG.fork.name,
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
        update_active_sessions(CONFIG.fork.name, len(ACTIVE_SESSIONS))
        
        fork_name = CONFIG.fork.name
        profile_id = ACTIVE_PROFILE_ID
        
        logger.info(
            "Graph invocation received",
            user_id=user_id,
            language=language,
            message_count=len(request.get("messages", [])),
            attachment_count=len(request.get("attachments", [])),
            workspace_document_count=len(request.get("workspace_documents", [])),
        )
        
        # Generate session_id (use request session_id if provided, otherwise derive from user_id + timestamp)
        import time
        session_id = request.get("session_id") or f"{user_id}_{int(time.time())}"
        
        # Prepare graph state
        state: GraphState = {
            "messages": request.get("messages", []),
            "user_id": user_id,
            "session_id": session_id,
            "language": language,
            "fork_name": fork_name,
            "profile_id": profile_id,
            "user_settings": request.get("user_settings", {}),  # Include user settings from request
            # Preserve adapter-provided probe snapshots so Layer 1 can resolve
            # tool-calling mode with probe-aware downgrade logic.
            "llm_capability_probes": request.get("llm_capability_probes", []),
            "attachments": request.get("attachments", []),
            "workspace_documents": request.get("workspace_documents", []),
            "workspace_writeback_requested": bool(request.get("workspace_writeback_requested", False)),
        }
        
        logger.debug("Graph state prepared", session_id=session_id, fork_name=fork_name)
        
        # Track request with metrics
        with track_graph_request(fork_name) as ctx:
            try:
                # Pass per-session thread id so checkpoint-enabled graphs can resume state correctly.
                invoke_config = {
                    "configurable": {
                        "thread_id": session_id,
                    }
                }

                # Execute graph (use invoke since HTTP is sync context)
                result = GRAPH.invoke(state, config=invoke_config)
                
                ctx["status"] = "success"
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
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "provider_used": result.get("provider_used", "unknown"),
                    "model_used": result.get("model_used", "unknown"),
                    "profile_id": result.get("profile_id", profile_id),
                    "summary_text": result.get("summary_text", ""),
                    "sources": result.get("sources", []),
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
