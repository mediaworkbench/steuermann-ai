"""LangGraph nodes for multi-agent crew orchestration.

This module provides nodes that integrate CrewAI crews into the LangGraph
orchestration pipeline. Currently implements:
- research_crew_node: Routes to Research Crew for web research tasks
- route_to_research_crew: Decision function for routing research queries
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from universal_agentic_framework.config import load_agents_config, load_core_config, load_features_config
from universal_agentic_framework.monitoring.metrics import (
    track_node_execution,
    track_crew_execution,
    track_crew_retry,
    track_crew_timeout,
)
from universal_agentic_framework.monitoring.logging import get_logger
from universal_agentic_framework.orchestration.performance_nodes import get_cache_manager

logger = get_logger(__name__)


class CrewExecutionState(TypedDict, total=False):
    """State for crew execution tracking."""
    crew_name: str
    topic: str
    success: bool
    result: str
    error: Optional[str]
    execution_time_ms: float


def _crew_cache_enabled() -> bool:
    """Check if crew result caching is enabled via features config."""
    try:
        features = load_features_config()
        return bool(getattr(features, "crew_result_caching", False))
    except Exception as e:
        logger.warning("Failed to load features config for crew caching", error=str(e))
        return False


def _get_crew_cache_ttl(crew_name: str) -> int:
    """Get cache TTL for a crew, preferring crew-specific config."""
    default_ttl = 3600
    try:
        features = load_features_config()
        default_ttl = int(getattr(features, "crew_cache_ttl_seconds", default_ttl))
    except Exception as e:
        logger.warning("Failed to load crew cache TTL from features config", error=str(e))

    try:
        agents_config = load_agents_config()
        crew_config = agents_config.crews.get(crew_name)
        crew_ttl = getattr(crew_config, "cache_ttl_seconds", None) if crew_config else None
        return int(crew_ttl) if crew_ttl else default_ttl
    except Exception as e:
        logger.warning("Failed to load crew cache TTL from agents config", error=str(e))
        return default_ttl


def _run_cache_coro(coro):
    """Run an async cache operation from sync context.

    If an event loop is already running in this thread, execute the coroutine in
    a dedicated worker thread so cache operations are not silently skipped.
    """
    import threading

    def _run_in_worker_thread() -> Any:
        result_box: Dict[str, Any] = {}
        error_box: Dict[str, Exception] = {}

        def _worker() -> None:
            try:
                result_box["result"] = asyncio.run(coro)
            except Exception as thread_exc:  # pragma: no cover - defensive branch
                error_box["error"] = thread_exc

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join()

        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("result")

    try:
        asyncio.get_running_loop()
        return _run_in_worker_thread()
    except RuntimeError:
        try:
            return asyncio.run(coro)
        except Exception as e:
            logger.warning("Crew cache operation failed", error=str(e))
            return None
    except Exception as e:
        logger.warning("Crew cache operation failed", error=str(e))
        return None


def _get_cached_crew_result(
    crew_name: str,
    state: Dict[str, Any],
    query: str,
    language: str,
) -> Optional[Dict[str, Any]]:
    """Attempt to fetch cached crew result."""
    if not _crew_cache_enabled() or not query:
        return None

    cache = get_cache_manager()
    user_id = state.get("user_id", "unknown")
    return _run_cache_coro(cache.get_crew_result(crew_name, user_id, query, language))


def _store_cached_crew_result(
    crew_name: str,
    state: Dict[str, Any],
    query: str,
    language: str,
    result: Dict[str, Any],
) -> None:
    """Store crew result in cache if enabled."""
    if not _crew_cache_enabled() or not query:
        return

    cache = get_cache_manager()
    user_id = state.get("user_id", "unknown")
    ttl_seconds = _get_crew_cache_ttl(crew_name)
    _run_cache_coro(
        cache.set_crew_result(crew_name, user_id, query, result, language, ttl_seconds)
    )


def _multi_agent_crews_enabled() -> bool:
    """Check if multi-agent crews are enabled via features config."""
    try:
        features = load_features_config()
        return bool(getattr(features, "multi_agent_crews", False))
    except Exception:
        return False


def route_to_research_crew(state: Dict[str, Any]) -> bool:
    """Determine if the user query should be routed to the research crew.
    
    The research crew is invoked if the query contains research-related keywords
    or follows patterns typical of research requests (search queries, fact-finding,
    current information requests, etc.).
    
    Routing keywords checked:
    - English: search, research, find, latest, recent, look up, what is, how to
    - German: suche, recherche, finde, neueste, aktuelle, nachschlage, informieren
    
    Args:
        state: GraphState dictionary with 'messages'
        
    Returns:
        True if query should be routed to research crew, False otherwise
    """
    if not _multi_agent_crews_enabled():
        return False

    messages = state.get("messages", [])
    if not messages:
        return False
    
    user_msg = messages[-1].get("content", "").lower()
    
    # Research-specific keywords that indicate a research task
    research_keywords_en = [
        "search",
        "research",
        "find",
        "look up",
        "latest",
        "recent",
        "current",
        "what is",
        "how to",
        "find out",
        "tell me about",
        "information about",
        "news about",
    ]
    
    research_keywords_de = [
        "suche",
        "recherche",
        "finde",
        "neueste",
        "aktuelle",
        "nachschlage",
        "informieren",
        "sag mir",
        "was ist",
        "wie",
        "recherchieren",
    ]
    
    all_keywords = research_keywords_en + research_keywords_de
    
    # Check if any research keyword is in the message
    for keyword in all_keywords:
        if keyword in user_msg:
            logger.info(
                "Research routing triggered",
                detected_keyword=keyword,
                message_length=len(user_msg),
            )
            return True
    
    # Pattern matching for typical research query structures
    # "what is X", "how does X work", "tell me about X"
    research_patterns = [
        r"^\s*(what|how|why|where|when)",  # Question-based
        r"(find|search|look).*?(for|about|on)",  # Search-based
        r"(wie|was|warum|wo|wann)",  # German questions
        r"(?:search|suche).*?(web|internet|online)",  # Explicit search
    ]
    
    for pattern in research_patterns:
        if re.search(pattern, user_msg, re.IGNORECASE):
            logger.info(
                "Research routing triggered by pattern",
                pattern=pattern,
            )
            return True
    
    # Default: do not route to research crew
    return False


def node_research_crew(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Research Crew for web research queries.
    
    This node:
    1. Detects if the user query is a research request
    2. Extracts the research topic from the query
    3. Invokes the Research Crew (Searcher → Analyst → Writer)
    4. Returns structured research results in the state
    
    Args:
        state: GraphState dictionary with 'messages' and 'language'
        
    Returns:
        Modified state with crew_results and research output added
    """
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    
    # Extract user message
    messages = state.get("messages", [])
    if not messages:
        return state
    
    user_msg = messages[-1].get("content", "")
    language = state.get("language", "en")

    cached_result = _get_cached_crew_result("research", state, user_msg, language)
    if cached_result:
        logger.info(
            "Research crew cache hit",
            fork_name=fork_name,
            language=language,
        )
        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["research"] = cached_result
        if cached_result.get("success"):
            state["messages"].append({
                "role": "assistant",
                "content": f"Research Result:\n{cached_result.get('result', 'No result')}",
            })
        return state
    
    logger.info(
        "Research crew node invoked",
        fork_name=fork_name,
        message_length=len(user_msg),
        language=language,
    )
    
    with track_node_execution(fork_name, "research_crew"):
        try:
            from universal_agentic_framework.crews import ResearchCrew
            
            crew = ResearchCrew(language=language)
            crew_result = crew.kickoff_with_retry(topic=user_msg)
            result = crew_result.model_dump()
            
            # Track crew metrics
            status = "success" if crew_result.success else ("timeout" if crew_result.timed_out else "error")
            track_crew_execution(fork_name, "research", crew_result.execution_time_ms / 1000, status)
            for _ in range(crew_result.retries_attempted):
                track_crew_retry(fork_name, "research")
            if crew_result.timed_out:
                track_crew_timeout(fork_name, "research")
            
            # Store crew results in state
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["research"] = result
            
            if result.get("success"):
                logger.info(
                    "Research crew completed successfully",
                    fork_name=fork_name,
                    language=language,
                )
                _store_cached_crew_result("research", state, user_msg, language, result)
                # Optionally append crew result to messages for LLM context
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Research Result:\n{result.get('result', 'No result')}",
                })
            else:
                logger.error(
                    "Research crew execution failed",
                    fork_name=fork_name,
                    error=result.get("error"),
                )
            
            return state
            
        except Exception as e:
            logger.error(
                "Research crew node error",
                fork_name=fork_name,
                error=str(e),
                exc_info=True,
            )
            # Store error in state but don't crash the graph
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["research"] = {
                "success": False,
                "error": str(e),
            }
            return state


def route_to_analytics_crew(state: Dict[str, Any]) -> bool:
    """Determine if the user query should be routed to the analytics crew.
    
    The analytics crew is invoked if the query contains analytics-related keywords
    or follows patterns typical of data analysis requests (pattern finding, trends,
    statistical analysis, etc.).
    
    Routing keywords checked:
    - English: analyze, analysis, trend, pattern, statistic, calculate, correlation, insight, metric
    - German: analysiere, analyse, trend, muster, statistik, berechne, korrelation, einsicht, metrik
    
    Args:
        state: GraphState dictionary with 'messages'
        
    Returns:
        True if query should be routed to analytics crew, False otherwise
    """
    if not _multi_agent_crews_enabled():
        return False

    messages = state.get("messages", [])
    if not messages:
        return False
    
    user_msg = messages[-1].get("content", "").lower()
    
    # Analytics-specific keywords that indicate a data analysis task
    analytics_keywords_en = [
        "analyze",
        "analysis",
        "trend",
        "trends",
        "pattern",
        "patterns",
        "statistic",
        "statistics",
        "statistical",
        "calculate",
        "compute",
        "correlation",
        "insight",
        "insights",
        "metric",
        "metrics",
        "data",
        "dataset",
        "distribution",
        "average",
        "mean",
        "median",
        "report",
        "summary",
        "compare",
        "comparison",
        "visualize",
        "chart",
        "graph",
        "dashboard",
        "performance",
        "forecast",
        "predict",
        "anomaly",
        "detect",
    ]
    
    analytics_keywords_de = [
        "analysiere",
        "analyse",
        "trend",
        "muster",
        "statistik",
        "statistisch",
        "berechne",
        "korrelation",
        "einsicht",
        "metrik",
        "daten",
        "datensatz",
        "verteilung",
        "durchschnitt",
        "mittelwert",
        "median",
        "bericht",
        "zusammenfassung",
        "vergleiche",
        "vergleich",
        "visualisiere",
        "diagramm",
        "grafik",
        "dashboard",
        "leistung",
        "prognose",
        "vorhersage",
        "anomalie",
        "erkennen",
    ]
    
    all_keywords = analytics_keywords_en + analytics_keywords_de
    
    # Check if any analytics keyword is in the message
    for keyword in all_keywords:
        if keyword in user_msg:
            logger.info(
                "Analytics routing triggered",
                detected_keyword=keyword,
                message_length=len(user_msg),
            )
            return True
    
    # Pattern matching for typical analytics query structures
    # "analyze X", "what trends in X", "compare X and Y"
    analytics_patterns = [
        r"(analyze|analyse|analysis).*?(data|trend|pattern|metric)",
        r"(what|which|how).*?(trend|pattern|correlation|insight)",
        r"(compare|vergleiche).*?(with|to|vs|gegen)",
        r"(calculate|compute|berechne).*?(average|mean|sum|total)",
        r"(show|zeige).*?(trend|pattern|distribution|chart|graph)",
        r"(find|finde).*?(pattern|anomaly|outlier|correlation)",
    ]
    
    for pattern in analytics_patterns:
        if re.search(pattern, user_msg, re.IGNORECASE):
            logger.info(
                "Analytics routing triggered by pattern",
                pattern=pattern,
            )
            return True
    
    # Default: do not route to analytics crew
    return False


def node_analytics_crew(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Analytics Crew for data analysis queries.
    
    This node:
    1. Detects if the user query is a data analysis request
    2. Extracts the analysis query from the message
    3. Invokes the Analytics Crew (Data Analyst → Statistician → Report Writer)
    4. Returns structured analysis results in the state
    
    The analytics crew uses a hierarchical process where a manager coordinates
    the three agents to perform comprehensive data analysis, statistical validation,
    and business reporting.
    
    Args:
        state: GraphState dictionary with 'messages' and 'language'
        
    Returns:
        Modified state with crew_results and analysis output added
    """
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    
    # Extract user message
    messages = state.get("messages", [])
    if not messages:
        return state
    
    user_msg = messages[-1].get("content", "")
    language = state.get("language", "en")
    
    logger.info(
        "Analytics crew node invoked",
        fork_name=fork_name,
        message_length=len(user_msg),
        language=language,
    )
    
    cached_result = _get_cached_crew_result("analytics", state, user_msg, language)
    if cached_result:
        logger.info(
            "Analytics crew cache hit",
            fork_name=fork_name,
            language=language,
        )
        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["analytics"] = cached_result
        if cached_result.get("success"):
            state["messages"].append({
                "role": "assistant",
                "content": f"Analysis Result:\n{cached_result.get('result', 'No result')}",
            })
        return state
    
    with track_node_execution(fork_name, "analytics_crew"):
        try:
            from universal_agentic_framework.crews import AnalyticsCrew
            
            crew = AnalyticsCrew(language=language)
            crew_result = crew.kickoff_with_retry(query=user_msg)
            result = crew_result.model_dump()
            
            status = "success" if crew_result.success else ("timeout" if crew_result.timed_out else "error")
            track_crew_execution(fork_name, "analytics", crew_result.execution_time_ms / 1000, status)
            for _ in range(crew_result.retries_attempted):
                track_crew_retry(fork_name, "analytics")
            if crew_result.timed_out:
                track_crew_timeout(fork_name, "analytics")
            
            # Store crew results in state
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["analytics"] = result
            
            if result.get("success"):
                logger.info(
                    "Analytics crew completed successfully",
                    fork_name=fork_name,
                    language=language,
                )
                _store_cached_crew_result("analytics", state, user_msg, language, result)
                # Append crew result to messages for LLM context
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Analysis Result:\n{result.get('result', 'No result')}",
                })
            else:
                logger.error(
                    "Analytics crew execution failed",
                    fork_name=fork_name,
                    error=result.get("error"),
                )
            
            return state
            
        except Exception as e:
            logger.error(
                "Analytics crew node error",
                fork_name=fork_name,
                error=str(e),
                exc_info=True,
            )
            # Store error in state but don't crash the graph
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["analytics"] = {
                "success": False,
                "error": str(e),
            }
            return state


def route_to_code_generation_crew(state: Dict[str, Any]) -> bool:
    """Determine if the user query should be routed to the code generation crew.
    
    The code generation crew is invoked if the query contains code-related keywords
    or follows patterns typical of software development requests (implementing functions,
    writing classes, creating APIs, generating tests, etc.).
    
    Routing keywords checked:
    - English: code, implement, write, function, class, API, create, generate, develop, refactor
    - German: code, implementiere, schreibe, funktion, klasse, erstelle, generiere, entwickle
    
    Args:
        state: GraphState dictionary with 'messages'
        
    Returns:
        True if query should be routed to code generation crew, False otherwise
    """
    if not _multi_agent_crews_enabled():
        return False

    messages = state.get("messages", [])
    if not messages:
        return False
    
    user_msg = messages[-1].get("content", "").lower()
    
    # Code generation-specific keywords that indicate a development task
    code_keywords_en = [
        "code",
        "implement",
        "implementation",
        "write",
        "create",
        "generate",
        "develop",
        "build",
        "function",
        "method",
        "class",
        "interface",
        "api",
        "endpoint",
        "rest",
        "microservice",
        "service",
        "module",
        "library",
        "component",
        "algorithm",
        "script",
        "program",
        "application",
        "test",
        "unit test",
        "integration test",
        "refactor",
        "optimize",
        "debug",
        "fix",
        "database",
        "query",
        "schema",
        "model",
        "dto",
        "validator",
        "parser",
        "handler",
        "controller",
        "route",
        "middleware",
    ]
    
    code_keywords_de = [
        "code",
        "implementiere",
        "implementierung",
        "schreibe",
        "erstelle",
        "generiere",
        "entwickle",
        "funktion",
        "methode",
        "klasse",
        "schnittstelle",
        "api",
        "endpunkt",
        "modul",
        "bibliothek",
        "komponente",
        "algorithmus",
        "skript",
        "programm",
        "anwendung",
        "test",
        "unittest",
        "refaktoriere",
        "optimiere",
        "debugging",
        "datenbank",
        "abfrage",
        "schema",
        "modell",
    ]
    
    all_keywords = code_keywords_en + code_keywords_de
    
    # Check if any code generation keyword is in the message
    for keyword in all_keywords:
        if keyword in user_msg:
            logger.info(
                "Code generation routing triggered",
                detected_keyword=keyword,
                message_length=len(user_msg),
            )
            return True
    
    # Pattern matching for typical code request structures
    # "write a function that", "create a class for", "implement X", "build an API"
    code_patterns = [
        r"(write|create|generate|build|implement).*?(function|method|class|api|service|module|component)",
        r"(how to|wie).*?(code|implement|write|create|build)",
        r"(need|want|require).*?(function|class|api|code|implementation)",
        r"(can you|could you|kannst du).*?(write|create|implement|build|code)",
        r"(make|erstelle).*?(function|class|api|rest|endpoint|service)",
        r"(develop|entwickle).*?(application|api|service|module|component)",
        r"(test|unittest).*?(for|für).*?(function|class|module|api)",
        r"(refactor|optimize|fix|debug).*?(code|function|class|module)",
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, user_msg, re.IGNORECASE):
            logger.info(
                "Code generation routing triggered by pattern",
                pattern=pattern,
            )
            return True
    
    # Reject if it's clearly not a code generation request
    # (greetings, general questions, etc.)
    non_code_patterns = [
        r"^(hi|hello|hey|hallo|guten tag)",
        r"(how are you|wie geht)",
        r"(thank|danke)",
        r"(what is the weather|wie ist das wetter)",
    ]
    
    for pattern in non_code_patterns:
        if re.search(pattern, user_msg, re.IGNORECASE):
            return False
    
    # Default: do not route to code generation crew
    return False


def node_code_generation_crew(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Code Generation Crew for software development queries.
    
    This node:
    1. Detects if the user query is a code generation request
    2. Extracts the requirement from the message
    3. Invokes the Code Generation Crew (Architect → Developer → QA Engineer)
    4. Returns code implementation, tests, and QA report in the state
    
    The code generation crew uses a sequential process where:
    - Architect designs the technical solution and specifications
    - Developer implements clean, production-ready code
    - QA Engineer reviews code quality and writes comprehensive tests
    
    Args:
        state: GraphState dictionary with 'messages' and 'language'
        
    Returns:
        Modified state with crew_results and code output added
    """
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    
    # Extract user message
    messages = state.get("messages", [])
    if not messages:
        return state
    
    user_msg = messages[-1].get("content", "")
    language = state.get("language", "en")
    
    logger.info(
        "Code generation crew node invoked",
        fork_name=fork_name,
        message_length=len(user_msg),
        language=language,
    )
    
    cached_result = _get_cached_crew_result("code_generation", state, user_msg, language)
    if cached_result:
        logger.info(
            "Code generation crew cache hit",
            fork_name=fork_name,
            language=language,
        )
        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["code_generation"] = cached_result
        if cached_result.get("success"):
            steps = cached_result.get("steps", {})
            output_parts = []
            if "design" in steps:
                output_parts.append(f"## Technical Design\n{steps['design']}")
            if "implementation" in steps:
                output_parts.append(f"## Implementation\n{steps['implementation']}")
            if "qa_testing" in steps:
                output_parts.append(f"## QA Report & Tests\n{steps['qa_testing']}")
            formatted_output = "\n\n".join(output_parts) if output_parts else cached_result.get('result', 'No result')
            state["messages"].append({
                "role": "assistant",
                "content": f"Code Generation Result:\n\n{formatted_output}",
            })
        return state
    
    with track_node_execution(fork_name, "code_generation_crew"):
        try:
            from universal_agentic_framework.crews import CodeGenerationCrew
            
            crew = CodeGenerationCrew(language=language)
            crew_result = crew.kickoff_with_retry(requirement=user_msg)
            result = crew_result.model_dump()
            
            status = "success" if crew_result.success else ("timeout" if crew_result.timed_out else "error")
            track_crew_execution(fork_name, "code_generation", crew_result.execution_time_ms / 1000, status)
            for _ in range(crew_result.retries_attempted):
                track_crew_retry(fork_name, "code_generation")
            if crew_result.timed_out:
                track_crew_timeout(fork_name, "code_generation")
            
            # Store crew results in state
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["code_generation"] = result
            
            if result.get("success"):
                logger.info(
                    "Code generation crew completed successfully",
                    fork_name=fork_name,
                    language=language,
                )
                _store_cached_crew_result("code_generation", state, user_msg, language, result)
                
                # Format output with design, implementation, and QA results
                steps = result.get("steps", {})
                output_parts = []
                
                if "design" in steps:
                    output_parts.append(f"## Technical Design\n{steps['design']}")
                
                if "implementation" in steps:
                    output_parts.append(f"## Implementation\n{steps['implementation']}")
                
                if "qa_testing" in steps:
                    output_parts.append(f"## QA Report & Tests\n{steps['qa_testing']}")
                
                formatted_output = "\n\n".join(output_parts) if output_parts else result.get('result', 'No result')
                
                # Append crew result to messages for LLM context
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Code Generation Result:\n\n{formatted_output}",
                })
            else:
                logger.error(
                    "Code generation crew execution failed",
                    fork_name=fork_name,
                    error=result.get("error"),
                )
            
            return state
            
        except Exception as e:
            logger.error(
                "Code generation crew node error",
                fork_name=fork_name,
                error=str(e),
                exc_info=True,
            )
            # Store error in state but don't crash the graph
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["code_generation"] = {
                "success": False,
                "error": str(e),
            }
            return state


def route_to_planning_crew(state: Dict[str, Any]) -> bool:
    """Determine if the user query should be routed to the planning crew.
    
    The planning crew is invoked if the query contains planning-related keywords
    or follows patterns typical of project planning requests (task breakdown,
    roadmap creation, sprint planning, etc.).
    
    Routing keywords checked:
    - English: plan, planning, roadmap, task, breakdown, sprint, milestone, project, schedule
    - German: plan, planung, fahrplan, aufgabe, sprint, meilenstein, projekt, zeitplan
    
    Args:
        state: GraphState dictionary with 'messages'
        
    Returns:
        True if query should be routed to planning crew, False otherwise
    """
    if not _multi_agent_crews_enabled():
        return False

    messages = state.get("messages", [])
    if not messages:
        return False
    
    user_msg = messages[-1].get("content", "").lower()
    
    # Planning-specific keywords that indicate a planning task
    planning_keywords_en = [
        "plan",
        "planning",
        "roadmap",
        "task",
        "tasks",
        "breakdown",
        "break down",
        "decompose",
        "sprint",
        "iteration",
        "milestone",
        "milestones",
        "project plan",
        "execution plan",
        "schedule",
        "timeline",
        "backlog",
        "epic",
        "user story",
        "story",
        "estimate",
        "estimation",
        "scope",
        "scoping",
        "phase",
        "phases",
        "initiative",
        "deliverable",
        "deliverables",
        "dependency",
        "dependencies",
        "critical path",
        "gantt",
        "workflow",
        "organize",
        "structure",
    ]
    
    planning_keywords_de = [
        "plan",
        "planung",
        "fahrplan",
        "roadmap",
        "aufgabe",
        "aufgaben",
        "aufschlüsseln",
        "sprint",
        "iteration",
        "meilenstein",
        "meilensteine",
        "projektplan",
        "zeitplan",
        "backlog",
        "schätzung",
        "umfang",
        "phase",
        "phasen",
        "abhängigkeit",
        "abhängigkeiten",
        "kritischer pfad",
        "arbeitsablauf",
        "organisieren",
        "struktur",
    ]
    
    all_keywords = planning_keywords_en + planning_keywords_de
    
    # Check if any planning keyword is in the message
    for keyword in all_keywords:
        if keyword in user_msg:
            logger.info(
                "Planning routing triggered",
                detected_keyword=keyword,
                message_length=len(user_msg),
            )
            return True
    
    # Pattern matching for typical planning request structures
    # "plan X", "create a roadmap for", "break down into tasks", "estimate X"
    planning_patterns = [
        r"(plan|planning).*?(project|migration|initiative|feature|epic)",
        r"(create|make|build).*?(plan|roadmap|timeline|schedule|backlog)",
        r"(break down|breakdown|decompose).*?(into|to).*?(task|story|epic|phase)",
        r"(estimate|estimation).*?(effort|time|duration|cost)",
        r"(need|want|require).*?(plan|roadmap|timeline|breakdown)",
        r"(how to|wie).*?(plan|organize|structure|schedule)",
        r"(sprint|iteration).*?(planning|plan|backlog)",
        r"(organize|structure).*?(work|tasks|project|initiative)",
        r"(identify|find|determine).*?(dependency|dependencies|milestone)",
    ]
    
    for pattern in planning_patterns:
        if re.search(pattern, user_msg, re.IGNORECASE):
            logger.info(
                "Planning routing triggered by pattern",
                pattern=pattern,
            )
            return True
    
    # Reject if it's clearly not a planning request
    # (greetings, simple questions, etc.)
    non_planning_patterns = [
        r"^(hi|hello|hey|hallo|guten tag)",
        r"(how are you|wie geht)",
        r"(thank|danke)",
    ]
    
    for pattern in non_planning_patterns:
        if re.search(pattern, user_msg, re.IGNORECASE):
            return False
    
    # Default: do not route to planning crew
    return False


def node_planning_crew(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Planning Crew for project planning queries.
    
    This node:
    1. Detects if the user query is a planning request
    2. Extracts the project description from the message
    3. Invokes the Planning Crew (Analyst → Planner → Reviewer)
    4. Returns planning output with task breakdown, dependencies, and risk assessment
    
    The planning crew uses a hierarchical process where a manager coordinates
    the three agents to perform requirements analysis, task decomposition, and plan review.
    
    Args:
        state: GraphState dictionary with 'messages' and 'language'
        
    Returns:
        Modified state with crew_results and planning output added
    """
    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    
    # Extract user message
    messages = state.get("messages", [])
    if not messages:
        return state
    
    user_msg = messages[-1].get("content", "")
    language = state.get("language", "en")
    
    logger.info(
        "Planning crew node invoked",
        fork_name=fork_name,
        message_length=len(user_msg),
        language=language,
    )
    
    cached_result = _get_cached_crew_result("planning", state, user_msg, language)
    if cached_result:
        logger.info(
            "Planning crew cache hit",
            fork_name=fork_name,
            language=language,
        )
        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["planning"] = cached_result
        if cached_result.get("success"):
            steps = cached_result.get("steps", {})
            output_parts = []
            if "analysis" in steps:
                output_parts.append(f"## Requirements Analysis\n{steps['analysis']}")
            if "planning" in steps:
                output_parts.append(f"## Execution Plan\n{steps['planning']}")
            if "review" in steps:
                output_parts.append(f"## Plan Review & Risk Assessment\n{steps['review']}")
            formatted_output = "\n\n".join(output_parts) if output_parts else cached_result.get('result', 'No result')
            state["messages"].append({
                "role": "assistant",
                "content": f"Planning Result:\n\n{formatted_output}",
            })
        return state
    
    with track_node_execution(fork_name, "planning_crew"):
        try:
            from universal_agentic_framework.crews import PlanningCrew
            
            crew = PlanningCrew(language=language)
            crew_result = crew.kickoff_with_retry(project_description=user_msg)
            result = crew_result.model_dump()
            
            status = "success" if crew_result.success else ("timeout" if crew_result.timed_out else "error")
            track_crew_execution(fork_name, "planning", crew_result.execution_time_ms / 1000, status)
            for _ in range(crew_result.retries_attempted):
                track_crew_retry(fork_name, "planning")
            if crew_result.timed_out:
                track_crew_timeout(fork_name, "planning")
            
            # Store crew results in state
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["planning"] = result
            
            if result.get("success"):
                logger.info(
                    "Planning crew completed successfully",
                    fork_name=fork_name,
                    language=language,
                )
                _store_cached_crew_result("planning", state, user_msg, language, result)
                
                # Format output with analysis, planning, and review results
                steps = result.get("steps", {})
                output_parts = []
                
                if "analysis" in steps:
                    output_parts.append(f"## Requirements Analysis\n{steps['analysis']}")
                
                if "planning" in steps:
                    output_parts.append(f"## Execution Plan\n{steps['planning']}")
                
                if "review" in steps:
                    output_parts.append(f"## Plan Review & Risk Assessment\n{steps['review']}")
                
                formatted_output = "\n\n".join(output_parts) if output_parts else result.get('result', 'No result')
                
                # Append crew result to messages for LLM context
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Planning Result:\n\n{formatted_output}",
                })
            else:
                logger.error(
                    "Planning crew execution failed",
                    fork_name=fork_name,
                    error=result.get("error"),
                )
            
            return state
            
        except Exception as e:
            logger.error(
                "Planning crew node error",
                fork_name=fork_name,
                error=str(e),
                exc_info=True,
            )
            # Store error in state but don't crash the graph
            state["crew_results"] = state.get("crew_results", {})
            state["crew_results"]["planning"] = {
                "success": False,
                "error": str(e),
            }
            return state


# ─── Crew chain & parallel execution nodes ─────────────────────────────


def node_crew_chain(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a configured crew chain (sequential multi-crew pipeline).

    Reads ``crew_chains`` from agents config and runs the first enabled chain
    whose name matches the ``chain_name`` key in ``state["crew_results"]``,
    or falls through to a default chain if configured.

    Feature-gated by ``crew_chaining`` in features.yaml.
    """
    try:
        features = load_features_config()
        if not getattr(features, "crew_chaining", False):
            logger.debug("Crew chaining feature disabled, skipping")
            return state
    except Exception:
        return state

    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    messages = state.get("messages", [])
    if not messages:
        return state

    user_msg = messages[-1].get("content", "")
    language = state.get("language", "en")

    try:
        from universal_agentic_framework.crews.chaining import CrewChain, ChainStep
        from universal_agentic_framework.monitoring.metrics import track_crew_chain

        agents_cfg = load_agents_config()
        chains = getattr(agents_cfg, "crew_chains", [])
        if not chains:
            return state

        # Pick first enabled chain (extensible to routing-based selection later)
        chain_def = next((c for c in chains if c.enabled), None)
        if not chain_def:
            return state

        steps = [
            ChainStep(
                crew_name=s.crew_name,
                input_key=s.input_key,
                output_key=s.output_key,
                input_from=s.input_from,
                transform=s.transform,
            )
            for s in chain_def.steps
        ]
        chain = CrewChain(
            steps=steps,
            language=language,
            fail_fast=chain_def.fail_fast,
        )

        with track_crew_chain(fork_name, chain_def.name):
            ctx = chain.execute({"topic": user_msg, "query": user_msg, "requirement": user_msg, "project_description": user_msg})

        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["chain"] = {
            "chain_name": chain_def.name,
            "results": {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in ctx.results.items()},
            "failed_step": ctx.failed_step,
            "error": ctx.error,
        }

        # Append final chain output to messages
        final_step = chain_def.steps[-1]
        final_result = ctx.results.get(final_step.crew_name)
        if final_result and getattr(final_result, "success", False):
            state["messages"].append({
                "role": "assistant",
                "content": f"Chain Result ({chain_def.name}):\n{final_result.result}",
            })

        return state

    except Exception as e:
        logger.error("Crew chain node error", error=str(e), exc_info=True)
        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["chain"] = {"success": False, "error": str(e)}
        return state


def node_crew_parallel(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute multiple crews in parallel for a single query.

    Feature-gated by ``crew_parallel_execution`` in features.yaml.
    Currently invokes research + analytics in parallel as a proof-of-concept.
    Extend this with config-driven crew selection.
    """
    try:
        features = load_features_config()
        if not getattr(features, "crew_parallel_execution", False):
            logger.debug("Crew parallel execution feature disabled, skipping")
            return state
    except Exception:
        return state

    config = load_core_config()
    fork_name = getattr(config.fork, "name", "default-fork")
    messages = state.get("messages", [])
    if not messages:
        return state

    user_msg = messages[-1].get("content", "")
    language = state.get("language", "en")

    try:
        from universal_agentic_framework.crews.executor import CrewParallelExecutor
        from universal_agentic_framework.monitoring.metrics import track_crew_parallel

        # Default: run research + analytics in parallel (configurable later)
        crew_names = ["research", "analytics"]
        kwargs_per_crew = {
            "research": {"topic": user_msg},
            "analytics": {"query": user_msg},
        }

        with track_crew_parallel(fork_name):
            results = CrewParallelExecutor.execute_parallel(
                crew_names=crew_names,
                language=language,
                kwargs_per_crew=kwargs_per_crew,
            )

        state = CrewParallelExecutor.merge_results_to_state(state, results)
        return state

    except Exception as e:
        logger.error("Crew parallel node error", error=str(e), exc_info=True)
        state["crew_results"] = state.get("crew_results", {})
        state["crew_results"]["parallel"] = {"success": False, "error": str(e)}
        return state

