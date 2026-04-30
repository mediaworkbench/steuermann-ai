"""Base crew class with retry, timeout, and validation infrastructure.

All crew implementations should inherit from BaseCrew to get:
- Configurable retry with exponential backoff
- Timeout enforcement
- Structured result validation (CrewResult)
- Prometheus performance metrics tracking
- Unified error handling and categorization
"""

from __future__ import annotations

import logging
import time
import signal
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from universal_agentic_framework.config import load_agents_config, load_core_config, load_features_config
from universal_agentic_framework.config.loader import load_tools_config
from universal_agentic_framework.llm.factory import LLMFactory
from universal_agentic_framework.tools.registry import ToolRegistry
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured result model
# ---------------------------------------------------------------------------

class CrewResult(BaseModel):
    """Structured result from any crew execution."""

    success: bool = False
    result: str = ""
    steps: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    crew_name: str = ""
    execution_time_ms: float = 0.0
    retries_attempted: int = 0
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Transient errors that may succeed on retry
TRANSIENT_ERROR_TYPES = (
    "ConnectionError",
    "TimeoutError",
    "HTTPError",
    "ServiceUnavailableError",
    "RateLimitError",
    "OSError",
    "IOError",
)


def _is_transient_error(exc: Exception) -> bool:
    """Return True if the error is likely transient and retryable."""
    etype = type(exc).__name__
    if etype in TRANSIENT_ERROR_TYPES:
        return True
    # Check cause chain
    cause = exc.__cause__
    while cause:
        if type(cause).__name__ in TRANSIENT_ERROR_TYPES:
            return True
        cause = cause.__cause__
    # Check string patterns in message
    msg = str(exc).lower()
    transient_keywords = ["timeout", "connection", "rate limit", "429", "502", "503", "504"]
    return any(kw in msg for kw in transient_keywords)


# ---------------------------------------------------------------------------
# Base crew class
# ---------------------------------------------------------------------------

class BaseCrew(ABC):
    """Abstract base class providing retry, timeout, validation, and metrics.

    Subclasses must implement:
        crew_name: str  — identifier used in configs, metrics, and caching
        _build_agents()  — construct CrewAI Agent instances
        _build_tasks()   — construct CrewAI Task instances
        _build_crew()    — assemble the Crew
        _kickoff(**kwargs) -> Dict[str, Any]  — execute and return raw result dict
    """

    crew_name: str = "base"

    def __init__(
        self,
        language: str = "en",
        config_dir: Optional[str] = None,
        model_override=None,
        manager_llm_override=None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.language = language
        self.config_dir = config_dir or "config"
        self.model_override = model_override
        self.manager_llm_override = manager_llm_override
        self.tool_registry = tool_registry

        # Load shared configs
        self.core_config = load_core_config(config_dir=self.config_dir)
        self.agents_config = load_agents_config(config_dir=self.config_dir)
        self.tools_config = load_tools_config(config_dir=self.config_dir)

        # LLM factory
        self.llm_factory = LLMFactory(self.core_config)

        # Initialise tool registry if not provided
        if self.tool_registry is None:
            fork_language = getattr(self.core_config.fork, "language", "en")
            self.tool_registry = ToolRegistry(config=self.tools_config, fork_language=fork_language)
            self.tool_registry.discover_and_load()

        # Load crew-specific config for retry / timeout
        crew_cfg = self.agents_config.crews.get(self.crew_name)
        self._max_retries: int = int(getattr(crew_cfg, "max_retries", 3) if crew_cfg else 3)
        self._timeout_seconds: int = int(getattr(crew_cfg, "timeout_seconds", 300) if crew_cfg else 300)
        self._retry_backoff_base: float = float(getattr(crew_cfg, "retry_backoff_base", 2.0) if crew_cfg else 2.0)

        # Internal state for agents / tasks / crew
        self._agents: Dict[str, Any] = {}
        self._tasks: Dict[str, Any] = {}
        self._crew: Optional[Any] = None

        # Let the subclass build agents → tasks → crew
        self._build_agents()
        self._build_tasks()
        self._build_crew()

    # ----- LLM helpers (shared) -----

    def _get_llm(self):
        """Get a crewai.LLM instance. CrewAI v1.x requires crewai.LLM, not LangChain objects."""
        if self.model_override:
            return self.model_override

        try:
            from crewai import LLM as CrewAILLM
        except ImportError as exc:
            raise RuntimeError("crewai package not available") from exc

        providers = self.core_config.llm.providers
        primary = providers.primary
        # Resolve model name for the current language
        llm_factory = self.llm_factory
        model_name: str = llm_factory._select_model(primary, self.language)
        
        # Extract actual model name if it contains a custom provider prefix (e.g., "liquid/lfm2-24b-a2b" -> "lfm2-24b-a2b")
        if "/" in model_name and model_name.count("/") == 1:
            parts = model_name.split("/")
            # Only treat as custom provider if first part is not a known provider
            if parts[0].lower() not in ("openai", "anthropic", "claude", "google", "gemini", "azure", "aws", "bedrock"):
                model_name = parts[1]

        # Build the provider-prefixed model string that crewai.LLM expects
        provider_type = primary.type.lower()
        endpoint = str(primary.endpoint) if primary.endpoint else None

        if provider_type == "ollama":
            # Use OpenAI-compatible mode for Ollama
            base_url = endpoint or "http://localhost:11434"
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            return CrewAILLM(
                model=f"openai/{model_name}",
                base_url=base_url,
                api_key="ollama",
                temperature=primary.temperature,
            )
        elif provider_type == "openai":
            # For OpenAI-compatible endpoints (custom base_url), use model name directly
            # CrewAI will route to the custom endpoint specified in base_url
            kwargs = dict(
                model=model_name,
                temperature=primary.temperature,
            )
            if endpoint:
                kwargs["base_url"] = endpoint
            return CrewAILLM(**kwargs)
        elif provider_type == "anthropic":
            return CrewAILLM(
                model=f"anthropic/{model_name}",
                temperature=primary.temperature,
            )
        else:
            # Generic fallback — pass model string as-is
            return CrewAILLM(
                model=model_name,
                temperature=primary.temperature,
            )

    def _get_tools_for_agent(self, agent_name: str) -> List[Any]:
        """Resolve tool instances for *agent_name* from config.

        Only returns tools that are crewai.tools.BaseTool instances to avoid
        CrewAI v1.x validation errors caused by LangChain or custom tool types.
        """
        crew_cfg = self.agents_config.crews.get(self.crew_name, {})
        if not crew_cfg:
            return []
        agent_def = crew_cfg.agents.get(agent_name, {})
        tool_names = agent_def.tools or [] if agent_def else []
        tools = []

        try:
            from crewai.tools import BaseTool as CrewAIBaseTool
            _crewai_base_tool = CrewAIBaseTool
        except ImportError:
            _crewai_base_tool = None

        for name in tool_names:
            if name not in self.tool_registry.tools:
                logger.warning("Tool not found for agent", tool=name, agent=agent_name, crew=self.crew_name)
                continue
            tool = self.tool_registry.tools[name]
            if _crewai_base_tool and isinstance(tool, _crewai_base_tool):
                tools.append(tool)
            else:
                logger.warning(
                    "Tool skipped — not a crewai.tools.BaseTool instance",
                    tool=name,
                    tool_type=type(tool).__name__,
                    agent=agent_name,
                    crew=self.crew_name,
                )
        return tools

    def _get_manager_llm(self):
        """Get manager LLM for hierarchical crews. Returns None for sequential crews."""
        if self.manager_llm_override:
            return self.manager_llm_override
        crew_cfg = self.agents_config.crews.get(self.crew_name, {})
        if crew_cfg and getattr(crew_cfg, "manager_llm", None):
            return self._get_llm()
        return None

    def _extract_task_results(self) -> Dict[str, str]:
        """Extract intermediate output from each task executed by the crew."""
        results: Dict[str, str] = {}
        for task_name, task in self._tasks.items():
            if hasattr(task, "output") and task.output:
                results[task_name] = str(task.output)
        return results

    # ----- Abstract methods (subclass must implement) -----

    @abstractmethod
    def _build_agents(self) -> None: ...

    @abstractmethod
    def _build_tasks(self) -> None: ...

    @abstractmethod
    def _build_crew(self) -> None: ...

    @abstractmethod
    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        """Execute the crew and return a raw result dict with keys: success, result, steps, error."""
        ...

    # ----- Public API -----

    def kickoff_with_retry(self, **kwargs) -> CrewResult:
        """Execute the crew with retry, timeout, and structured result.

        Retries transient errors with exponential backoff.
        Enforces *timeout_seconds* using a thread-pool executor.
        Returns a validated :class:`CrewResult`.
        """
        last_error: Optional[Exception] = None
        retries = 0
        start_total = time.monotonic()

        for attempt in range(1, self._max_retries + 1):
            retries = attempt - 1
            try:
                raw = self._execute_with_timeout(**kwargs)
                elapsed_ms = (time.monotonic() - start_total) * 1000

                return CrewResult(
                    success=raw.get("success", False),
                    result=raw.get("result", ""),
                    steps=raw.get("steps", {}),
                    error=raw.get("error"),
                    error_type=raw.get("error_type"),
                    crew_name=self.crew_name,
                    execution_time_ms=elapsed_ms,
                    retries_attempted=retries,
                )

            except FuturesTimeoutError:
                elapsed_ms = (time.monotonic() - start_total) * 1000
                logger.warning(
                    "Crew execution timed out",
                    crew=self.crew_name,
                    attempt=attempt,
                    timeout_seconds=self._timeout_seconds,
                )
                return CrewResult(
                    success=False,
                    error=f"Crew timed out after {self._timeout_seconds}s",
                    error_type="TimeoutError",
                    crew_name=self.crew_name,
                    execution_time_ms=elapsed_ms,
                    retries_attempted=retries,
                    timed_out=True,
                )

            except Exception as exc:
                last_error = exc
                if not _is_transient_error(exc) or attempt == self._max_retries:
                    break
                wait = self._retry_backoff_base ** attempt
                logger.warning(
                    "Crew execution failed, retrying",
                    crew=self.crew_name,
                    attempt=attempt,
                    max_retries=self._max_retries,
                    wait_seconds=wait,
                    error=str(exc),
                )
                time.sleep(wait)

        # All retries exhausted
        elapsed_ms = (time.monotonic() - start_total) * 1000
        return CrewResult(
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            error_type=type(last_error).__name__ if last_error else "UnknownError",
            crew_name=self.crew_name,
            execution_time_ms=elapsed_ms,
            retries_attempted=retries,
        )

    def _execute_with_timeout(self, **kwargs) -> Dict[str, Any]:
        """Run ``_kickoff`` in a thread with timeout enforcement."""
        if self._timeout_seconds <= 0:
            return self._kickoff(**kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._kickoff, **kwargs)
            return future.result(timeout=self._timeout_seconds)

    # ----- Legacy compatibility -----

    def kickoff(self, **kwargs) -> Dict[str, Any]:
        """Legacy API — delegates to kickoff_with_retry and returns a plain dict."""
        result = self.kickoff_with_retry(**kwargs)
        return result.model_dump()
