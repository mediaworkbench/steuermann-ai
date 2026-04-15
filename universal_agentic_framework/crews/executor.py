"""Crew parallel executor: run independent crews concurrently.

Uses ``ThreadPoolExecutor`` to execute multiple crews at the same time,
collecting all results into a unified dictionary.  This is invoked from
LangGraph nodes when the routing logic determines that multiple crews
should run simultaneously (e.g. research + analytics on the same query).

Example usage in a LangGraph node:

    from universal_agentic_framework.crews.executor import CrewParallelExecutor

    results = CrewParallelExecutor.execute_parallel(
        crew_names=["research", "analytics"],
        language=state.get("language", "en"),
        kwargs_per_crew={
            "research": {"topic": user_msg},
            "analytics": {"query": user_msg},
        },
    )
"""

from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Type

from universal_agentic_framework.crews.base import BaseCrew, CrewResult
from universal_agentic_framework.crews.chaining import CrewChain
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


class CrewParallelExecutor:
    """Execute multiple crews in parallel and collect results."""

    @staticmethod
    def execute_parallel(
        crew_names: List[str],
        language: str = "en",
        kwargs_per_crew: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: int = 4,
        timeout_seconds: int = 600,
    ) -> Dict[str, CrewResult]:
        """Run *crew_names* concurrently, each with its own kwargs.

        Args:
            crew_names: List of crew identifiers (e.g. ["research", "analytics"]).
            language: Language code for all crews.
            kwargs_per_crew: Optional mapping of crew_name → kickoff kwargs.
                             If a crew is not in this dict, it receives no kwargs.
            max_workers: Max concurrent threads.
            timeout_seconds: Global timeout for the entire parallel batch.

        Returns:
            Dict mapping crew_name → CrewResult for each crew (including failures).
        """
        kwargs_per_crew = kwargs_per_crew or {}
        results: Dict[str, CrewResult] = {}
        start = time.monotonic()

        def _run_crew(name: str) -> tuple[str, CrewResult]:
            try:
                crew_cls = CrewChain._resolve_crew_class(name)
                crew = crew_cls(language=language)
                kw = kwargs_per_crew.get(name, {})
                result = crew.kickoff_with_retry(**kw)
                return name, result
            except Exception as e:
                logger.error("Parallel crew failed", crew=name, error=str(e))
                return name, CrewResult(
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    crew_name=name,
                )

        logger.info("Starting parallel crew execution", crews=crew_names, max_workers=max_workers)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(crew_names))) as pool:
            futures = {pool.submit(_run_crew, name): name for name in crew_names}

            for future in as_completed(futures, timeout=timeout_seconds):
                crew_name = futures[future]
                try:
                    name, result = future.result()
                    results[name] = result
                    logger.info(
                        "Parallel crew completed",
                        crew=name,
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                    )
                except Exception as e:
                    logger.error("Parallel crew future error", crew=crew_name, error=str(e))
                    results[crew_name] = CrewResult(
                        success=False,
                        error=str(e),
                        error_type=type(e).__name__,
                        crew_name=crew_name,
                    )

        total_ms = (time.monotonic() - start) * 1000
        successes = sum(1 for r in results.values() if r.success)
        logger.info(
            "Parallel execution complete",
            total_crews=len(crew_names),
            successes=successes,
            failures=len(crew_names) - successes,
            total_ms=total_ms,
        )
        return results

    @staticmethod
    def merge_results_to_state(
        state: Dict[str, Any],
        results: Dict[str, CrewResult],
    ) -> Dict[str, Any]:
        """Merge parallel crew results into GraphState.

        Stores results in ``state["crew_results"]`` and appends successful
        outputs to ``state["messages"]``.
        """
        state["crew_results"] = state.get("crew_results", {})

        for crew_name, result in results.items():
            state["crew_results"][crew_name] = result.model_dump()

            if result.success:
                label = crew_name.replace("_", " ").title()
                state["messages"] = state.get("messages", [])
                state["messages"].append({
                    "role": "assistant",
                    "content": f"{label} Result:\n{result.result}",
                })

        return state
