"""Crew chaining: execute a sequence of crews, passing output between them.

LangGraph remains the orchestrator — ``CrewChain`` is invoked from a LangGraph
node and returns structured results merged back into ``GraphState``.

Example chain config (agents.yaml):

    crew_chains:
      research_and_analytics:
        description: "Research a topic then analyze findings"
        steps:
          - crew: research
            input_key: topic          # kwarg name for the crew's kickoff
            output_key: research_out  # key in the chain context that receives the result
          - crew: analytics
            input_key: query
            input_from: research_out  # maps output of previous step → this step's input
            output_key: analytics_out
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from universal_agentic_framework.crews.base import BaseCrew, CrewResult
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chain step definition
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """One step in a crew chain."""

    crew_name: str
    input_key: str  # kwarg name used when calling crew.kickoff_with_retry()
    output_key: str  # key to store result in chain context
    input_from: Optional[str] = None  # if set, read input value from this context key
    transform: Optional[str] = None  # optional transform: "result_only" extracts .result


# ---------------------------------------------------------------------------
# Chain context passed between steps
# ---------------------------------------------------------------------------

@dataclass
class ChainContext:
    """Mutable context flowing through a crew chain."""

    initial_input: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, CrewResult] = field(default_factory=dict)
    failed_step: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Crew chain executor
# ---------------------------------------------------------------------------

class CrewChain:
    """Execute a sequence of crew steps, passing output from one to the next.

    Each step:
    1. Resolves the input (from initial_input or previous step output)
    2. Instantiates the crew class
    3. Calls ``crew.kickoff_with_retry(**{input_key: value})``
    4. Stores the :class:`CrewResult` in the chain context
    5. On failure, stops the chain and records the error
    """

    # Lazy registry — maps crew_name → crew class
    _crew_registry: Dict[str, Type[BaseCrew]] = {}

    def __init__(
        self,
        steps: List[ChainStep],
        language: str = "en",
        fail_fast: bool = True,
    ):
        self.steps = steps
        self.language = language
        self.fail_fast = fail_fast  # stop on first failure

    # ---- Crew class resolution ----

    @classmethod
    def register_crew(cls, name: str, crew_class: Type[BaseCrew]) -> None:
        cls._crew_registry[name] = crew_class

    @classmethod
    def _resolve_crew_class(cls, crew_name: str) -> Type[BaseCrew]:
        """Resolve crew name to class, using registry or lazy import."""
        if crew_name in cls._crew_registry:
            return cls._crew_registry[crew_name]

        # Lazy import from crews package
        from universal_agentic_framework.crews import (
            ResearchCrew,
            AnalyticsCrew,
            CodeGenerationCrew,
            PlanningCrew,
        )

        _builtin = {
            "research": ResearchCrew,
            "analytics": AnalyticsCrew,
            "code_generation": CodeGenerationCrew,
            "planning": PlanningCrew,
        }
        if crew_name in _builtin:
            return _builtin[crew_name]

        raise ValueError(f"Unknown crew: {crew_name}. Register it via CrewChain.register_crew().")

    # ---- Execution ----

    def execute(self, initial_input: str) -> ChainContext:
        """Run the full chain and return the populated context."""
        ctx = ChainContext(initial_input=initial_input)
        start = time.monotonic()

        for i, step in enumerate(self.steps):
            step_label = f"step[{i}]:{step.crew_name}"
            logger.info("Chain step starting", step=step_label, crew=step.crew_name)

            # Resolve input value
            if step.input_from and step.input_from in ctx.results:
                prev_result = ctx.results[step.input_from]
                input_value = prev_result.result if prev_result.success else initial_input
            elif step.input_from and step.input_from in ctx.data:
                input_value = ctx.data[step.input_from]
            else:
                input_value = initial_input

            # Apply optional transform
            if step.transform == "result_only" and isinstance(input_value, CrewResult):
                input_value = input_value.result

            # Instantiate crew
            try:
                crew_cls = self._resolve_crew_class(step.crew_name)
                crew = crew_cls(language=self.language)
            except Exception as e:
                logger.error("Failed to instantiate crew", crew=step.crew_name, error=str(e))
                ctx.failed_step = step_label
                ctx.error = str(e)
                if self.fail_fast:
                    break
                continue

            # Execute
            crew_result = crew.kickoff_with_retry(**{step.input_key: input_value})

            # Store result
            ctx.results[step.output_key] = crew_result
            ctx.data[step.output_key] = crew_result.result if crew_result.success else ""

            logger.info(
                "Chain step completed",
                step=step_label,
                success=crew_result.success,
                execution_time_ms=crew_result.execution_time_ms,
            )

            if not crew_result.success and self.fail_fast:
                ctx.failed_step = step_label
                ctx.error = crew_result.error
                logger.warning("Chain stopped on failure", step=step_label, error=crew_result.error)
                break

        total_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Chain execution complete",
            steps_total=len(self.steps),
            steps_executed=len(ctx.results),
            total_ms=total_ms,
            success=ctx.failed_step is None,
        )
        return ctx
