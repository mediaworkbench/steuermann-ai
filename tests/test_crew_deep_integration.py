"""End-to-end tests for Week 31 CrewAI Deep Integration.

Tests cover:
- BaseCrew retry logic with exponential backoff
- BaseCrew timeout enforcement
- CrewResult structured output validation
- CrewChain sequential execution with context passing
- CrewParallelExecutor concurrent crew execution
- Prometheus crew metrics tracking
- Feature flag gating (multi_agent_crews, crew_chaining, crew_parallel_execution)
- Bug-fix verification (copy-paste cache fix, multi_agent_crews flag)
"""

from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from universal_agentic_framework.crews.base import (
    BaseCrew,
    CrewResult,
    _is_transient_error,
)
from universal_agentic_framework.crews.chaining import CrewChain, ChainStep, ChainContext
from universal_agentic_framework.crews.executor import CrewParallelExecutor
from universal_agentic_framework.config.schemas import (
    CrewChainStep as CrewChainStepSchema,
    CrewChainDefinition,
    FeaturesConfig,
)


# ---------------------------------------------------------------------------
# Helpers: concrete crew subclass for testing
# ---------------------------------------------------------------------------


class _FakeCrew(BaseCrew):
    """Minimal BaseCrew subclass for unit-testing the base infrastructure."""

    crew_name = "fake"

    def _build_agents(self) -> None:
        pass

    def _build_tasks(self) -> None:
        pass

    def _build_crew(self) -> None:
        pass

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        return {
            "success": True,
            "result": f"fake result for {kwargs.get('topic', 'no-topic')}",
            "steps": {"step1": "done"},
        }


class _FailingCrew(BaseCrew):
    """Crew that always raises an exception (for retry/timeout testing)."""

    crew_name = "failing"

    def _build_agents(self) -> None:
        pass

    def _build_tasks(self) -> None:
        pass

    def _build_crew(self) -> None:
        pass

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        raise ConnectionError("Simulated transient connection error")


class _SlowCrew(BaseCrew):
    """Crew that sleeps longer than timeout."""

    crew_name = "slow"

    def _build_agents(self) -> None:
        pass

    def _build_tasks(self) -> None:
        pass

    def _build_crew(self) -> None:
        pass

    def _kickoff(self, **kwargs) -> Dict[str, Any]:
        time.sleep(10)
        return {"success": True, "result": "should not reach here"}


# ---------------------------------------------------------------------------
# Shared mock patch for config loading (avoids hitting real YAML files)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_configs(monkeypatch):
    """Patch config loaders so tests run without real config files."""
    from universal_agentic_framework.config import schemas as sch

    fake_core = MagicMock()
    fake_core.fork.name = "test-fork"
    fake_core.fork.language = "en"
    fake_core.llm.providers.primary.type = "mock"
    fake_core.llm.providers.primary.models.en = "mock-model"
    fake_core.llm.providers.primary.temperature = 0.7
    fake_core.llm.providers.primary.max_tokens = 1000
    fake_core.llm.providers.fallback = None

    fake_agents = MagicMock()
    fake_crew_def = MagicMock()
    fake_crew_def.max_retries = 2
    fake_crew_def.timeout_seconds = 2  # Short timeout for testing
    fake_crew_def.retry_backoff_base = 1.1  # Fast backoff for tests
    fake_crew_def.max_iterations = 10
    fake_crew_def.process = "sequential"
    fake_crew_def.agents = {}
    fake_crew_def.manager_llm = None
    fake_crew_def.cache_ttl_seconds = None
    fake_agents.crews = {
        "fake": fake_crew_def,
        "failing": fake_crew_def,
        "slow": fake_crew_def,
        "research": fake_crew_def,
        "analytics": fake_crew_def,
    }
    fake_agents.crew_chains = []

    fake_tools = MagicMock()
    fake_tools.tools = []

    fake_features = MagicMock(spec=sch.FeaturesConfig)
    fake_features.multi_agent_crews = False
    fake_features.crew_result_caching = False
    fake_features.crew_chaining = False
    fake_features.crew_parallel_execution = False

    monkeypatch.setattr("universal_agentic_framework.crews.base.load_core_config", lambda **kw: fake_core)
    monkeypatch.setattr("universal_agentic_framework.crews.base.load_agents_config", lambda **kw: fake_agents)
    monkeypatch.setattr("universal_agentic_framework.crews.base.load_tools_config", lambda **kw: fake_tools)
    monkeypatch.setattr("universal_agentic_framework.crews.base.load_features_config", lambda **kw: fake_features)

    # Patch LLMFactory
    mock_factory = MagicMock()
    mock_factory.return_value.get_model.return_value = MagicMock()
    monkeypatch.setattr("universal_agentic_framework.crews.base.LLMFactory", mock_factory)

    # Patch ToolRegistry
    mock_registry = MagicMock()
    mock_registry.return_value.discover_and_load.return_value = []
    mock_registry.return_value.tools = {}
    monkeypatch.setattr("universal_agentic_framework.crews.base.ToolRegistry", mock_registry)


# ===========================================================================
# CrewResult tests
# ===========================================================================


class TestCrewResult:
    def test_default_values(self):
        r = CrewResult()
        assert r.success is False
        assert r.result == ""
        assert r.steps == {}
        assert r.error is None
        assert r.crew_name == ""
        assert r.execution_time_ms == 0.0
        assert r.retries_attempted == 0
        assert r.timed_out is False

    def test_successful_result(self):
        r = CrewResult(
            success=True,
            result="done",
            crew_name="research",
            execution_time_ms=123.4,
        )
        assert r.success is True
        d = r.model_dump()
        assert d["crew_name"] == "research"
        assert d["execution_time_ms"] == 123.4

    def test_error_result(self):
        r = CrewResult(
            success=False,
            error="boom",
            error_type="RuntimeError",
            crew_name="analytics",
            retries_attempted=2,
        )
        assert r.error == "boom"
        assert r.retries_attempted == 2


# ===========================================================================
# Error classification tests
# ===========================================================================


class TestTransientErrors:
    def test_connection_error_is_transient(self):
        assert _is_transient_error(ConnectionError("refused"))

    def test_timeout_error_is_transient(self):
        assert _is_transient_error(TimeoutError("timed out"))

    def test_value_error_is_not_transient(self):
        assert not _is_transient_error(ValueError("bad value"))

    def test_rate_limit_in_message_is_transient(self):
        assert _is_transient_error(RuntimeError("rate limit exceeded (429)"))

    def test_chained_cause_is_transient(self):
        inner = ConnectionError("refused")
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert _is_transient_error(outer)


# ===========================================================================
# BaseCrew retry + timeout tests
# ===========================================================================


class TestBaseCrewRetry:
    def test_successful_execution(self):
        crew = _FakeCrew(language="en")
        result = crew.kickoff_with_retry(topic="AI trends")
        assert result.success is True
        assert "fake result" in result.result
        assert result.retries_attempted == 0
        assert result.crew_name == "fake"
        assert result.execution_time_ms > 0

    def test_transient_error_retries(self):
        crew = _FailingCrew(language="en")
        result = crew.kickoff_with_retry()
        assert result.success is False
        assert "connection error" in result.error.lower()
        assert result.retries_attempted >= 1

    def test_timeout_enforcement(self):
        crew = _SlowCrew(language="en")
        # _timeout_seconds is 2 from mock config
        result = crew.kickoff_with_retry()
        assert result.success is False
        assert result.timed_out is True
        assert "timed out" in result.error.lower()

    def test_legacy_kickoff_returns_dict(self):
        crew = _FakeCrew(language="en")
        result = crew.kickoff(topic="test")
        assert isinstance(result, dict)
        assert result["success"] is True


# ===========================================================================
# CrewChain tests
# ===========================================================================


class TestCrewChain:
    def test_two_step_chain(self, monkeypatch):
        """Chain: fake → fake, second step receives first step's output."""
        monkeypatch.setattr(
            "universal_agentic_framework.crews.chaining.CrewChain._resolve_crew_class",
            staticmethod(lambda name: _FakeCrew),
        )

        steps = [
            ChainStep(crew_name="fake", input_key="topic", output_key="step1_out"),
            ChainStep(crew_name="fake", input_key="topic", output_key="step2_out", input_from="step1_out"),
        ]
        chain = CrewChain(steps=steps, language="en")
        ctx = chain.execute({"topic": "test"})

        assert ctx.error is None
        assert ctx.failed_step is None
        assert "step1_out" in ctx.results
        assert "step2_out" in ctx.results
        assert ctx.results["step1_out"].success is True
        assert ctx.results["step2_out"].success is True

    def test_chain_fail_fast(self, monkeypatch):
        """If first step fails and fail_fast=True, chain stops immediately."""
        monkeypatch.setattr(
            "universal_agentic_framework.crews.chaining.CrewChain._resolve_crew_class",
            staticmethod(lambda name: _FailingCrew if name == "failing" else _FakeCrew),
        )

        steps = [
            ChainStep(crew_name="failing", input_key="topic", output_key="step1"),
            ChainStep(crew_name="fake", input_key="topic", output_key="step2"),
        ]
        chain = CrewChain(steps=steps, language="en", fail_fast=True)
        ctx = chain.execute({"topic": "test"})

        assert ctx.failed_step is not None and "failing" in ctx.failed_step
        assert ctx.error is not None
        # Second step should not have run
        assert "step2" not in ctx.results


# ===========================================================================
# CrewParallelExecutor tests
# ===========================================================================


class TestCrewParallelExecutor:
    def test_parallel_execution(self, monkeypatch):
        """Two FakeCrew instances run in parallel and both succeed."""
        monkeypatch.setattr(
            "universal_agentic_framework.crews.executor.CrewChain._resolve_crew_class",
            staticmethod(lambda name: _FakeCrew),
        )

        results = CrewParallelExecutor.execute_parallel(
            crew_names=["fake_a", "fake_b"],
            language="en",
            kwargs_per_crew={
                "fake_a": {"topic": "topic_a"},
                "fake_b": {"topic": "topic_b"},
            },
        )

        assert len(results) == 2
        for name, result in results.items():
            assert isinstance(result, CrewResult)
            assert result.success is True

    def test_merge_results_to_state(self):
        """merge_results_to_state adds crew results and messages."""
        state: Dict[str, Any] = {"messages": [], "crew_results": {}}
        results = {
            "research": CrewResult(success=True, result="research output", crew_name="research"),
        }
        updated = CrewParallelExecutor.merge_results_to_state(state, results)
        assert "research" in updated["crew_results"]
        assert len(updated["messages"]) == 1
        assert "Research" in updated["messages"][0]["content"]


# ===========================================================================
# Config schema tests
# ===========================================================================


class TestConfigSchemas:
    def test_crew_chain_definition(self):
        chain = CrewChainDefinition(
            name="research_then_plan",
            steps=[
                CrewChainStepSchema(crew_name="research", input_key="topic", output_key="research_out"),
                CrewChainStepSchema(crew_name="planning", input_key="project_description", output_key="plan_out", input_from="research_out"),
            ],
        )
        assert chain.name == "research_then_plan"
        assert len(chain.steps) == 2
        assert chain.fail_fast is True  # default

    def test_features_config_new_flags(self):
        features = FeaturesConfig(
            multi_agent_crews=True,
            crew_chaining=True,
            crew_parallel_execution=True,
            crew_result_validation=True,
        )
        assert features.crew_chaining is True
        assert features.crew_parallel_execution is True
        assert features.crew_result_validation is True


# ===========================================================================
# Prometheus metrics tests
# ===========================================================================


class TestCrewMetrics:
    def test_crew_execution_metric(self):
        from universal_agentic_framework.monitoring.metrics import (
            track_crew_execution,
            CREW_EXECUTIONS_TOTAL,
        )

        before = CREW_EXECUTIONS_TOTAL.labels(
            fork_name="test", crew_name="research", status="success"
        )._value.get()
        track_crew_execution("test", "research", 1.5, "success")
        after = CREW_EXECUTIONS_TOTAL.labels(
            fork_name="test", crew_name="research", status="success"
        )._value.get()
        assert after == before + 1

    def test_crew_retry_metric(self):
        from universal_agentic_framework.monitoring.metrics import (
            track_crew_retry,
            CREW_RETRIES_TOTAL,
        )

        before = CREW_RETRIES_TOTAL.labels(fork_name="test", crew_name="analytics")._value.get()
        track_crew_retry("test", "analytics")
        after = CREW_RETRIES_TOTAL.labels(fork_name="test", crew_name="analytics")._value.get()
        assert after == before + 1

    def test_crew_timeout_metric(self):
        from universal_agentic_framework.monitoring.metrics import (
            track_crew_timeout,
            CREW_TIMEOUTS_TOTAL,
        )

        before = CREW_TIMEOUTS_TOTAL.labels(fork_name="test", crew_name="slow")._value.get()
        track_crew_timeout("test", "slow")
        after = CREW_TIMEOUTS_TOTAL.labels(fork_name="test", crew_name="slow")._value.get()
        assert after == before + 1


# ===========================================================================
# Feature flag gating tests
# ===========================================================================


class TestFeatureFlagGating:
    """Verify that route_to_*_crew returns False when multi_agent_crews is disabled."""

    def test_research_routing_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.crew_nodes._multi_agent_crews_enabled",
            lambda: False,
        )
        from universal_agentic_framework.orchestration.crew_nodes import route_to_research_crew
        state = {"messages": [{"content": "search the web for AI trends"}]}
        assert route_to_research_crew(state) is False

    def test_research_routing_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.crew_nodes._multi_agent_crews_enabled",
            lambda: True,
        )
        from universal_agentic_framework.orchestration.crew_nodes import route_to_research_crew
        state = {"messages": [{"content": "search the web for AI trends"}]}
        assert route_to_research_crew(state) is True

    def test_analytics_routing_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.crew_nodes._multi_agent_crews_enabled",
            lambda: False,
        )
        from universal_agentic_framework.orchestration.crew_nodes import route_to_analytics_crew
        state = {"messages": [{"content": "analyze the sales trends"}]}
        assert route_to_analytics_crew(state) is False

    def test_code_generation_routing_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.crew_nodes._multi_agent_crews_enabled",
            lambda: False,
        )
        from universal_agentic_framework.orchestration.crew_nodes import route_to_code_generation_crew
        state = {"messages": [{"content": "write a Python function to sort data"}]}
        assert route_to_code_generation_crew(state) is False

    def test_planning_routing_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.crew_nodes._multi_agent_crews_enabled",
            lambda: False,
        )
        from universal_agentic_framework.orchestration.crew_nodes import route_to_planning_crew
        state = {"messages": [{"content": "plan the database migration project"}]}
        assert route_to_planning_crew(state) is False


# ===========================================================================
# Bug-fix verification: node_research_crew cache key fix
# ===========================================================================


class TestCacheBugFix:
    """Verify the copy-paste cache bug (analytics cache checked first) is fixed."""

    def test_research_crew_uses_research_cache(self, monkeypatch):
        """Ensure node_research_crew checks 'research' cache not 'analytics'."""
        from universal_agentic_framework.orchestration import crew_nodes

        cache_calls = []

        def mock_get_cached(crew_name, state, query, language):
            cache_calls.append(crew_name)
            return None

        monkeypatch.setattr(crew_nodes, "_get_cached_crew_result", mock_get_cached)
        monkeypatch.setattr(crew_nodes, "_crew_cache_enabled", lambda: True)
        monkeypatch.setattr(crew_nodes, "load_core_config", lambda **kw: MagicMock(fork=MagicMock(name="test")))

        # Mock the crew import to avoid real initialization
        mock_crew = MagicMock()
        mock_crew_result = CrewResult(success=True, result="mocked", crew_name="research")
        mock_crew.return_value.kickoff_with_retry.return_value = mock_crew_result

        monkeypatch.setattr(
            "universal_agentic_framework.orchestration.crew_nodes.track_node_execution",
            lambda *a, **kw: MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False)),
        )

        state = {
            "messages": [{"content": "research AI"}],
            "language": "en",
        }

        with patch("universal_agentic_framework.crews.ResearchCrew", mock_crew):
            crew_nodes.node_research_crew(state)

        # The FIRST cache check should be for "research", NOT "analytics"
        assert len(cache_calls) >= 1
        assert cache_calls[0] == "research", (
            f"Expected first cache check for 'research' but got '{cache_calls[0]}' "
            "(copy-paste bug not fixed)"
        )
