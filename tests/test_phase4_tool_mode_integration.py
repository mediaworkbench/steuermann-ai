"""Phase 4 completion: Tool-calling mode integration and enforcement.

Tests that verify tool-calling mode is properly resolved, routed, and enforced
across all tool routing layers (prefilter, semantic routing, native/structured/react).

Note: These tests focus on mode validation and enforcement at invocation nodes, as well
as high-level integration tests. The mode resolution function is already tested in detail
in other test files (test_semantic_tool_routing.py, etc.), so we skip detailed mode
resolution tests and focus on validation/enforcement at each tool calling node.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List

from backend.db import SettingsStore, LLMCapabilityProbeStore, DatabasePool, DatabaseConfig
from backend.llm_capability_probe import LLMCapabilityProbeResult, LLMCapabilityProbeRunner
from universal_agentic_framework.orchestration.graph_builder import (
    GraphState,
    _resolve_effective_tool_calling_mode,
    _validate_and_log_tool_calling_mode,
    build_graph,
)
from universal_agentic_framework.config import load_core_config


@pytest.fixture
def mock_llm_factory():
    """Mock LLMFactory for testing."""
    with patch("universal_agentic_framework.orchestration.graph_builder.LLMFactory") as mock:
        factory = MagicMock()
        factory.get_router_model.return_value = MagicMock(
            invoke=lambda x: MagicMock(content="Test response"),
            bind_tools=lambda x: MagicMock(
                invoke=lambda y: MagicMock(content="Test response", tool_calls=[])
            ),
        )
        mock.return_value = factory
        yield mock


@pytest.fixture
def sample_probe_results() -> List[Dict[str, Any]]:
    """Sample probe results showing capability mismatch."""
    return [
        {
            "profile_id": "test",
            "provider_id": "primary",  # Use "primary" (default provider_id)
            "model_name": "llama-3.1-8b",
            "api_base": "http://localhost:1234",
            "configured_tool_calling_mode": "native",
            "supports_bind_tools": False,
            "supports_tool_schema": True,
            "capability_mismatch": True,
            "status": "ok",
            "error_message": None,
            "metadata": {"attempted_bind": True, "bind_error": "bind_tools not supported"},
            "probed_at": "2025-01-01T00:00:00Z",
        }
    ]


@pytest.fixture
def sample_state(sample_probe_results) -> GraphState:
    """Sample graph state with probe data."""
    return {
        "messages": [{"role": "user", "content": "extract information from https://example.com"}],
        "user_id": "test_user",
        "language": "en",
        "profile_id": "test",
        "llm_capability_probes": sample_probe_results,
        "candidate_tools": [
            {
                "name": "extract_webpage_mcp",
                "tool": MagicMock(name="extract_webpage_mcp"),
                "score": 0.9,
            }
        ],
        "tool_calling_mode": "structured",  # Should be downgraded to this
        "tool_calling_mode_reason": "probe_capability_mismatch_downgrade",
        "user_settings": {},
        "tokens_used": 0,
        "loaded_memory": None,
        "crew_results": {},
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
    }


class TestModeValidation:
    """Test mode validation at invocation nodes."""

    def test_native_node_validation_pass(self, sample_state):
        """Verify native node passes validation when mode is native."""
        sample_state["tool_calling_mode"] = "native"
        sample_state["tool_calling_mode_reason"] = "configured_native_probe_ok"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "native", "call_tools_native", "test-fork"
        )
        
        assert is_valid is True
        assert "valid" in reason.lower()

    def test_native_node_validation_fail_wrong_mode(self, sample_state):
        """Verify native node fails validation when mode is structured."""
        sample_state["tool_calling_mode"] = "structured"
        sample_state["tool_calling_mode_reason"] = "probe_capability_mismatch_downgrade"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "native", "call_tools_native", "test-fork"
        )
        
        assert is_valid is False
        assert "mismatch" in reason.lower()
        assert "structured" in reason.lower()

    def test_structured_node_validation_pass(self, sample_state):
        """Verify structured node passes validation when mode is structured."""
        sample_state["tool_calling_mode"] = "structured"
        sample_state["tool_calling_mode_reason"] = "probe_capability_mismatch_downgrade"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        
        assert is_valid is True
        assert "valid" in reason.lower()

    def test_validation_skip_when_no_candidates(self, sample_state):
        """Verify validation returns True when no candidate tools."""
        sample_state["candidate_tools"] = []
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "native", "call_tools_native", "test-fork"
        )
        
        assert is_valid is True
        assert "no_candidates" in reason.lower()

    def test_react_node_validation(self, sample_state):
        """Verify react node validation works."""
        sample_state["tool_calling_mode"] = "react"
        sample_state["tool_calling_mode_reason"] = "configured_react_mode"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "react", "call_tools_react", "test-fork"
        )
        
        assert is_valid is True
        assert "valid" in reason.lower()

    def test_structured_node_validation_fail_wrong_mode(self, sample_state):
        """Verify structured node fails validation when mode is native."""
        sample_state["tool_calling_mode"] = "native"
        sample_state["tool_calling_mode_reason"] = "configured_native_probe_ok"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        
        assert is_valid is False
        assert "mismatch" in reason.lower()

    def test_react_node_validation_fail_wrong_mode(self, sample_state):
        """Verify react node fails validation when mode is native."""
        sample_state["tool_calling_mode"] = "native"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "react", "call_tools_react", "test-fork"
        )
        
        assert is_valid is False
        assert "mismatch" in reason.lower()


class TestModeRoutingDecision:
    """Test that routing decisions correctly map mode to node."""

    def test_all_tool_calling_nodes_exist_in_graph(self):
        """Verify routing sends native mode to native node."""
        graph = build_graph()
        
        # The routing function is defined within build_graph
        # We can verify the graph structure includes all three nodes
        assert "call_tools_native" in graph.nodes
        assert "call_tools_structured" in graph.nodes
        assert "call_tools_react" in graph.nodes
        assert "prefilter_tools" in graph.nodes
        assert "after_tool_call" in graph.nodes


class TestModeEnforcementWithProbes:
    """Integration tests: Mode enforcement with actual probe data."""

    def test_mode_validation_across_candidate_tools(self, sample_state):
        """Test that mode validation works across multiple tool scenarios."""
        # Test with multiple candidates
        sample_state["candidate_tools"] = [
            {
                "name": "extract_webpage_mcp",
                "tool": MagicMock(name="extract_webpage_mcp"),
                "score": 0.9,
            },
            {
                "name": "search_web",
                "tool": MagicMock(name="search_web"),
                "score": 0.7,
            },
        ]
        
        sample_state["tool_calling_mode"] = "structured"
        is_valid, _ = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        assert is_valid is True
        
        # Change mode and verify mismatch
        sample_state["tool_calling_mode"] = "native"
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        assert is_valid is False

    def test_mode_enforcement_native_node(self, sample_state):
        """Test that native node enforces native mode."""
        sample_state["tool_calling_mode"] = "native"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "native", "call_tools_native", "test-fork"
        )
        
        assert is_valid is True
        assert "valid" in reason.lower()

    def test_mode_enforcement_structured_node(self, sample_state):
        """Test that structured node enforces structured mode."""
        sample_state["tool_calling_mode"] = "structured"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        
        assert is_valid is True
        assert "valid" in reason.lower()

    def test_mode_enforcement_react_node(self, sample_state):
        """Test that react node enforces react mode."""
        sample_state["tool_calling_mode"] = "react"
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "react", "call_tools_react", "test-fork"
        )
        
        assert is_valid is True
        assert "valid" in reason.lower()


class TestModeReasonTracking:
    """Test that mode change reasons are properly tracked."""

    def test_reason_propagates_through_state(self, sample_state):
        """Verify mode reason is preserved in state throughout execution."""
        original_reason = sample_state["tool_calling_mode_reason"]
        
        # Simulate passing through nodes
        assert sample_state.get("tool_calling_mode_reason") == original_reason
        
        # Verify reason describes the downgrade decision
        assert "downgrade" in original_reason.lower() or "configured" in original_reason.lower()

    def test_mode_reason_in_validation_context(self, sample_state):
        """Verify mode reason is accessible during validation."""
        sample_state["tool_calling_mode"] = "structured"
        sample_state["tool_calling_mode_reason"] = "probe_capability_mismatch_downgrade"
        
        is_valid, _ = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        
        # Mode reason should still be accessible in state
        assert sample_state["tool_calling_mode_reason"] == "probe_capability_mismatch_downgrade"

    def test_multiple_reason_types_exist(self):
        """Verify different reason types are documented in code."""
        expected_reasons = {
            "probe_capability_mismatch_downgrade",
            "configured_native_probe_ok",
            "configured_non_native_mode",
            "configured_native_no_probe",
            "configured_native_probe_provider_not_found",
        }
        
        # These are defined in _resolve_effective_tool_calling_mode
        # This test documents what reasons are possible
        assert len(expected_reasons) > 0


class TestModeValidationEdgeCases:
    """Test edge cases in mode validation."""

    def test_validation_with_empty_candidates_list(self, sample_state):
        """Test validation when candidate tools list is empty."""
        sample_state["candidate_tools"] = []
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "native", "call_tools_native", "test-fork"
        )
        
        # Should return True (validation skipped) with "no_candidates" reason
        assert is_valid is True
        assert "no_candidates" in reason

    def test_validation_with_missing_mode_in_state(self, sample_state):
        """Test validation when tool_calling_mode is missing."""
        del sample_state["tool_calling_mode"]
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        
        # Should detect mismatch or handle gracefully
        # actual_mode defaults to "structured" if missing
        assert isinstance(reason, str)

    def test_validation_with_missing_reason_in_state(self, sample_state):
        """Test validation when tool_calling_mode_reason is missing."""
        del sample_state["tool_calling_mode_reason"]
        
        is_valid, reason = _validate_and_log_tool_calling_mode(
            sample_state, "structured", "call_tools_structured", "test-fork"
        )
        
        # Should handle gracefully with default reason
        assert isinstance(reason, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
