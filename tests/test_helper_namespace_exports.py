"""Regression tests for helper namespace export clarity."""

import inspect

from universal_agentic_framework.orchestration.helpers import (
    build_prefilter_tool_kwargs,
    build_semantic_tool_kwargs,
)


def test_prefilter_and_semantic_kwargs_builders_are_distinct_exports():
    """Prefilter and semantic kwargs builders must remain separate helpers."""
    assert build_prefilter_tool_kwargs is not build_semantic_tool_kwargs
    assert build_prefilter_tool_kwargs.__module__.endswith("helpers.tool_scoring")
    assert build_semantic_tool_kwargs.__module__.endswith("helpers.semantic_execution")


def test_kwargs_builder_signatures_remain_intentionally_different():
    """Guard against accidental namespace collapse or function shadowing."""
    prefilter_sig = inspect.signature(build_prefilter_tool_kwargs)
    semantic_sig = inspect.signature(build_semantic_tool_kwargs)

    assert "state" in prefilter_sig.parameters
    assert "all_tools" in prefilter_sig.parameters

    assert "tool" in semantic_sig.parameters
    assert "user_msg" in semantic_sig.parameters
    assert "wants_save_to_rag" in semantic_sig.parameters
