"""Tests for forced tool-use in structured mode (Part E — web search refusal fix).

Covers:
- force_tool_use intent flag from detect_tool_routing_intents()
- Mandatory prompt footer when force_tool_use=True
- Retry-on-declination when model returns text-only response
- Normal opt-out path when force_tool_use=False
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call

from universal_agentic_framework.orchestration.helpers.intent_detection import (
    detect_tool_routing_intents as _detect_intents,
)
from universal_agentic_framework.orchestration.graph_builder import node_call_tools_structured


# ---------------------------------------------------------------------------
# Helpers shared with test_semantic_tool_routing.py
# ---------------------------------------------------------------------------

def _set_mock_config(mock_config):
    provider = SimpleNamespace(
        type="ollama",
        api_base="http://localhost:11434/v1",
        tool_calling="structured",
        get_tool_calling_mode=lambda _m: "structured",
    )
    config = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en"),
        tool_routing=SimpleNamespace(
            similarity_threshold=0.3,
            top_k=5,
            min_top_score=0.7,
            min_spread=0.10,
            max_retries=2,
        ),
        tokens=SimpleNamespace(default_budget=10000, per_node_budgets={}),
        llm=SimpleNamespace(
            get_role_provider=lambda _role: provider,
            get_role_model_name=lambda _role, _lang: "ollama/llama",
            get_role_provider_chain_with_models=lambda role, _lang: [
                ("ollama", provider, "ollama/llama")
            ] if role in {"chat", "auxiliary"} else [],
        ),
    )
    mock_config.return_value = config
    return config


class FakeTool:
    def __init__(self, name="web_search_mcp"):
        self.name = name
        self.description = f"Tool: {name}"
        self.args_schema = None
        self.calls = []

    def _run(self, **kwargs):
        self.calls.append(kwargs)
        return f"{self.name} result"


class _TextOnlyModel:
    """Always returns plain text (no JSON tool call)."""
    def __init__(self, text="I cannot search the web."):
        self.text = text
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(content=self.text)


class _SequencedModel:
    """Returns predefined responses in order."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        resp = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(content=resp)


def _structured_state(user_msg, tool_name="web_search_mcp", score=0.93, intents=None):
    tool = FakeTool(tool_name)
    return {
        "messages": [{"role": "user", "content": user_msg}],
        "candidate_tools": [{"tool": tool, "name": tool_name, "score": score}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "tool_calling_mode": "structured",
        "tool_calling_mode_reason": "probe_no_tool_calling",
        "language": "en",
        "prefilter_intents": intents or {},
        "user_settings": {},
    }, tool


# ---------------------------------------------------------------------------
# Part E3 — force_tool_use in intent detection
# ---------------------------------------------------------------------------

class TestForceToolUseIntentFlag:
    """force_tool_use must be True for explicit web search, False otherwise."""

    @pytest.mark.parametrize("msg", [
        "search the web for metformin side effects",
        "look up the latest news about AI",
        "google current exchange rate EUR USD",
        "search for rosuvastatin drug interactions",
    ])
    def test_web_search_sets_force_tool_use(self, msg):
        intents = _detect_intents(msg, language="en")
        assert intents["mentions_web_search"] is True
        assert intents["force_tool_use"] is True, f"Expected force_tool_use=True for: {msg!r}"

    @pytest.mark.parametrize("msg", [
        "What is type 2 diabetes?",
        "Hello there",
        "What is 5 * 12?",
        "What tools do you have?",
    ])
    def test_non_web_query_does_not_set_force_tool_use(self, msg):
        intents = _detect_intents(msg, language="en")
        assert intents["force_tool_use"] is False, f"Expected force_tool_use=False for: {msg!r}"

    def test_force_tool_use_key_always_present(self):
        intents = _detect_intents("anything", language="en")
        assert "force_tool_use" in intents


# ---------------------------------------------------------------------------
# Part E1 — mandatory prompt when force_tool_use=True
# ---------------------------------------------------------------------------

class TestMandatoryPromptContent:
    """System prompt must include mandatory instruction when force_tool_use=True."""

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_mandatory_footer_in_system_prompt_when_force(self, mock_config, mock_model):
        _set_mock_config(mock_config)

        captured_messages = []

        class CapturingModel:
            def invoke(self, messages):
                captured_messages.extend(messages)
                return SimpleNamespace(content='{"tool": "web_search_mcp", "args": {"query": "test"}}')

        mock_model.return_value = CapturingModel()
        state, _ = _structured_state(
            "search the web for test",
            score=0.93,
            intents={"force_tool_use": True, "mentions_web_search": True},
        )

        node_call_tools_structured(state)

        system_msg = next((m for m in captured_messages if hasattr(m, "content") and "MUST" in m.content), None)
        assert system_msg is not None, "Expected mandatory 'MUST' instruction in system prompt"
        assert "MUST call" in system_msg.content
        # Opt-out must be absent
        assert "If no tool is needed, respond normally" not in system_msg.content

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_opt_out_footer_when_not_forced(self, mock_config, mock_model):
        _set_mock_config(mock_config)

        captured_messages = []

        class CapturingModel:
            def invoke(self, messages):
                captured_messages.extend(messages)
                return SimpleNamespace(content="I'll answer directly.")

        mock_model.return_value = CapturingModel()
        state, _ = _structured_state(
            "What is diabetes?",
            score=0.50,  # below 0.75 threshold
            intents={"force_tool_use": False},
        )

        node_call_tools_structured(state)

        system_msg = next((m for m in captured_messages if hasattr(m, "content") and "tool" in m.content.lower()), None)
        assert system_msg is not None
        assert "If no tool is needed, respond normally in plain text" in system_msg.content
        assert "MUST" not in system_msg.content

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_high_score_alone_triggers_mandatory_prompt(self, mock_config, mock_model):
        """A candidate with score ≥ 0.75 should trigger force_tool_use even without web intent."""
        _set_mock_config(mock_config)

        captured_messages = []

        class CapturingModel:
            def invoke(self, messages):
                captured_messages.extend(messages)
                return SimpleNamespace(content='{"tool": "web_search_mcp", "args": {"query": "x"}}')

        mock_model.return_value = CapturingModel()
        # Score ≥ 0.75, no explicit web intent
        state, _ = _structured_state(
            "Tell me about metformin",
            score=0.80,
            intents={"force_tool_use": False},
        )

        node_call_tools_structured(state)

        system_msg = next((m for m in captured_messages if hasattr(m, "content") and "MUST" in m.content), None)
        assert system_msg is not None, "Expected mandatory prompt for high-score candidate"


# ---------------------------------------------------------------------------
# Part E2 — retry on silent declination
# ---------------------------------------------------------------------------

class TestRetryOnDeclination:
    """Model must be retried when it declines despite force_tool_use=True."""

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_retry_fires_after_plain_text_response(self, mock_config, mock_model):
        """First attempt: text-only. Second attempt: valid JSON tool call. Tool must execute."""
        _set_mock_config(mock_config)

        model = _SequencedModel(
            "I cannot search the web.",
            '{"tool": "web_search_mcp", "args": {"query": "metformin"}}',
        )
        mock_model.return_value = model

        state, tool = _structured_state(
            "search the web for metformin",
            score=0.93,
            intents={"force_tool_use": True, "mentions_web_search": True},
        )

        result = node_call_tools_structured(state)

        # Two invocations: initial + retry
        assert len(model.invocations) == 2, f"Expected 2 model calls, got {len(model.invocations)}"
        # Tool must have executed on the second attempt
        assert "web_search_mcp" in result["tool_results"]
        assert tool.calls  # _run was called

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_retry_message_contains_mandatory_instruction(self, mock_config, mock_model):
        """The retry message appended to the conversation must re-state the obligation."""
        _set_mock_config(mock_config)

        all_messages = []

        class TrackingModel:
            call_count = 0

            def invoke(self, messages):
                all_messages.append(list(messages))
                TrackingModel.call_count += 1
                if TrackingModel.call_count == 1:
                    return SimpleNamespace(content="I'll just answer directly.")
                return SimpleNamespace(
                    content='{"tool": "web_search_mcp", "args": {"query": "test"}}'
                )

        mock_model.return_value = TrackingModel()
        state, _ = _structured_state(
            "search the web for test",
            score=0.93,
            intents={"force_tool_use": True},
        )

        node_call_tools_structured(state)

        # Second invocation messages must include the retry instruction
        assert len(all_messages) >= 2, "Expected at least 2 model invocations"
        second_call_msgs = all_messages[1]
        retry_content = " ".join(
            m.content for m in second_call_msgs if hasattr(m, "content")
        )
        assert "MUST call" in retry_content or "You did not call" in retry_content

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_no_retry_when_not_forced(self, mock_config, mock_model):
        """When force_tool_use=False, declining on first attempt must not trigger a retry."""
        _set_mock_config(mock_config)

        model = _TextOnlyModel("No tool needed, here's the answer.")
        mock_model.return_value = model

        state, tool = _structured_state(
            "What is diabetes?",
            score=0.50,  # below 0.75
            intents={"force_tool_use": False},
        )

        node_call_tools_structured(state)

        assert len(model.invocations) == 1, "Should not retry when not forced"
        assert not tool.calls, "Tool must not have been called"

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_graceful_exit_when_all_retries_decline(self, mock_config, mock_model):
        """If model keeps declining through all retries, node exits without crash."""
        _set_mock_config(mock_config)

        model = _TextOnlyModel("I cannot search the web.")
        mock_model.return_value = model

        state, tool = _structured_state(
            "search the web for test",
            score=0.93,
            intents={"force_tool_use": True},
        )

        # Should complete without raising
        result = node_call_tools_structured(state)

        # max_retries=2 → up to 3 total invocations (attempt 0, 1, 2)
        assert len(model.invocations) <= 3
        assert not tool.calls, "Tool must not have been called"
        assert isinstance(result, dict)  # state returned normally

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_no_retry_when_model_calls_tool_first_attempt(self, mock_config, mock_model):
        """When the model calls the tool on the first attempt, no retry should occur."""
        _set_mock_config(mock_config)

        model = _SequencedModel('{"tool": "web_search_mcp", "args": {"query": "test"}}')
        mock_model.return_value = model

        state, tool = _structured_state(
            "search the web for test",
            score=0.93,
            intents={"force_tool_use": True},
        )

        result = node_call_tools_structured(state)

        assert len(model.invocations) == 1, "No retry needed when first attempt succeeds"
        assert "web_search_mcp" in result["tool_results"]
