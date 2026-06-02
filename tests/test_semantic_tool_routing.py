"""Test semantic tool routing (model-agnostic tool execution)."""

import pytest
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from universal_agentic_framework.orchestration.graph_builder import (
    node_prefilter_tools,
    node_call_tools_native,
    node_call_tools_structured,
    node_generate_response,
)
from universal_agentic_framework.orchestration.helpers.embedding_provider import (
    clear_embedding_cache as _clear_embedding_cache,
)
from universal_agentic_framework.orchestration.helpers.intent_detection import (
    detect_tool_routing_intents as _detect_tool_routing_intents,
)
from universal_agentic_framework.orchestration.helpers.tool_preparation import (
    apply_top_k_scored_tools as _apply_top_k_scored_tools,
)
from universal_agentic_framework.orchestration.helpers.tool_scoring import _tool_embedding_cache
from universal_agentic_framework.tools.datetime.tool import DateTimeTool


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    """Clear embedding model cache before each test to ensure mocks work."""
    _tool_embedding_cache.clear()
    _clear_embedding_cache()
    yield
    _tool_embedding_cache.clear()
    _clear_embedding_cache()


class FakeTool:
    """Fake tool for testing."""
    
    def __init__(self, name: str, description: str, return_value: str = "tool output"):
        self.name = name
        self.description = description
        self.return_value = return_value
    
    def _run(self, **kwargs) -> str:
        """Execute fake tool."""
        return self.return_value


class DummyModel:
    """Capture invocations for response node tests."""

    def __init__(self, output: str = "model output"):
        self.output = output
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(content=self.output)


class SequencedDummyModel:
    """Return predefined outputs in order for successive invoke calls."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.outputs:
            return SimpleNamespace(content=self.outputs.pop(0))
        return SimpleNamespace(content="")


class ListContentStructuredModel:
    """Return structured JSON tool call embedded in list content blocks."""

    def __init__(self):
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(
            content=[
                {"type": "text", "text": '{"tool": "datetime_tool", "args": {}}'},
            ]
        )


class DictContentStructuredModel:
    """Return structured JSON tool call in dict-shaped content."""

    def __init__(self):
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(
            content={"type": "text", "text": '{"tool": "datetime_tool", "args": {}}'}
        )


class MixedContentStructuredModel:
    """Return mixed blocks where only one block contains valid JSON call text."""

    def __init__(self):
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(
            content=[
                {"type": "reasoning", "content": [{"summary": "planning"}]},
                {"type": "text", "text": '{"tool": "datetime_tool", "args": {}}'},
            ]
        )


class DummyNativeModel:
    """Minimal native tool-calling model stub."""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        return SimpleNamespace(content="", tool_calls=self.tool_calls)


def set_mock_config(
    mock_config,
    *,
    timezone: str = "UTC",
    similarity_threshold: float = 0.3,
    top_k: int | None = 10,
    embedding_model: str | None = None,
    language: str = "en",
    profile_name: str = "test-fork",
):
    """Helper to configure core settings for tool routing tests."""

    provider = SimpleNamespace(
        type="ollama",
        api_base="http://localhost:11434/v1",
        models=SimpleNamespace(
            en="ollama/llama",
            model_dump=lambda: {language: "ollama/llama"},
        ),
        tool_calling="structured",
        get_tool_calling_mode=lambda _model_name: provider.tool_calling,
    )

    config = SimpleNamespace(
        profile=SimpleNamespace(name=profile_name, timezone=timezone, language=language),
        memory=SimpleNamespace(
            embeddings=SimpleNamespace(
                dimension=768,
            )
        ),
        tool_routing=SimpleNamespace(
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            top_k=top_k,
            min_top_score=0.7,
            min_spread=0.10,
        ),
        tokens=SimpleNamespace(default_budget=1000, per_node_budgets={"response_node": 1000}),
        llm=SimpleNamespace(
            roles=SimpleNamespace(
                chat=SimpleNamespace(provider_id="ollama", model="ollama/llama", api_base="http://localhost:11434/v1")
            ),
            get_role_provider=lambda _role: provider,
            get_role_model_name=lambda _role, _lang: "ollama/llama",
            get_role_provider_chain_with_models=lambda role_name, _lang: [
                ("ollama", provider, "ollama/llama")
            ] if role_name in {"chat", "vision", "auxiliary"} else [],
            get_embedding_provider_type=lambda: "remote",
            get_embedding_remote_endpoint=lambda: "http://localhost:11434/v1",
        ),
    )

    mock_config.return_value = config
    return config


class TestToolResultInjection:
    """Tests for tool result and knowledge injection into the response node."""

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_injects_tool_results_and_knowledge(self, mock_config, mock_model_factory):
        """Response node should inject both tool results and knowledge context."""
        config = SimpleNamespace(
            profile=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
            tokens=SimpleNamespace(default_budget=1000, per_node_budgets={"response_node": 1000}),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        )
        mock_config.return_value = config

        fake_model = DummyModel()
        mock_model_factory.return_value = fake_model

        state = {
            "messages": [{"role": "user", "content": "frage"}],
            "tool_results": {"datetime_tool": "UTC 2024-01-22 12:00"},
            "knowledge_context": [{"text": "Doc text", "file_name": "doc1.md", "score": 0.9}],
        }

        result = node_generate_response(state)

        system_prompt = fake_model.invocations[0][0].content
        assert "=== TOOL RESULTS ===" in system_prompt
        assert "datetime_tool" in system_prompt
        assert "=== WISSENSDATENBANK ===" in system_prompt
        assert "Doc text" in system_prompt
        assert result["messages"][-1]["content"] == "model output"

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_skips_tool_section_when_empty(self, mock_config, mock_model_factory):
        """Response node should not add tool section if no tool results are present."""
        config = SimpleNamespace(
            profile=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
            tokens=SimpleNamespace(default_budget=1000, per_node_budgets={"response_node": 1000}),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        )
        mock_config.return_value = config

        fake_model = DummyModel()
        mock_model_factory.return_value = fake_model

        state = {
            "messages": [{"role": "user", "content": "frage"}],
            "tool_results": {},
            "knowledge_context": [{"text": "Doc text", "file_name": "doc1.md", "score": 0.8}],
        }

        node_generate_response(state)

        system_prompt = fake_model.invocations[0][0].content
        assert "=== TOOL RESULTS ===" not in system_prompt
        assert "=== WISSENSDATENBANK ===" in system_prompt
        assert "Doc text" in system_prompt

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_keeps_low_priority_memory_when_web_tool_results_present(self, mock_config, mock_model_factory):
        """Past memory should be retained as background while tool results remain primary."""
        config = SimpleNamespace(
            profile=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
            tokens=SimpleNamespace(default_budget=1000, per_node_budgets={"response_node": 1000}),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        )
        mock_config.return_value = config

        fake_model = DummyModel()
        mock_model_factory.return_value = fake_model

        state = {
            "messages": [{"role": "user", "content": "summarize https://www.mediaworkbench.com"}],
            "tool_results": {"extract_webpage_mcp": "Fresh extracted content"},
            "loaded_memory": [{"text": "Old unrelated memory about The Shamen"}],
            "knowledge_context": [],
        }

        node_generate_response(state)

        system_prompt = fake_model.invocations[0][0].content
        assert "=== TOOL RESULTS ===" in system_prompt
        assert "Fresh extracted content" in system_prompt
        assert "=== PAST CONTEXT (LOW PRIORITY) ===" in system_prompt
        assert "Old unrelated memory about The Shamen" in system_prompt
        assert "=== CONTEXT PRIORITY ===" in system_prompt

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_retries_when_extract_succeeded_but_model_claims_access_error(self, mock_config, mock_model_factory):
        """Response node should correct contradictory access-error claims after successful extraction."""
        config = SimpleNamespace(
            profile=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
            tokens=SimpleNamespace(default_budget=1000, per_node_budgets={"response_node": 1000}),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        )
        mock_config.return_value = config

        fake_model = SequencedDummyModel(
            outputs=[
                "The headline is not directly retrievable due to a connection error.",
                "The headline is: Die Wahrheit ist eine Waffe.",
            ]
        )
        mock_model_factory.return_value = fake_model

        state = {
            "messages": [{"role": "user", "content": "What is the headline on https://www.tagesschau.de?"}],
            "tool_results": {
                "extract_webpage_mcp": "tagesschau.de ... Die Wahrheit ist eine Waffe ...",
            },
            "knowledge_context": [],
            "routing_metadata": {"extract_webpage_mcp": "url detected"},
            "tool_execution_results": {
                "extract_webpage_mcp": {"status": "success", "summary": None}
            },
        }

        result = node_generate_response(state)

        assert result["messages"][-1]["content"] == "The headline is: Die Wahrheit ist eine Waffe."
        assert len(fake_model.invocations) >= 2

    @patch("universal_agentic_framework.orchestration.graph_builder.get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_retries_for_german_access_refusal_after_successful_extract(self, mock_config, mock_model_factory):
        """German refusal wording should trigger correction retry when extraction succeeded."""
        config = SimpleNamespace(
            profile=SimpleNamespace(language="de", name="test-fork", timezone="UTC"),
            tokens=SimpleNamespace(default_budget=1000, per_node_budgets={"response_node": 1000}),
            llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"de": "llama"}))),
        )
        mock_config.return_value = config

        fake_model = SequencedDummyModel(
            outputs=[
                "Die Schlagzeile ist aktuell wegen eines Verbindungsproblems nicht abrufbar.",
                "Die Schlagzeile lautet: Die Wahrheit ist eine Waffe.",
            ]
        )
        mock_model_factory.return_value = fake_model

        state = {
            "messages": [{"role": "user", "content": "Was ist die Schlagzeile auf https://www.tagesschau.de?"}],
            "tool_results": {
                "extract_webpage_mcp": "tagesschau.de ... Die Wahrheit ist eine Waffe ...",
            },
            "knowledge_context": [],
            "routing_metadata": {"extract_webpage_mcp": "url detected"},
            "tool_execution_results": {
                "extract_webpage_mcp": {"status": "success", "summary": None}
            },
        }

        result = node_generate_response(state)

        assert result["messages"][-1]["content"] == "Die Schlagzeile lautet: Die Wahrheit ist eine Waffe."
        assert len(fake_model.invocations) >= 2


class TestRoutingIntentDetection:
    """Focused tests for intent helper used by semantic routing."""

    def test_detects_datetime_and_save_intents(self):
        intents = _detect_tool_routing_intents(
            user_msg="How old am I and save this result please",
            language="en",
        )

        assert intents["mentions_datetime"] is True
        assert intents["wants_save_to_rag"] is True
        assert intents["search_language"] == "en"
        assert intents["search_region"] == "us-en"

    def test_detects_url_and_enhances_sentiment_query(self):
        intents = _detect_tool_routing_intents(
            user_msg="Check sentiment for BTC at www.example.com/news",
            language="de",
        )

        assert intents["url_in_query"] == "https://www.example.com/news"
        assert intents["enhanced_web_query"] == "BTC coin crypto sentiment market outlook social media news"
        assert intents["search_language"] == "de"
        assert intents["search_region"] == "de-de"

    def test_country_specific_region_override_for_web_search(self):
        intents = _detect_tool_routing_intents(
            user_msg="can you search the web for latest news in Spain?",
            language="en",
        )

        assert intents["search_language"] == "en"
        assert intents["search_region"] == "es-es"

    def test_country_region_inference_for_non_hardcoded_country(self):
        intents = _detect_tool_routing_intents(
            user_msg="Find the latest headlines in Japan",
            language="en",
        )

        assert intents["search_language"] == "en"
        assert intents["search_region"] == "jp-jp"

    def test_country_alias_region_inference_for_uk(self):
        intents = _detect_tool_routing_intents(
            user_msg="what happened in the UK today?",
            language="en",
        )

        assert intents["search_language"] == "en"
        assert intents["search_region"] == "uk-en"

    def test_cleans_conversational_web_query_and_infers_result_count(self):
        intents = _detect_tool_routing_intents(
            user_msg="can you search the web for the 3 latest recent news in spain?",
            language="en",
        )

        assert intents["enhanced_web_query"] == "latest recent news in spain"
        assert intents["requested_web_results"] == 3

    def test_detects_explicit_web_search_intent(self):
        intents = _detect_tool_routing_intents(
            user_msg="search the web for rosuvastatin",
            language="en",
        )

        assert intents["mentions_web_search"] is True
        assert intents["enhanced_web_query"] == "rosuvastatin"


@patch("universal_agentic_framework.orchestration.graph_builder.get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_native_extract_injects_request_url_when_missing(mock_config, mock_model_factory):
    """Native mode should inject request_url from user message when model omits URL args."""
    set_mock_config(mock_config)

    captured_kwargs = {}

    class CaptureExtractTool:
        name = "extract_webpage_mcp"
        description = "Extract webpage content"
        args_schema = None

        def _run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "extracted content"

    fake_tool = CaptureExtractTool()
    mock_model_factory.return_value = DummyNativeModel(
        tool_calls=[
            {
                "name": "extract_webpage_mcp",
                "args": {"query": "headline", "args": [], "kwargs": {"selector": "#teaser"}},
            }
        ]
    )

    state = {
        "messages": [{"role": "user", "content": "extract from https://www.tagesschau.de/ the headline"}],
        "candidate_tools": [{"tool": fake_tool}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "language": "en",
        "prefilter_intents": {"url_in_query": "https://www.tagesschau.de/", "wants_save_to_rag": False},
    }

    result = node_call_tools_native(state)

    assert captured_kwargs["request_url"] == "https://www.tagesschau.de/"
    assert result["tool_results"]["extract_webpage_mcp"] == "extracted content"


@patch("universal_agentic_framework.orchestration.graph_builder.score_tool_similarity")
@patch("universal_agentic_framework.orchestration.graph_builder._get_routing_embedding_provider")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_prefilter_keeps_web_search_candidate_for_explicit_web_intent(
    mock_config,
    mock_embedding_provider,
    mock_score_similarity,
):
    """Explicit web-search requests should keep web_search_mcp through strict prefilter gates."""
    set_mock_config(mock_config, similarity_threshold=0.55, top_k=5)

    fake_provider = Mock()
    fake_provider.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_embedding_provider.return_value = (fake_provider, "fake-embedding")

    # Keep all similarities intentionally low to verify intent override behavior.
    mock_score_similarity.return_value = 0.05

    loaded_tools = [
        FakeTool("datetime_tool", "Get date and time"),
        FakeTool("calculator_tool", "Perform calculations"),
        FakeTool("file_ops_tool", "Read and write files"),
        FakeTool("web_search_mcp", "Search the web for current information"),
        FakeTool("extract_webpage_mcp", "Extract webpage content"),
    ]

    state = {
        "messages": [{"role": "user", "content": "please search the web for current EV battery breakthroughs"}],
        "loaded_tools": loaded_tools,
        "language": "en",
    }

    result = node_prefilter_tools(state)
    candidate_names = [c["name"] for c in result.get("candidate_tools", [])]
    assert "web_search_mcp" in candidate_names


@patch("universal_agentic_framework.orchestration.graph_builder.score_tool_similarity")
@patch("universal_agentic_framework.orchestration.graph_builder._get_routing_embedding_provider")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_prefilter_downgrades_native_mode_when_probe_signals_mismatch(
    mock_config,
    mock_embedding_provider,
    mock_score_similarity,
):
    set_mock_config(mock_config, similarity_threshold=0.10, top_k=5)
    mock_config.return_value.llm.get_role_provider("chat").tool_calling = "native"

    fake_provider = Mock()
    fake_provider.encode.return_value = np.array([0.2, 0.3, 0.4])
    mock_embedding_provider.return_value = (fake_provider, "fake-embedding")
    mock_score_similarity.return_value = 0.95

    state = {
        "messages": [{"role": "user", "content": "what time is it"}],
        "loaded_tools": [FakeTool("datetime_tool", "Get date and time")],
        "language": "en",
        "llm_capability_probes": [
            {
                "provider_id": "ollama",
                "model_name": "ollama/llama",
                "supports_bind_tools": False,
                "supports_tool_schema": False,
                "capability_mismatch": True,
                "status": "warning",
                "probed_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }

    result = node_prefilter_tools(state)

    assert result["tool_calling_mode"] == "structured"
    assert result["tool_calling_mode_reason"] == "probe_capability_mismatch_downgrade"


@patch("universal_agentic_framework.orchestration.graph_builder.score_tool_similarity")
@patch("universal_agentic_framework.orchestration.graph_builder._get_routing_embedding_provider")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_prefilter_keeps_native_mode_when_probe_is_ok(
    mock_config,
    mock_embedding_provider,
    mock_score_similarity,
):
    set_mock_config(mock_config, similarity_threshold=0.10, top_k=5)
    mock_config.return_value.llm.get_role_provider("chat").tool_calling = "native"

    fake_provider = Mock()
    fake_provider.encode.return_value = np.array([0.2, 0.3, 0.4])
    mock_embedding_provider.return_value = (fake_provider, "fake-embedding")
    mock_score_similarity.return_value = 0.95

    state = {
        "messages": [{"role": "user", "content": "what time is it"}],
        "loaded_tools": [FakeTool("datetime_tool", "Get date and time")],
        "language": "en",
        "llm_capability_probes": [
            {
                "provider_id": "ollama",
                "model_name": "ollama/llama",
                "supports_bind_tools": True,
                "supports_tool_schema": True,
                "capability_mismatch": False,
                "status": "ok",
                "probed_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }

    result = node_prefilter_tools(state)

    assert result["tool_calling_mode"] == "native"
    assert result["tool_calling_mode_reason"] == "probe_confirmed_native"


@patch("universal_agentic_framework.orchestration.graph_builder.get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_native_extract_fallback_runs_when_model_skips_tool_calls(mock_config, mock_model_factory):
    """Native mode should still execute extract tool for URL prompts when model emits no tool calls."""
    set_mock_config(mock_config)

    captured_kwargs = {}

    class CaptureExtractTool:
        name = "extract_webpage_mcp"
        description = "Extract webpage content"
        args_schema = None

        def _run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "extracted via fallback"

    fake_tool = CaptureExtractTool()
    mock_model_factory.return_value = DummyNativeModel(tool_calls=[])

    state = {
        "messages": [{"role": "user", "content": "can you summarize https://www.mediaworkbench.com ?"}],
        "candidate_tools": [{"tool": fake_tool}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "language": "en",
        "prefilter_intents": {"url_in_query": "https://www.mediaworkbench.com", "wants_save_to_rag": False},
    }

    result = node_call_tools_native(state)

    assert captured_kwargs["request_url"] == "https://www.mediaworkbench.com"
    assert result["tool_results"]["extract_webpage_mcp"] == "extracted via fallback"


@patch("universal_agentic_framework.orchestration.graph_builder.get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_native_extract_retries_with_inferred_url_after_protocol_error(mock_config, mock_model_factory):
    """Native mode should retry extract with inferred URL when first call returns protocol-missing error."""
    set_mock_config(mock_config)

    seen_calls = []

    class RetryExtractTool:
        name = "extract_webpage_mcp"
        description = "Extract webpage content"
        args_schema = None

        def _run(self, **kwargs):
            seen_calls.append(kwargs)
            if len(seen_calls) == 1:
                return "Error: Could not access the webpage (Request URL is missing an 'http://' or 'https://' protocol.)"
            return "retry success content"

    fake_tool = RetryExtractTool()
    mock_model_factory.return_value = DummyNativeModel(
        tool_calls=[
            {
                "name": "extract_webpage_mcp",
                "args": {"query": "summary", "args": [], "kwargs": {}},
            }
        ]
    )

    state = {
        "messages": [{"role": "user", "content": "summarize https://www.mediaworkbench.com"}],
        "candidate_tools": [{"tool": fake_tool}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "language": "en",
        "prefilter_intents": {"url_in_query": "https://www.mediaworkbench.com", "wants_save_to_rag": False},
    }

    result = node_call_tools_native(state)

    assert len(seen_calls) == 2
    assert seen_calls[1]["request_url"] == "https://www.mediaworkbench.com"
    assert result["tool_results"]["extract_webpage_mcp"] == "retry success content"


@patch("universal_agentic_framework.orchestration.graph_builder.get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_structured_tool_call_parses_list_content_blocks(mock_config, mock_model_factory):
    """Structured mode should parse JSON tool call from list-based content blocks."""
    set_mock_config(mock_config)
    mock_model_factory.return_value = ListContentStructuredModel()

    dt_tool = DateTimeTool()
    state = {
        "messages": [{"role": "user", "content": "what time is it"}],
        "candidate_tools": [{"tool": dt_tool, "name": "datetime_tool", "score": 0.9}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "tool_calling_mode": "structured",
        "tool_calling_mode_reason": "model_config_non_native_mode",
        "language": "en",
    }

    result = node_call_tools_structured(state)

    assert "datetime_tool" in result["tool_results"]


@patch("universal_agentic_framework.orchestration.graph_builder.get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_structured_tool_call_parses_dict_content_blocks(mock_config, mock_model_factory):
    """Structured mode should parse JSON tool call from dict-shaped content."""
    set_mock_config(mock_config)
    mock_model_factory.return_value = DictContentStructuredModel()

    dt_tool = DateTimeTool()
    state = {
        "messages": [{"role": "user", "content": "what time is it"}],
        "candidate_tools": [{"tool": dt_tool, "name": "datetime_tool", "score": 0.9}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "tool_calling_mode": "structured",
        "tool_calling_mode_reason": "model_config_non_native_mode",
        "language": "en",
    }

    result = node_call_tools_structured(state)

    assert "datetime_tool" in result["tool_results"]


@patch("universal_agentic_framework.orchestration.graph_builder.get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
def test_structured_tool_call_parses_mixed_content_blocks(mock_config, mock_model_factory):
    """Structured mode should tolerate mixed content blocks and still parse JSON call."""
    set_mock_config(mock_config)
    mock_model_factory.return_value = MixedContentStructuredModel()

    dt_tool = DateTimeTool()
    state = {
        "messages": [{"role": "user", "content": "what time is it"}],
        "candidate_tools": [{"tool": dt_tool, "name": "datetime_tool", "score": 0.9}],
        "tool_results": {},
        "tool_execution_results": {},
        "routing_metadata": {},
        "tool_calling_mode": "structured",
        "tool_calling_mode_reason": "model_config_non_native_mode",
        "language": "en",
    }

    result = node_call_tools_structured(state)

    assert "datetime_tool" in result["tool_results"]


class TestTopKScoredToolsHelper:
    """Focused tests for top-k scored-tools selection helper."""

    def test_applies_top_k_in_descending_order(self):
        tools = [("a", 0.2), ("b", 0.9), ("c", 0.6)]
        selected = _apply_top_k_scored_tools(tools, 2)
        assert selected == [("b", 0.9), ("c", 0.6)]

    def test_returns_original_when_top_k_not_set_or_invalid(self):
        tools = [("a", 0.2), ("b", 0.9)]
        assert _apply_top_k_scored_tools(tools, None) == tools
        assert _apply_top_k_scored_tools(tools, 0) == tools
        assert _apply_top_k_scored_tools(tools, -1) == tools
