"""Test semantic tool routing (model-agnostic tool execution)."""

import pytest
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from universal_agentic_framework.orchestration.graph_builder import (
    node_prefilter_tools,
    node_route_tools,
    node_call_tools_native,
    node_generate_response,
)
from universal_agentic_framework.orchestration.helpers.embedding_provider import (
    clear_embedding_cache as _clear_embedding_cache,
)
from universal_agentic_framework.orchestration.helpers.intent_detection import (
    detect_tool_routing_intents as _detect_tool_routing_intents,
)
from universal_agentic_framework.orchestration.helpers.semantic_execution import (
    build_semantic_tool_kwargs as _build_semantic_tool_kwargs,
    extract_calculator_expression as _extract_calculator_expression,
    run_forced_tool as _run_forced_tool,
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
    fork_name: str = "test-fork",
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
        fork=SimpleNamespace(name=fork_name, timezone=timezone, language=language),
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


class TestSemanticToolRouting:
    """Tests for semantic tool routing logic."""
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_empty_query_no_tools_executed(self, mock_config):
        """Test tool routing with empty query (no tools executed)."""
        set_mock_config(mock_config)
        
        state = {
            "messages": [],
            "loaded_tools": [FakeTool("tool1", "Tool for X"), FakeTool("tool2", "Tool for Y")]
        }
        
        result = node_route_tools(state)
        assert result["tool_results"] == {}
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_similarity_scoring_selects_relevant_tools(self, mock_provider_factory, mock_config):
        """Test tool scoring by cosine similarity."""
        set_mock_config(mock_config)
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        
        # High similarity to tool1, low to tool2
        query_emb = np.array([1.0, 0.0, 0.0])
        tool1_emb = np.array([0.95, 0.1, 0.0])
        tool2_emb = np.array([0.1, 0.9, 0.0])
        
        mock_embedder.encode.side_effect = [query_emb, tool1_emb, tool2_emb]
        
        tool1 = FakeTool("web_search", "Search the internet")
        tool2 = FakeTool("db_query", "Query database")
        
        state = {
            "messages": [{"role": "user", "content": "search the web"}],
            "loaded_tools": [tool1, tool2]
        }
        
        result = node_route_tools(state)
        
        # High similarity tool should be executed, low similarity should not
        assert "web_search" in result["tool_results"]
        assert "db_query" not in result["tool_results"]
        assert "tool_execution_results" in result
        assert result["tool_execution_results"]["web_search"]["status"] == "success"
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_threshold_prevents_low_similarity_tools(self, mock_provider_factory, mock_config):
        """Test that tools below similarity threshold are not executed."""
        set_mock_config(mock_config)
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        
        # Very low similarity (below 0.3 threshold)
        query_emb = np.array([1.0, 0.0])
        tool_emb = np.array([0.05, 0.999])
        
        mock_embedder.encode.side_effect = [query_emb, tool_emb]
        
        tool = FakeTool("unrelated", "Unrelated tool")
        state = {"messages": [{"role": "user", "content": "help"}], "loaded_tools": [tool]}
        
        result = node_route_tools(state)
        assert "unrelated" not in result["tool_results"]

    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_high_threshold_blocks_tools(self, mock_provider_factory, mock_config):
        """High similarity threshold should block moderately similar tools."""
        set_mock_config(mock_config, similarity_threshold=0.95)

        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder

        query_emb = np.array([1.0, 0.0])
        tool_emb = np.array([0.7, 0.7])  # cosine ~0.7 < 0.95 threshold
        mock_embedder.encode.side_effect = [query_emb, tool_emb]

        tool = FakeTool("maybe", "Maybe relevant")
        state = {"messages": [{"role": "user", "content": "something"}], "loaded_tools": [tool]}

        result = node_route_tools(state)
        assert result["tool_results"] == {}
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_forced_datetime_on_date_patterns(self, mock_provider_factory, mock_config):
        """Test forced execution of datetime_tool when date patterns detected."""
        set_mock_config(mock_config)
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        
        dt_tool = DateTimeTool()
        state = {
            "messages": [{"role": "user", "content": "ich bin am 24.05.1974 geboren"}],
            "loaded_tools": [dt_tool]
        }
        
        result = node_route_tools(state)
        
        # datetime_tool should be force-executed due to date pattern
        assert "datetime_tool" in result["tool_results"]
        assert "UTC" in result["tool_results"]["datetime_tool"]
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_forced_datetime_on_keywords(self, mock_provider_factory, mock_config):
        """Test forced execution of datetime_tool when time keywords detected."""
        set_mock_config(mock_config, timezone="Europe/Berlin")
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        
        dt_tool = DateTimeTool()
        
        # Test each keyword
        for keyword in ["today", "today", "what time is it", "how old am i"]:
            mock_embedder.reset_mock()
            mock_embedder.encode.side_effect = [np.ones(384), np.ones(384)]
            
            state = {
                "messages": [{"role": "user", "content": keyword}],
                "loaded_tools": [dt_tool]
            }
            
            result = node_route_tools(state)
            
            # Datetime tool should be in results for time keywords
            if keyword in ["what time is it", "how old am i", "today"]:
                assert "datetime_tool" in result["tool_results"], f"Failed for: {keyword}"

    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_forced_web_search_on_explicit_query(self, mock_provider_factory, mock_config):
        """Explicit web-search wording should force-run web_search_mcp."""
        set_mock_config(mock_config)

        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        mock_embedder.encode.side_effect = [np.array([1.0, 0.0])]

        web_tool = FakeTool("web_search_mcp", "Search the web", return_value="search results")
        state = {
            "messages": [{"role": "user", "content": "search the web for rosuvastatin"}],
            "loaded_tools": [web_tool],
        }

        result = node_route_tools(state)

        assert "web_search_mcp" in result["tool_results"]
        assert result["tool_results"]["web_search_mcp"] == "search results"
        assert "routing_metadata" in result
        assert "explicit web-search request detected" in result["routing_metadata"]["web_search_mcp"]
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_multiple_tools_executed(self, mock_provider_factory, mock_config):
        """Test execution of multiple similar tools."""
        set_mock_config(mock_config)
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        
        # All embeddings similar
        query_emb = np.array([1.0, 0.0])
        tool1_emb = np.array([0.95, 0.1])
        tool2_emb = np.array([0.9, 0.05])
        
        mock_embedder.encode.side_effect = [query_emb, tool1_emb, tool2_emb]
        
        tool1 = FakeTool("search", "Search", return_value="results")
        tool2 = FakeTool("analyze", "Analyze", return_value="analysis")
        
        state = {
            "messages": [{"role": "user", "content": "search and analyze"}],
            "loaded_tools": [tool1, tool2]
        }
        
        result = node_route_tools(state)
        
        assert "search" in result["tool_results"]
        assert "analyze" in result["tool_results"]
        assert result["tool_results"]["search"] == "results"
        assert result["tool_results"]["analyze"] == "analysis"

    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_top_k_limits_executed_tools(self, mock_provider_factory, mock_config):
        """Configured top_k should limit the number of executed tools."""
        set_mock_config(mock_config, top_k=1)

        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder

        query_emb = np.array([1.0, 0.0])
        tool_high = np.array([0.99, 0.01])
        tool_mid = np.array([0.9, 0.1])
        tool_low = np.array([0.8, 0.2])

        mock_embedder.encode.side_effect = [query_emb, tool_high, tool_mid, tool_low]

        tools = [
            FakeTool("top", "Top match"),
            FakeTool("mid", "Mid match"),
            FakeTool("low", "Low match"),
        ]

        state = {"messages": [{"role": "user", "content": "do stuff"}], "loaded_tools": tools}

        result = node_route_tools(state)

        assert set(result["tool_results"].keys()) == {"top"}
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_tool_execution_error_handling(self, mock_provider_factory, mock_config):
        """Test graceful handling of tool execution errors."""
        set_mock_config(mock_config)
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        
        query_emb = np.array([1.0])
        tool_emb = np.array([0.95])
        mock_embedder.encode.side_effect = [query_emb, tool_emb]
        
        error_tool = Mock()
        error_tool.name = "error_tool"
        error_tool.description = "Tool that fails"
        error_tool._run.side_effect = ValueError("Tool failed")
        
        state = {
            "messages": [{"role": "user", "content": "do something"}],
            "loaded_tools": [error_tool]
        }
        
        result = node_route_tools(state)
        
        # Error should be captured in result
        assert "error_tool" in result["tool_results"]
        assert "Tool failed" in result["tool_results"]["error_tool"]
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_missing_loaded_tools_field(self, mock_config):
        """Test handling of missing loaded_tools field."""
        set_mock_config(mock_config)
        
        state = {"messages": [{"role": "user", "content": "do something"}]}
        
        result = node_route_tools(state)
        assert result["tool_results"] == {}
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_empty_tools_list(self, mock_config):
        """Test handling of empty tools list."""
        set_mock_config(mock_config)
        
        state = {
            "messages": [{"role": "user", "content": "do something"}],
            "loaded_tools": []
        }
        
        result = node_route_tools(state)
        assert result["tool_results"] == {}
    
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
    def test_state_preservation(self, mock_provider_factory, mock_config):
        """Test that tool routing preserves other state fields."""
        set_mock_config(mock_config)
        
        mock_embedder = MagicMock()
        mock_provider_factory.return_value = mock_embedder
        mock_embedder.encode.side_effect = [np.ones(384), np.ones(384)]
        
        tool = FakeTool("tool1", "Tool", return_value="output")
        
        state = {
            "messages": [{"role": "user", "content": "test"}],
            "loaded_tools": [tool],
            "user_id": "user123",
            "other_field": "preserved"
        }
        
        result = node_route_tools(state)
        
        # All original fields should be preserved
        assert result["user_id"] == "user123"
        assert result["other_field"] == "preserved"
        assert result["loaded_tools"] == [tool]
        assert "tool_results" in result
        assert "tool_execution_results" in result

    @patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_injects_tool_results_and_knowledge(self, mock_config, mock_model_factory):
        """Response node should inject both tool results and knowledge context."""
        config = SimpleNamespace(
            fork=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
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

    @patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_skips_tool_section_when_empty(self, mock_config, mock_model_factory):
        """Response node should not add tool section if no tool results are present."""
        config = SimpleNamespace(
            fork=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
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

    @patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_keeps_low_priority_memory_when_web_tool_results_present(self, mock_config, mock_model_factory):
        """Past memory should be retained as background while tool results remain primary."""
        config = SimpleNamespace(
            fork=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
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

    @patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_retries_when_extract_succeeded_but_model_claims_access_error(self, mock_config, mock_model_factory):
        """Response node should correct contradictory access-error claims after successful extraction."""
        config = SimpleNamespace(
            fork=SimpleNamespace(language="en", name="test-fork", timezone="UTC"),
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

    @patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_response_retries_for_german_access_refusal_after_successful_extract(self, mock_config, mock_model_factory):
        """German refusal wording should trigger correction retry when extraction succeeded."""
        config = SimpleNamespace(
            fork=SimpleNamespace(language="de", name="test-fork", timezone="UTC"),
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


@patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
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


@patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
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
    }

    result = node_call_tools_native(state)

    assert captured_kwargs["request_url"] == "https://www.mediaworkbench.com"
    assert result["tool_results"]["extract_webpage_mcp"] == "extracted via fallback"


@patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model")
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
    }

    result = node_call_tools_native(state)

    assert len(seen_calls) == 2
    assert seen_calls[1]["request_url"] == "https://www.mediaworkbench.com"
    assert result["tool_results"]["extract_webpage_mcp"] == "retry success content"


class TestSemanticKwargsBuilder:
    """Focused tests for semantic kwargs helper."""

    def test_web_search_kwargs_include_region_and_save(self):
        tool = SimpleNamespace(server_url="http://localhost:8000")
        should_skip, kwargs = _build_semantic_tool_kwargs(
            tool=tool,
            tool_name="web_search_mcp",
            user_msg="search this",
            url_in_query=None,
            wants_save_to_rag=True,
            enhanced_web_query="BTC sentiment",
            web_max_results=3,
            search_language="de",
            search_region="de-de",
            timezone="UTC",
        )

        assert should_skip is False
        assert kwargs["query"] == "BTC sentiment"
        assert "language" not in kwargs  # DuckDuckGo MCP uses region only
        assert kwargs["region"] == "de-de"
        assert kwargs["max_results"] == 3
        assert kwargs["save_to_rag"] is True

    def test_extract_webpage_without_url_skips(self):
        tool = SimpleNamespace(server_url="http://localhost:8000")
        should_skip, kwargs = _build_semantic_tool_kwargs(
            tool=tool,
            tool_name="extract_webpage_mcp",
            user_msg="extract content",
            url_in_query=None,
            wants_save_to_rag=False,
            enhanced_web_query="ignored",
            web_max_results=8,
            search_language="en",
            search_region="us-en",
            timezone="UTC",
        )

        assert should_skip is True
        assert kwargs == {}


class TestCalculatorExpressionExtraction:
    """Focused tests for calculator expression extraction helper."""

    def test_extracts_infix_expression(self):
        expression = _extract_calculator_expression("can you calculate 12 * (4 + 3) for me?")
        assert expression == "12 * (4 + 3)"

    def test_extracts_function_expression(self):
        expression = _extract_calculator_expression("please compute sqrt(16) now")
        assert expression == "sqrt(16)"

    def test_falls_back_to_full_message_when_no_expression(self):
        user_msg = "what is the weather today"
        expression = _extract_calculator_expression(user_msg)
        assert expression == user_msg


class TestForcedToolExecutionHelper:
    """Focused tests for forced-tool execution helper."""

    def test_run_forced_tool_success_records_outputs(self):
        tool = FakeTool("datetime_tool", "datetime", return_value="UTC 2026-03-13")
        tool_results = {}
        tool_execution_results = {}
        routing_metadata = {}
        executed_forced = set()

        _run_forced_tool(
            tool=tool,
            tool_name="datetime_tool",
            run_kwargs={"timezone": "UTC"},
            reason="date/time pattern detected",
            log_label="forced datetime",
            tool_results=tool_results,
            tool_execution_results=tool_execution_results,
            routing_metadata=routing_metadata,
            executed_forced=executed_forced,
        )

        assert tool_results["datetime_tool"] == "UTC 2026-03-13"
        assert tool_execution_results["datetime_tool"]["status"] == "success"
        assert routing_metadata["datetime_tool"] == "date/time pattern detected"
        assert "datetime_tool" in executed_forced

    def test_run_forced_tool_error_records_error_envelope(self):
        failing_tool = Mock()
        failing_tool._run.side_effect = ValueError("boom")
        tool_results = {}
        tool_execution_results = {}
        routing_metadata = {}
        executed_forced = set()

        _run_forced_tool(
            tool=failing_tool,
            tool_name="datetime_tool",
            run_kwargs={"timezone": "UTC"},
            reason="date/time pattern detected",
            log_label="forced datetime",
            tool_results=tool_results,
            tool_execution_results=tool_execution_results,
            routing_metadata=routing_metadata,
            executed_forced=executed_forced,
        )

        assert "Tool execution failed: boom" in tool_results["datetime_tool"]
        assert tool_execution_results["datetime_tool"]["status"] == "error"
        assert "datetime_tool" not in executed_forced


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
