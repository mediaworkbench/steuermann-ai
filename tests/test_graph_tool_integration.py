"""Integration tests for graph builder with tool registry."""

from pathlib import Path

import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from universal_agentic_framework.orchestration.graph_builder import build_graph
from universal_agentic_framework.tools.datetime.tool import DateTimeTool


@pytest.mark.integration
def test_graph_loads_tools():
    """Verify that build_graph loads tools via registry and populates state."""
    graph = build_graph()
    
    state = {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "test_user",
        "language": "de",
    }
    
    result = graph.invoke(state)
    
    # Assert tools were loaded
    assert "loaded_tools" in result
    assert len(result["loaded_tools"]) > 0
    
    # Assert DateTime tool is present (enabled by default in config/tools.yaml)
    tool_names = [tool.name for tool in result["loaded_tools"]]
    assert "datetime_tool" in tool_names
    
    # Assert DateTime tool is correct type
    datetime_tools = [t for t in result["loaded_tools"] if isinstance(t, DateTimeTool)]
    assert len(datetime_tools) == 1


@pytest.mark.integration
def test_graph_skips_disabled_tools():
    """Verify that disabled tools (like mcp_stub) are not loaded."""
    graph = build_graph()
    
    state = {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "test_user",
        "language": "de",
    }
    
    result = graph.invoke(state)
    
    # Assert mcp_stub is NOT loaded (disabled in config/tools.yaml)
    tool_names = [tool.name for tool in result.get("loaded_tools", [])]
    assert "mcp_stub" not in tool_names


class DummyModel:
    def __init__(self, output: str = "model output"):
        self.output = output
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        # In structured tool-calling mode, return a JSON tool call when prompted
        system_msg = messages[0].content if messages else ""
        if "respond with ONLY a JSON object" in system_msg:
            return SimpleNamespace(content='{"tool": "datetime_tool", "args": {}}')
        return SimpleNamespace(content=self.output)


@patch("httpx.post")
@patch("universal_agentic_framework.orchestration.graph_builder._safe_get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
@pytest.mark.integration
def test_graph_injects_tool_and_knowledge_context(
    mock_provider_factory, mock_config, mock_features, mock_model_factory, mock_httpx_post
):
    """Full graph should inject both tool results and RAG context into system prompt."""

    # Embedder mock (query + one tool description)
    mock_embedder = MagicMock()

    def fake_encode(data):
        if isinstance(data, list):
            return np.ones((len(data), 3))
        return np.ones(3)

    mock_embedder.encode.side_effect = fake_encode
    mock_provider_factory.return_value = mock_embedder

    # Config mock
    mock_config.return_value = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en", timezone="UTC"),
        memory=SimpleNamespace(
            vector_store=SimpleNamespace(type="qdrant", host="qdrant", port=6333, collection_prefix="test"),
            embeddings=SimpleNamespace(model="test-model", dimension=3, provider="local", remote_endpoint=None),
        ),
        tool_routing=SimpleNamespace(similarity_threshold=1.1, embedding_model="test-model", top_k=None, intent_boost=0.2, max_retries=2, min_top_score=0.7, min_spread=0.10),
        tokens=SimpleNamespace(
            default_budget=1000,
            per_node_budgets={"response_node": 1000, "summarization_node": 1000, "update_memory": 1000},
        ),
        llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}, tool_calling="structured"))),
        rag=None,
    )
    mock_features.return_value = SimpleNamespace(rag_retrieval=True, long_term_memory=False)

    # LLM mock
    dummy_model = DummyModel()
    mock_model_factory.return_value = dummy_model

    # Qdrant HTTP mock
    fake_resp = Mock()
    fake_resp.json.return_value = {
        "result": [
            {
                "payload": {"text": "Doc text", "file_name": "doc1.md"},
                "score": 0.9,
            }
        ]
    }
    fake_resp.raise_for_status.return_value = None
    mock_httpx_post.return_value = fake_resp

    graph = build_graph()

    state = {
        "messages": [{"role": "user", "content": "what time is it today?"}],
        "user_id": "u123",
        "language": "en",
    }

    result = graph.invoke(state)

    # invocations[0] = structured tool-calling node prompt
    # invocations[1] = response node prompt (where tool results are injected)
    assert len(dummy_model.invocations) >= 2
    system_prompt = dummy_model.invocations[1][0].content

    assert "=== TOOL RESULTS ===" in system_prompt
    assert "datetime_tool" in system_prompt
    assert "=== WISSENSDATENBANK ===" in system_prompt
    assert "Doc text" in system_prompt
    assert "Cite sources using numbered references" not in system_prompt
    assert result["messages"][-1]["content"] == "model output"


@patch("httpx.post")
@patch("universal_agentic_framework.orchestration.graph_builder._safe_get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
@pytest.mark.integration
def test_rag_request_uses_config(
    mock_provider_factory, mock_config, mock_features, mock_model_factory, mock_httpx_post
):
    """RAG retrieval should respect collection, top_k, score threshold, and payload settings."""
    mock_embedder = MagicMock()

    def fake_encode(data):
        if isinstance(data, list):
            return np.ones((len(data), 3))
        return np.ones(3)

    mock_embedder.encode.side_effect = fake_encode
    mock_provider_factory.return_value = mock_embedder

    mock_config.return_value = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en", timezone="UTC"),
        memory=SimpleNamespace(
            vector_store=SimpleNamespace(type="qdrant", host="qdrant", port=6333, collection_prefix="test"),
            embeddings=SimpleNamespace(model="test-model", dimension=3, provider="local", remote_endpoint=None),
        ),
        tokens=SimpleNamespace(
            default_budget=1000,
            per_node_budgets={"response_node": 1000, "summarization_node": 1000, "update_memory": 1000},
        ),
        llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        rag=SimpleNamespace(
            enabled=True,
            collection_name="my-collection",
            top_k=2,
            score_threshold=0.42,
            with_payload=["text"],
            with_vectors=False,
            timeout_seconds=5,
        ),
    )
    mock_features.return_value = SimpleNamespace(rag_retrieval=True)

    dummy_model = DummyModel()
    mock_model_factory.return_value = dummy_model

    fake_resp = Mock()
    fake_resp.json.return_value = {"result": []}
    fake_resp.raise_for_status.return_value = None
    mock_httpx_post.return_value = fake_resp

    graph = build_graph()
    state = {
        "messages": [{"role": "user", "content": "tell me about docs"}],
        "user_id": "u123",
        "language": "en",
    }

    graph.invoke(state)

    assert mock_httpx_post.called
    url = mock_httpx_post.call_args.args[0]
    payload = mock_httpx_post.call_args.kwargs["json"]
    timeout = mock_httpx_post.call_args.kwargs["timeout"]

    assert url.endswith("/collections/my-collection/points/search")
    assert payload["limit"] == 2
    assert payload["score_threshold"] == 0.42
    assert payload["with_payload"] == ["text"]
    assert payload["with_vector"] is False
    assert timeout == 5


@patch("httpx.post")
@patch("universal_agentic_framework.orchestration.graph_builder._safe_get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
@pytest.mark.integration
def test_rag_disabled_via_features(
    mock_provider_factory, mock_config, mock_features, mock_model_factory, mock_httpx_post
):
    """RAG retrieval should be skipped when the feature flag is disabled."""
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.ones(3)
    mock_provider_factory.return_value = mock_embedder

    mock_config.return_value = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en", timezone="UTC"),
        memory=SimpleNamespace(
            vector_store=SimpleNamespace(type="qdrant", host="qdrant", port=6333, collection_prefix="test"),
            embeddings=SimpleNamespace(model="test-model", dimension=3, provider="local", remote_endpoint=None),
        ),
        tool_routing=SimpleNamespace(similarity_threshold=1.1, embedding_model="test-model", top_k=None, min_top_score=0.7, min_spread=0.10),
        tokens=SimpleNamespace(
            default_budget=1000,
            per_node_budgets={"response_node": 1000, "summarization_node": 1000, "update_memory": 1000},
        ),
        llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
    )
    mock_features.return_value = SimpleNamespace(rag_retrieval=False)

    dummy_model = DummyModel()
    mock_model_factory.return_value = dummy_model

    graph = build_graph()
    state = {
        "messages": [{"role": "user", "content": "tell me about docs"}],
        "user_id": "u123",
        "language": "en",
    }

    result = graph.invoke(state)

    assert result.get("knowledge_context") == []
    assert not mock_httpx_post.called


@patch("httpx.post")
@patch("universal_agentic_framework.orchestration.graph_builder._safe_get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
@pytest.mark.integration
def test_rag_keyword_fallback_search(
    mock_provider_factory, mock_config, mock_features, mock_model_factory, mock_httpx_post
):
    """RAG should attempt keyword-focused fallback search when full query returns empty or low results."""
    mock_embedder = MagicMock()

    def fake_encode(data):
        if isinstance(data, list):
            return np.ones((len(data), 3))
        return np.ones(3)

    mock_embedder.encode.side_effect = fake_encode
    mock_provider_factory.return_value = mock_embedder

    mock_config.return_value = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en", timezone="UTC"),
        memory=SimpleNamespace(
            vector_store=SimpleNamespace(type="qdrant", host="qdrant", port=6333, collection_prefix="test"),
            embeddings=SimpleNamespace(model="test-model", dimension=3, provider="local", remote_endpoint=None),
        ),
        tool_routing=SimpleNamespace(similarity_threshold=1.1, embedding_model="test-model", top_k=None, min_top_score=0.7, min_spread=0.10),
        tokens=SimpleNamespace(
            default_budget=1000,
            per_node_budgets={"response_node": 1000, "summarization_node": 1000, "update_memory": 1000},
        ),
        llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        rag=SimpleNamespace(
            enabled=True,
            collection_name="test-collection",
            top_k=5,
            score_threshold=0.75,
            with_payload=["text", "file_path"],
            with_vectors=False,
            timeout_seconds=10,
        ),
    )
    mock_features.return_value = SimpleNamespace(rag_retrieval=True)

    dummy_model = DummyModel()
    mock_model_factory.return_value = dummy_model

    # Mock Qdrant responses: 
    # First search (full query) returns empty (no matches above threshold)
    # Second search (keyword fallback) returns good results
    first_search_response = Mock()
    first_search_response.json.return_value = {
        "result": []  # No results above threshold for full query
    }
    first_search_response.raise_for_status.return_value = None

    second_search_response = Mock()
    second_search_response.json.return_value = {
        "result": [
            {"id": 2, "score": 0.8, "payload": {"text": "ezetimib is a statin", "file_path": "drugs.md"}},
            {"id": 3, "score": 0.76, "payload": {"text": "ezetimib interactions", "file_path": "interactions.md"}},
        ]
    }
    second_search_response.raise_for_status.return_value = None

    mock_httpx_post.side_effect = [first_search_response, second_search_response]

    graph = build_graph()
    state = {
        "messages": [{"role": "user", "content": "what is ezetimib?"}],
        "user_id": "u123",
        "language": "en",
    }

    result = graph.invoke(state)

    # Verify fallback search was called (two POST requests to Qdrant)
    assert mock_httpx_post.call_count == 2
    # Verify results from fallback search are in knowledge context
    assert result.get("knowledge_context") is not None
    assert len(result["knowledge_context"]) == 2
    assert any("ezetimib" in chunk.get("text", "").lower() for chunk in result["knowledge_context"])


@patch("httpx.post")
@patch("universal_agentic_framework.orchestration.graph_builder._safe_get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
@pytest.mark.integration
def test_response_url_stripping_guardrail(
    mock_provider_factory, mock_config, mock_features, mock_model_factory, mock_httpx_post
):
    """Response post-processing should strip URLs not in allowed_sources list."""
    mock_embedder = MagicMock()

    def fake_encode(data):
        if isinstance(data, list):
            return np.ones((len(data), 3))
        return np.ones(3)

    mock_embedder.encode.side_effect = fake_encode
    mock_provider_factory.return_value = mock_embedder

    mock_config.return_value = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en", timezone="UTC"),
        memory=SimpleNamespace(
            vector_store=SimpleNamespace(type="qdrant", host="qdrant", port=6333, collection_prefix="test"),
            embeddings=SimpleNamespace(model="test-model", dimension=3, provider="local", remote_endpoint=None),
        ),
        tool_routing=SimpleNamespace(similarity_threshold=1.1, embedding_model="test-model", top_k=None, min_top_score=0.7, min_spread=0.10),
        tokens=SimpleNamespace(
            default_budget=1000,
            per_node_budgets={"response_node": 1000, "summarization_node": 1000, "update_memory": 1000},
        ),
        llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        rag=SimpleNamespace(
            enabled=False,
            collection_name="test-collection",
            top_k=5,
            score_threshold=None,
            with_payload=["text", "file_path"],
            with_vectors=False,
            timeout_seconds=10,
        ),
    )
    mock_features.return_value = SimpleNamespace(rag_retrieval=False, long_term_memory=False)

    # Mock LLM response with mixed URLs (trusted and untrusted)
    response_text = """According to examples at https://example.com/doc1, ezetimib is effective.
    See also https://untrusted-site.com/fake for more info.
    The documentation at https://example.com/doc2 confirms this."""
    
    dummy_model = DummyModel(output=response_text)
    mock_model_factory.return_value = dummy_model

    graph = build_graph()
    state = {
        "messages": [{"role": "user", "content": "tell me about ezetimib"}],
        "user_id": "u123",
        "language": "en",
        "knowledge_context": [
            {"payload": {"text": "ezetimib info", "file_path": "example.com/doc1"}},
            {"payload": {"text": "ezetimib interactions", "file_path": "example.com/doc2"}},
        ],
        "loaded_tools": [],
    }

    result = graph.invoke(state)

    # Verify response has untrusted URL stripped
    final_response = result["messages"][-1]["content"] if result.get("messages") else ""
    assert "untrusted-site.com" not in final_response
    # At least one trusted URL source indicator or "source omitted" should be present
    assert "example.com" in final_response or "source omitted" in final_response


@patch("universal_agentic_framework.orchestration.graph_builder._safe_get_model")
@patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
@patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
@pytest.mark.integration
def test_long_term_memory_disabled_via_features(
    mock_provider_factory, mock_config, mock_features, mock_model_factory
):
    """Long-term memory load/update nodes should be skipped when feature flag is disabled."""
    mock_embedder = MagicMock()

    def fake_encode(data):
        if isinstance(data, list):
            return np.ones((len(data), 3))
        return np.ones(3)

    mock_embedder.encode.side_effect = fake_encode
    mock_provider_factory.return_value = mock_embedder

    mock_config.return_value = SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en", timezone="UTC"),
        memory=SimpleNamespace(
            vector_store=SimpleNamespace(type="qdrant", host="qdrant", port=6333, collection_prefix="test"),
            embeddings=SimpleNamespace(model="test-model", dimension=3, provider="local", remote_endpoint=None),
        ),
        tool_routing=SimpleNamespace(similarity_threshold=1.1, embedding_model="test-model", top_k=None, min_top_score=0.7, min_spread=0.10),
        tokens=SimpleNamespace(
            default_budget=1000,
            per_node_budgets={"response_node": 1000, "summarization_node": 1000, "update_memory": 1000},
        ),
        llm=SimpleNamespace(providers=SimpleNamespace(primary=SimpleNamespace(type="ollama", models={"en": "llama"}))),
        rag=SimpleNamespace(
            enabled=False,
            collection_name="test-collection",
            top_k=5,
            score_threshold=None,
            with_payload=["text"],
            with_vectors=False,
            timeout_seconds=10,
        ),
    )
    # Disable long-term memory via features flag (default is already off but be explicit)
    mock_features.return_value = SimpleNamespace(rag_retrieval=False, long_term_memory=False)

    dummy_model = DummyModel()
    mock_model_factory.return_value = dummy_model

    graph = build_graph()
    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "user_id": "u123",
        "language": "en",
    }

    result = graph.invoke(state)

    # Verify long-term memory was not loaded (feature flag blocks it)
    # The loaded_memory key should not be present or should be empty/default
    # Important: When feature is disabled, the node skips memory loading entirely
    assert result.get("loaded_memory") is None or result.get("loaded_memory") == []
