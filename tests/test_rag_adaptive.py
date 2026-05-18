"""Tests for adaptive RAG: intent short-circuit, per-session toggle, and transparency fields."""

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from universal_agentic_framework.orchestration.helpers.intent_detection import (
    detect_tool_routing_intents as _detect_intents,
    _RAG_SKIP_SHORT_QUERY_CHARS,
)
from universal_agentic_framework.orchestration.graph_builder import node_retrieve_knowledge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rag_config(enabled=True, collection_name="test-collection", *, top_k=5, score_threshold=0.6):
    return SimpleNamespace(
        enabled=enabled,
        collection_name=collection_name,
        top_k=top_k,
        score_threshold=score_threshold,
        with_payload=["text"],
        with_vectors=False,
        timeout_seconds=30,
    )


def _make_core_config(rag_enabled=True):
    return SimpleNamespace(
        fork=SimpleNamespace(name="test-fork", language="en"),
        rag=_make_rag_config(enabled=rag_enabled),
        memory=SimpleNamespace(
            embeddings=SimpleNamespace(dimension=768),
            vector_store=SimpleNamespace(host="qdrant", port=6333),
        ),
        llm=SimpleNamespace(
            get_role_model_name=lambda _role, _lang: "test/model",
            get_embedding_provider_type=lambda: "remote",
            get_embedding_remote_endpoint=lambda: "http://localhost:11434/v1",
        ),
    )


def _base_state(**overrides):
    state = {
        "messages": [{"role": "user", "content": "What is type 2 diabetes?"}],
        "user_settings": {},
        "prefilter_intents": {},
        "knowledge_context": [],
    }
    state.update(overrides)
    return state


def _mock_httpx_post(qdrant_hits):
    """Return a mock for httpx.post that yields the given Qdrant hits."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": qdrant_hits}
    mock_response.raise_for_status = MagicMock()
    return MagicMock(return_value=mock_response)


# ---------------------------------------------------------------------------
# Part A — skip_rag intent detection
# ---------------------------------------------------------------------------

class TestSkipRagIntentDetection:
    """skip_rag should be True for trivial queries that cannot benefit from RAG."""

    # Greetings — all supported languages
    @pytest.mark.parametrize("greeting", [
        "Hello",
        "Hi there",
        "Hey!",
        "Hallo",
        "Guten Tag",
        "Guten Morgen, wie geht's?",
        "Bonjour",
        "Salut",
        "Ciao",
        "Yo",
    ])
    def test_greeting_triggers_skip_rag(self, greeting):
        intents = _detect_intents(greeting, language="en")
        assert intents["skip_rag"] is True, f"Expected skip_rag=True for greeting: {greeting!r}"

    # Short pure math — no web intent
    @pytest.mark.parametrize("msg", [
        "What is 5 * 12?",
        "sqrt(144)",
        "berechne 200 / 4",
        "calculate 15% of 80",
    ])
    def test_short_math_without_web_triggers_skip_rag(self, msg):
        assert len(msg) < _RAG_SKIP_SHORT_QUERY_CHARS, f"Test message too long: {msg!r}"
        intents = _detect_intents(msg, language="en")
        assert intents["mentions_calculation"] is True
        assert intents["skip_rag"] is True, f"Expected skip_rag=True for: {msg!r}"

    # Short datetime — no web intent (use keywords in the actual detector list)
    @pytest.mark.parametrize("msg", [
        "What time is it?",           # "time" is in datetime_keywords
        "What's today's date?",       # "date" is in datetime_keywords
        "What day is it heute?",      # "heute" is in datetime_keywords
        "Wie viel Uhr ist es?",       # "uhr" is in datetime_keywords
    ])
    def test_short_datetime_without_web_triggers_skip_rag(self, msg):
        assert len(msg) < _RAG_SKIP_SHORT_QUERY_CHARS, f"Test message too long: {msg!r}"
        intents = _detect_intents(msg, language="en")
        assert intents["mentions_datetime"] is True
        assert intents["skip_rag"] is True, f"Expected skip_rag=True for: {msg!r}"

    # Tool meta-questions — must match the actual asks_about_tools keyword list
    @pytest.mark.parametrize("msg", [
        "What tools do you have?",             # "tools do you have"
        "What tools are available to you?",    # "available"
        "Which tools are available?",          # "available"
        "What tools hast du?",                 # "tools hast du"
    ])
    def test_tool_meta_question_triggers_skip_rag(self, msg):
        intents = _detect_intents(msg, language="en")
        assert intents["asks_about_tools"] is True, f"Expected asks_about_tools=True for: {msg!r}"
        assert intents["skip_rag"] is True, f"Expected skip_rag=True for: {msg!r}"

    # Substantive knowledge queries — must NOT skip
    @pytest.mark.parametrize("msg", [
        "What are the diagnostic criteria for type 2 diabetes?",
        "Explain the mechanism of action of metformin",
        "How does aspirin inhibit COX enzymes?",
        "Was sind die Symptome von Herzinsuffizienz?",
    ])
    def test_knowledge_query_does_not_skip_rag(self, msg):
        intents = _detect_intents(msg, language="en")
        assert intents["skip_rag"] is False, f"Expected skip_rag=False for knowledge query: {msg!r}"

    def test_short_math_with_web_intent_does_not_skip_rag(self):
        """Math + explicit web search should still run RAG (web search supersedes the math skip)."""
        msg = "search the web for math"
        intents = _detect_intents(msg, language="en")
        assert intents["mentions_web_search"] is True
        assert intents["skip_rag"] is False

    def test_long_math_query_does_not_skip_rag(self):
        """A long math query exceeds the short-query threshold and should not be skipped."""
        msg = "Can you calculate the compound interest over 10 years at 5% per annum for me?"
        assert len(msg) >= _RAG_SKIP_SHORT_QUERY_CHARS
        intents = _detect_intents(msg, language="en")
        assert intents["skip_rag"] is False

    def test_skip_rag_key_always_present(self):
        """skip_rag must be in the returned dict regardless of query content."""
        intents = _detect_intents("anything", language="en")
        assert "skip_rag" in intents


# ---------------------------------------------------------------------------
# Part B — node_retrieve_knowledge skip paths
# ---------------------------------------------------------------------------

class TestNodeRetrieveKnowledgeSkipPaths:
    """node_retrieve_knowledge should return early (no Qdrant call) on all skip conditions."""

    @patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_user_toggle_disabled_skips_qdrant(self, mock_config, mock_features):
        """When rag_config.enabled=False in user_settings, Qdrant must not be called."""
        mock_config.return_value = _make_core_config()
        mock_features.return_value = SimpleNamespace(rag_retrieval=True)

        state = _base_state(user_settings={"rag_config": {"enabled": False}})

        with patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:
            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result["knowledge_context"] == []
        assert result.get("rag_attempted", False) is False

    @patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_intent_skip_rag_skips_qdrant(self, mock_config, mock_features):
        """When prefilter_intents.skip_rag=True, Qdrant must not be called."""
        mock_config.return_value = _make_core_config()
        mock_features.return_value = SimpleNamespace(rag_retrieval=True)

        state = _base_state(prefilter_intents={"skip_rag": True})

        with patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:
            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result["knowledge_context"] == []
        assert result.get("rag_attempted", False) is False

    @patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_features_flag_disabled_skips_qdrant(self, mock_config, mock_features):
        """Feature flag rag_retrieval=False must prevent any Qdrant call."""
        mock_config.return_value = _make_core_config()
        mock_features.return_value = SimpleNamespace(rag_retrieval=False)

        state = _base_state()

        with patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:
            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result["knowledge_context"] == []
        assert result.get("rag_attempted", False) is False

    @patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_config_rag_disabled_skips_qdrant(self, mock_config, mock_features):
        """rag.enabled=false in profile config must prevent any Qdrant call."""
        mock_config.return_value = _make_core_config(rag_enabled=False)
        mock_features.return_value = SimpleNamespace(rag_retrieval=True)

        state = _base_state()

        with patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:
            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result["knowledge_context"] == []

    @patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_user_toggle_enabled_reaches_qdrant(self, mock_config, mock_features):
        """rag_config.enabled=True must proceed to the embedding + Qdrant path."""
        mock_config.return_value = _make_core_config()
        mock_features.return_value = SimpleNamespace(rag_retrieval=True)

        embedder_mock = MagicMock()
        embedder_mock.encode.return_value = [0.1] * 768

        state = _base_state(user_settings={"rag_config": {"enabled": True}})

        with patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider", return_value=embedder_mock), \
             patch("httpx.post", _mock_httpx_post([])):
            result = node_retrieve_knowledge(state)

        # The embedding provider must have been called — confirming the Qdrant path was reached
        embedder_mock.encode.assert_called()
        assert result.get("rag_attempted") is True

    @patch("universal_agentic_framework.orchestration.graph_builder.load_features_config")
    @patch("universal_agentic_framework.orchestration.graph_builder.load_core_config")
    def test_empty_message_skips_without_error(self, mock_config, mock_features):
        """Empty user message should return gracefully with no Qdrant call."""
        mock_config.return_value = _make_core_config()
        mock_features.return_value = SimpleNamespace(rag_retrieval=True)

        state = _base_state(messages=[{"role": "user", "content": ""}])

        with patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:
            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result["knowledge_context"] == []


# ---------------------------------------------------------------------------
# Part C — rag_attempted / rag_doc_count state fields
# ---------------------------------------------------------------------------

class TestRagTransparencyFields:
    """rag_attempted and rag_doc_count must reflect actual Qdrant activity."""

    def _run_with_qdrant_response(self, qdrant_hits, user_settings=None, intents=None):
        """Run node_retrieve_knowledge with a mocked httpx.post response."""
        state = _base_state(
            user_settings=user_settings or {},
            prefilter_intents=intents or {},
        )

        embedder_mock = MagicMock()
        embedder_mock.encode.return_value = [0.1] * 768

        core_config = _make_core_config()
        features_config = SimpleNamespace(rag_retrieval=True)

        with patch("universal_agentic_framework.orchestration.graph_builder.load_core_config", return_value=core_config), \
             patch("universal_agentic_framework.orchestration.graph_builder.load_features_config", return_value=features_config), \
             patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider", return_value=embedder_mock), \
             patch("httpx.post", _mock_httpx_post(qdrant_hits)):
            return node_retrieve_knowledge(state)

    def test_rag_attempted_true_when_qdrant_queried(self):
        result = self._run_with_qdrant_response(qdrant_hits=[])
        assert result.get("rag_attempted") is True

    def test_rag_doc_count_zero_when_no_results(self):
        result = self._run_with_qdrant_response(qdrant_hits=[])
        assert result.get("rag_attempted") is True
        assert result.get("rag_doc_count") == 0
        assert result["knowledge_context"] == []

    def test_rag_doc_count_reflects_injected_results(self):
        """When Qdrant returns hits above the score threshold, rag_doc_count must reflect them."""
        # Use unique ids so deduplication doesn't collapse results from the keyword search
        hits = [
            {"id": "doc-a", "payload": {"text": "doc A", "file_path": "a.pdf"}, "score": 0.9},
            {"id": "doc-b", "payload": {"text": "doc B", "file_path": "b.pdf"}, "score": 0.8},
        ]
        result = self._run_with_qdrant_response(qdrant_hits=hits)
        assert result.get("rag_attempted") is True
        # Both hits are above the 0.6 threshold; deduplication by id keeps both unique docs
        assert result.get("rag_doc_count") == 2
        assert len(result["knowledge_context"]) == 2

    def test_rag_doc_count_zero_when_hits_below_threshold(self):
        """Hits below score_threshold must be filtered out even though Qdrant was queried."""
        hits = [
            {"id": "low-1", "payload": {"text": "noisy doc"}, "score": 0.3},
            {"id": "low-2", "payload": {"text": "borderline doc"}, "score": 0.55},
        ]
        result = self._run_with_qdrant_response(qdrant_hits=hits)
        assert result.get("rag_attempted") is True
        assert result.get("rag_doc_count") == 0

    def test_rag_attempted_absent_when_user_toggle_off(self):
        """rag_attempted must not be set when the user toggle skips Qdrant."""
        state = _base_state(user_settings={"rag_config": {"enabled": False}})

        with patch("universal_agentic_framework.orchestration.graph_builder.load_features_config") as mock_features, \
             patch("universal_agentic_framework.orchestration.graph_builder.load_core_config") as mock_config, \
             patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:

            mock_config.return_value = _make_core_config()
            mock_features.return_value = SimpleNamespace(rag_retrieval=True)

            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result.get("rag_attempted", False) is False

    def test_rag_attempted_absent_when_intent_skip(self):
        """rag_attempted must not be set when the intent short-circuit fires."""
        state = _base_state(prefilter_intents={"skip_rag": True})

        with patch("universal_agentic_framework.orchestration.graph_builder.load_features_config") as mock_features, \
             patch("universal_agentic_framework.orchestration.graph_builder.load_core_config") as mock_config, \
             patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider") as mock_embed:

            mock_config.return_value = _make_core_config()
            mock_features.return_value = SimpleNamespace(rag_retrieval=True)

            result = node_retrieve_knowledge(state)

        mock_embed.assert_not_called()
        assert result.get("rag_attempted", False) is False

    def test_user_toggle_re_enabled_runs_qdrant(self):
        """After re-enabling the toggle, rag_attempted must be True and results returned."""
        hits = [{"id": "doc-x", "payload": {"text": "relevant doc", "file_path": "x.pdf"}, "score": 0.85}]
        result = self._run_with_qdrant_response(
            qdrant_hits=hits,
            user_settings={"rag_config": {"enabled": True}},
        )
        assert result.get("rag_attempted") is True
        assert result.get("rag_doc_count") == 1
