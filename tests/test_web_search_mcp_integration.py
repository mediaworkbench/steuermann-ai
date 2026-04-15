"""Integration tests for Web Search MCP Server with ToolRegistry."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from universal_agentic_framework.tools.registry import ToolRegistry, MCPServerTool
from universal_agentic_framework.config import load_tools_config
from universal_agentic_framework.embeddings import build_embedding_provider

# Force deterministic MCP URL for these tests regardless of outer environment.
os.environ["WEB_SEARCH_MCP_URL"] = "http://localhost:8000/mcp"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _make_mock_mcp_context(call_tool_return):
    """Create a mock streamable_http_client context yielding a mock session."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=call_tool_return)

    @asynccontextmanager
    async def _mock_session_ctx(read_stream, write_stream):
        yield mock_session

    @asynccontextmanager
    async def _mock_http_client(url):
        yield (AsyncMock(), AsyncMock(), AsyncMock())

    return _mock_http_client, _mock_session_ctx, mock_session


def _make_call_tool_result(text: str, is_error: bool = False):
    """Create a mock CallToolResult."""
    content_block = MagicMock()
    content_block.text = text
    result = MagicMock()
    result.content = [content_block]
    result.isError = is_error
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


class TestWebSearchMCPIntegration:
    """Tests for web_search_mcp integration with ToolRegistry."""

    def test_registry_discovers_web_search_mcp(self):
        """ToolRegistry should discover and load web_search_mcp tool."""
        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()

        # Check tool is discovered
        assert "web_search_mcp" in registry.tools
        tool = registry.tools["web_search_mcp"]

        # Verify it's an MCP server tool
        assert isinstance(tool, MCPServerTool)
        assert tool.name == "web_search_mcp"

    def test_web_search_mcp_has_correct_server_url(self):
        """web_search_mcp should have configured server_url."""
        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()
        tool = registry.tools["web_search_mcp"]

        # The tool should have server_url attribute from config
        assert hasattr(tool, "server_url")
        assert "/mcp" in tool.server_url

    @patch("mcp.ClientSession")
    @patch("mcp.client.streamable_http.streamablehttp_client")
    def test_web_search_mcp_invocation(self, mock_http_client, mock_session_cls):
        """web_search_mcp should invoke MCP call_tool with correct arguments."""
        call_result = _make_call_tool_result(
            '[{"title": "Result 1", "url": "https://example.com", "snippet": "Test"}]'
        )
        mock_http_ctx, mock_session_ctx, mock_session = _make_mock_mcp_context(call_result)
        mock_http_client.side_effect = mock_http_ctx
        mock_session_cls.side_effect = mock_session_ctx

        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()
        tool = registry.tools["web_search_mcp"]

        result = tool._run(query="test query")

        # Verify call_tool was invoked with the default tool name
        mock_session.call_tool.assert_called_once()
        call_args = mock_session.call_tool.call_args
        assert call_args[0][0] == "search"  # default tool
        assert call_args[1]["arguments"]["query"] == "test query"

    @patch("mcp.ClientSession")
    @patch("mcp.client.streamable_http.streamablehttp_client")
    def test_web_search_mcp_handles_error(self, mock_http_client, mock_session_cls):
        """web_search_mcp should handle MCP tool errors gracefully."""
        call_result = _make_call_tool_result("Rate limit exceeded", is_error=True)
        mock_http_ctx, mock_session_ctx, mock_session = _make_mock_mcp_context(call_result)
        mock_http_client.side_effect = mock_http_ctx
        mock_session_cls.side_effect = mock_session_ctx

        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()
        tool = registry.tools["web_search_mcp"]

        result = tool._run(query="test query")

        # Result should indicate error
        assert "error" in result.lower()

    @patch("mcp.ClientSession")
    @patch("mcp.client.streamable_http.streamablehttp_client")
    def test_web_search_mcp_connection_failure(self, mock_http_client, mock_session_cls):
        """web_search_mcp should handle connection failures gracefully."""
        @asynccontextmanager
        async def _failing_client(url):
            raise ConnectionError("Connection refused")
            yield  # pragma: no cover

        mock_http_client.side_effect = _failing_client

        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()
        tool = registry.tools["web_search_mcp"]

        result = tool._run(query="test query")

        assert "failed" in result.lower()


class TestWebSearchMCPSemanticRouting:
    """Tests for semantic routing of web search queries."""

    def test_search_query_matches_web_search_description(self):
        """Search-related queries should semantically match web_search_mcp."""
        import numpy as np

        model = build_embedding_provider(
            model_name="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            provider_type="remote",
            remote_endpoint="$EMBEDDING_SERVER/v1",
        )
        if getattr(model, "_fallback", False):
            pytest.skip("Semantic threshold assertions require real embedding model")

        # Get tool description
        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()
        tool = registry.tools["web_search_mcp"]
        tool_desc = f"{tool.name}: {tool.description}"

        # Test queries that should match web search
        search_queries = [
            "suche im Internet nach Python tutorials",
            "find information about machine learning",
            "search for recent news about AI",
            "look up documentation for FastAPI",
        ]

        tool_embedding = np.array(model.encode(tool_desc))

        for query in search_queries:
            query_embedding = np.array(model.encode(query))
            similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )

            # Should have reasonable similarity for search queries
            assert similarity > 0.2, f"Query '{query}' had low similarity: {similarity}"

    def test_non_search_query_lower_similarity(self):
        """Non-search queries should have lower similarity to web_search_mcp."""
        import numpy as np

        model = build_embedding_provider(
            model_name="text-embedding-granite-embedding-278m-multilingual",
            dimension=768,
            provider_type="remote",
            remote_endpoint="$EMBEDDING_SERVER/v1",
        )
        if getattr(model, "_fallback", False):
            pytest.skip("Semantic threshold assertions require real embedding model")

        tools_config = load_tools_config(config_dir=_repo_root() / "config")
        registry = ToolRegistry(config=tools_config, base_dir=_repo_root())
        registry.discover_and_load()
        tool = registry.tools["web_search_mcp"]
        tool_desc = f"{tool.name}: {tool.description}"
        tool_embedding = np.array(model.encode(tool_desc))

        # Test queries that should NOT match web search
        non_search_queries = [
            "what time is it now",
            "calculate 5 + 3",
            "wie alt bin ich wenn ich 1990 geboren bin",
        ]

        for query in non_search_queries:
            query_embedding = np.array(model.encode(query))
            similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )

            # Should have lower similarity than search queries
            # Note: datetime queries should go to datetime_tool, not web_search
            assert similarity < 0.5, f"Non-search query '{query}' had high similarity: {similarity}"
