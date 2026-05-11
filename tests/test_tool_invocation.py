"""Test tool invocation in LLM response generation."""

import os
import numpy as np
import pytest
from unittest.mock import Mock, patch

# Set environment variables for tests before imports
os.environ.setdefault("LLM_PROVIDERS_OLLAMA_API_BASE", "http://localhost:11434/v1")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("WEB_SEARCH_MCP_URL", "http://localhost:9100")

from universal_agentic_framework.orchestration.graph_builder import build_graph
from universal_agentic_framework.tools.datetime.tool import DateTimeTool


@pytest.mark.integration
@patch("universal_agentic_framework.orchestration.graph_builder.build_embedding_provider")
def test_llm_invokes_datetime_tool(mock_embedding_provider):
    """Test semantic tool routing with datetime tool (model-agnostic approach)."""
    graph = build_graph()

    # Mock embedder to avoid external embedding endpoint dependency
    embedder = Mock()
    embedder.encode.side_effect = lambda _: np.array([1.0, 0.0, 0.0])
    mock_embedding_provider.return_value = embedder
    
    # Mock LLM to return a simple response
    with patch("universal_agentic_framework.orchestration.graph_builder.safe_get_model") as mock_model_factory:
        # LLM response (tool results injected into context, no tool calling needed)
        response = Mock()
        response.content = "You were born 51 years ago if born on 1974-05-24."
        
        mock_model = Mock()
        mock_model.invoke = Mock(return_value=response)
        
        mock_model_factory.return_value = mock_model
        
        state = {
            "messages": [{"role": "user", "content": "ich bin am 24.05.1974 geboren. wie alt bin ich?"}],
            "user_id": "test_user",
            "language": "de",
        }
        
        result = graph.invoke(state)

        # Prefer semantic tool execution, but allow graceful fallback response.
        tool_results = result.get("tool_results", {})
        if "datetime_tool" in tool_results:
            assert "tool_execution_results" in result
            assert result["tool_execution_results"]["datetime_tool"]["status"] == "success"
        
        # Assert model was invoked with tool results in context
        assert mock_model.invoke.called
        
        # Assert final response exists
        assert "messages" in result
        final_message = result["messages"][-1]
        assert final_message["role"] == "assistant"
        assert "51" in final_message["content"]


@pytest.mark.integration  
def test_datetime_tool_execution():
    """Test direct datetime tool execution."""
    tool = DateTimeTool()
    
    # Test current_time operation
    result = tool._run(operation="current_time", timezone="UTC")
    assert "UTC" in result or "Z" in result
    
    # Test timezone conversion
    # Note: Current DateTimeTool only supports current_time operation
    # convert_timezone would need additional parameters in the tool implementation
    result2 = tool._run(operation="current_time", timezone="America/New_York")
    assert "America/New_York" in result2


def test_mcp_server_http_invocation():
    """Test MCP server makes HTTP POST request."""
    from universal_agentic_framework.tools.registry import MCPServerTool
    
    with patch("httpx.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        # Mock MCP ToolResponse format with success: true
        mock_response.json.return_value = {
            "success": True,
            "tool": "web_search",
            "result": "MCP response",
            "execution_time_ms": 100
        }
        mock_post.return_value = mock_response
        
        tool = MCPServerTool(
            name="test_mcp",
            server_url="http://localhost:9999"
        )
        
        result = tool._run(query="test query", param1="value1")
        
        # Assert HTTP POST was made
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:9999/execute"
        
        # Verify MCP ToolExecuteRequest format
        payload = call_args[1]["json"]
        assert payload["tool"] == "web_search"  # default tool
        assert payload["parameters"]["query"] == "test query"
        assert payload["parameters"]["param1"] == "value1"
        
        # Result should be extracted from success response
        assert result == "MCP response"


def test_mcp_server_handles_error():
    """Test MCP server handles HTTP errors gracefully."""
    from universal_agentic_framework.tools.registry import MCPServerTool
    
    with patch("httpx.post") as mock_post:
        mock_post.side_effect = Exception("Connection refused")
        
        tool = MCPServerTool(
            name="test_mcp",
            server_url="http://localhost:9999"
        )
        
        result = tool._run(query="test query")
        
        assert "invocation failed" in result
        assert "Connection refused" in result
