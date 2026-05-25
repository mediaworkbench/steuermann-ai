"""Test tool invocation in LLM response generation."""

import pytest
from unittest.mock import Mock, patch

from universal_agentic_framework.tools.datetime.tool import DateTimeTool



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
