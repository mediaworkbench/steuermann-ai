"""Tests for ToolRegistry covering LangChain and MCP server stubs."""

from pathlib import Path
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from universal_agentic_framework.tools.registry import ToolRegistry, MCPServerTool
from universal_agentic_framework.tools.datetime.tool import DateTimeTool


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_loads_datetime_tool() -> None:
    registry = ToolRegistry(
        {
            "tools": [
                {
                    "name": "datetime_tool",
                    "path": "universal_agentic_framework/tools/datetime",
                    "enabled": True,
                }
            ]
        },
        base_dir=_repo_root(),
    )

    tools = registry.discover_and_load()

    assert any(isinstance(tool, DateTimeTool) for tool in tools)
    assert "datetime_tool" in registry.tools


def _write_mcp_manifest(tmp_path: Path, *, server_url: str = "http://localhost:9999") -> Path:
    """Create a temporary ``type: mcp_server`` tool manifest and return its dir."""
    tool_dir = tmp_path / "temp_mcp"
    tool_dir.mkdir(parents=True)
    tool_dir.joinpath("tool.yaml").write_text(
        f"""
name: temp_mcp
type: mcp_server
config:
  server_url: {server_url}
        """,
        encoding="utf-8",
    )
    return tool_dir


def test_loads_mcp_server_tool(tmp_path: Path) -> None:
    tool_dir = _write_mcp_manifest(tmp_path)
    registry = ToolRegistry(
        {"tools": [{"name": "temp_mcp", "path": str(tool_dir)}]},
        base_dir=_repo_root(),
    )

    registry.discover_and_load()

    assert "temp_mcp" in registry.tools
    tool = registry.tools["temp_mcp"]
    assert isinstance(tool, MCPServerTool)
    assert tool.server_url == "http://localhost:9999"
    # MCP now makes real HTTP calls, so expect connection failure message
    result = tool._run("ping")
    assert "invocation failed" in result or "Connection refused" in result


def test_mcp_manifest_default_config_used(tmp_path: Path) -> None:
    tool_dir = _write_mcp_manifest(tmp_path, server_url="http://manifest-default:9999")
    registry = ToolRegistry(
        {"tools": [{"name": "temp_mcp", "path": str(tool_dir)}]},
        base_dir=_repo_root(),
    )

    registry.discover_and_load()

    tool = registry.tools["temp_mcp"]
    assert isinstance(tool, MCPServerTool)
    assert tool.server_url == "http://manifest-default:9999"


def test_mcp_config_override_wins(tmp_path: Path) -> None:
    tool_dir = _write_mcp_manifest(tmp_path, server_url="http://manifest-default:9999")
    registry = ToolRegistry(
        {
            "tools": [
                {
                    "name": "temp_mcp",
                    "path": str(tool_dir),
                    "config": {"server_url": "http://override:8888"},
                }
            ]
        },
        base_dir=_repo_root(),
    )

    registry.discover_and_load()

    assert registry.tools["temp_mcp"].server_url == "http://override:8888"


def test_explicit_loading_mode_does_not_auto_discover() -> None:
    """Default explicit mode should only load tools declared in config."""
    registry = ToolRegistry(
        {
            "tools": [
                {
                    "name": "datetime_tool",
                    "path": "universal_agentic_framework/tools/datetime",
                    "enabled": True,
                }
            ]
        },
        base_dir=_repo_root(),
    )

    tools = registry.discover_and_load()
    names = [tool.name for tool in tools]

    assert names == ["datetime_tool"]
    assert "calculator_tool" not in names


def test_auto_discover_loading_mode_includes_builtins() -> None:
    """Auto-discover mode should include built-ins not explicitly listed."""
    registry = ToolRegistry(
        {
            "loading_mode": "auto_discover",
            "tools": [
                {
                    "name": "datetime_tool",
                    "path": "universal_agentic_framework/tools/datetime",
                    "enabled": True,
                }
            ],
        },
        base_dir=_repo_root(),
    )

    tools = registry.discover_and_load()
    names = [tool.name for tool in tools]

    assert "datetime_tool" in names
    assert "calculator_tool" in names


def test_registry_discovers_profile_tools_from_extra_tools_dir(tmp_path: Path) -> None:
    profile_tools_dir = tmp_path / "tools"
    custom_tool_dir = profile_tools_dir / "patient_lookup"
    custom_tool_dir.mkdir(parents=True)
    custom_tool_dir.joinpath("tool.yaml").write_text(
        """
name: patient_lookup
type: langchain_tool
entry_point: universal_agentic_framework.tools.datetime.tool:DateTimeTool
description: Custom profile tool
        """,
        encoding="utf-8",
    )

    registry = ToolRegistry({"loading_mode": "auto_discover", "tools": []}, base_dir=_repo_root(), extra_tools_dir=profile_tools_dir)

    tools = registry.discover_and_load()

    assert any(tool.name == "patient_lookup" for tool in tools)


def test_registry_profile_tool_shadows_builtin_by_name(tmp_path: Path) -> None:
    profile_tools_dir = tmp_path / "tools"
    custom_tool_dir = profile_tools_dir / "datetime_tool"
    custom_tool_dir.mkdir(parents=True)
    custom_tool_dir.joinpath("tool.yaml").write_text(
        """
name: datetime_tool
type: langchain_tool
entry_point: universal_agentic_framework.tools.datetime.tool:DateTimeTool
description: Shadowed profile datetime tool
        """,
        encoding="utf-8",
    )

    registry = ToolRegistry({"loading_mode": "auto_discover", "tools": []}, base_dir=_repo_root(), extra_tools_dir=profile_tools_dir)
    registry.discover_and_load()

    assert registry.tools["datetime_tool"].description == "Shadowed profile datetime tool"


def test_registry_duplicate_tool_names_within_profile_fail(tmp_path: Path) -> None:
    profile_tools_dir = tmp_path / "tools"
    for dirname in ("one", "two"):
        tool_dir = profile_tools_dir / dirname
        tool_dir.mkdir(parents=True)
        tool_dir.joinpath("tool.yaml").write_text(
            """
name: duplicate_tool
type: langchain_tool
entry_point: universal_agentic_framework.tools.datetime.tool:DateTimeTool
            """,
            encoding="utf-8",
        )

    registry = ToolRegistry({"loading_mode": "auto_discover", "tools": []}, base_dir=_repo_root(), extra_tools_dir=profile_tools_dir)

    with pytest.raises(ValueError, match="Duplicate profile tool name"):
        registry.discover_and_load()


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_mcp_async_invocation_supports_save_to_rag(mock_async_client) -> None:
    """Async MCP wrapper should accept save_to_rag without NameError."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "tool": "web_search",
        "result": "ok",
    }
    mock_response.raise_for_status.return_value = None

    mock_client.post = AsyncMock(return_value=mock_response)
    mock_async_client.return_value.__aenter__.return_value = mock_client

    tool = MCPServerTool(name="test_mcp", server_url="http://localhost:9999")
    result = await tool._arun(query="test", save_to_rag=True)

    assert result == "ok"
    payload = mock_client.post.call_args[1]["json"]
    assert payload["save_to_rag"] is True


@pytest.mark.asyncio
@patch("mcp.ClientSession")
@patch("mcp.client.streamable_http.streamablehttp_client")
async def test_mcp_streamable_fetch_content_accepts_request_url(mock_http_client, mock_session_cls) -> None:
    """Streamable MCP wrapper should accept request_url kwargs from native tool-calling."""
    content_block = Mock()
    content_block.text = "ok"
    call_result = Mock()
    call_result.content = [content_block]
    call_result.isError = False

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=call_result)

    @asynccontextmanager
    async def _mock_session_ctx(read_stream, write_stream):
        yield mock_session

    @asynccontextmanager
    async def _mock_http_ctx(url):
        yield (AsyncMock(), AsyncMock(), AsyncMock())

    mock_session_cls.side_effect = _mock_session_ctx
    mock_http_client.side_effect = _mock_http_ctx

    tool = MCPServerTool(name="extract_webpage_mcp", server_url="http://localhost:8000/mcp", default_tool="fetch_content")
    result = await tool._arun(request_url="https://www.tagesschau.de/")

    assert result == "ok"
    mock_session.call_tool.assert_called_once()
    call_args = mock_session.call_tool.call_args
    assert call_args[0][0] == "fetch_content"
    assert call_args[1]["arguments"]["url"] == "https://www.tagesschau.de/"


@pytest.mark.asyncio
@patch("mcp.ClientSession")
@patch("mcp.client.streamable_http.streamablehttp_client")
async def test_mcp_streamable_fetch_content_accepts_nested_args_url(mock_http_client, mock_session_cls) -> None:
    """Streamable MCP wrapper should extract URL from nested args payloads."""
    content_block = Mock()
    content_block.text = "ok"
    call_result = Mock()
    call_result.content = [content_block]
    call_result.isError = False

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=call_result)

    @asynccontextmanager
    async def _mock_session_ctx(read_stream, write_stream):
        yield mock_session

    @asynccontextmanager
    async def _mock_http_ctx(url):
        yield (AsyncMock(), AsyncMock(), AsyncMock())

    mock_session_cls.side_effect = _mock_session_ctx
    mock_http_client.side_effect = _mock_http_ctx

    tool = MCPServerTool(name="extract_webpage_mcp", server_url="http://localhost:8000/mcp", default_tool="fetch_content")
    result = await tool._arun(query="headline", args=[{"query": "https://www.tagesschau.de/"}])

    assert result == "ok"
    mock_session.call_tool.assert_called_once()
    call_args = mock_session.call_tool.call_args
    assert call_args[0][0] == "fetch_content"
    assert call_args[1]["arguments"]["url"] == "https://www.tagesschau.de/"
