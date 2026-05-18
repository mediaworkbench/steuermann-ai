"""Standard tools and registry helpers for the Steuermann."""

from typing import Any

__all__ = [
    "ToolRegistry",
    "MCPServerTool",
    "ToolManifest",
]


def __getattr__(name: str) -> Any:
    if name in {"ToolRegistry", "MCPServerTool", "ToolManifest"}:
        from .registry import ToolRegistry, MCPServerTool, ToolManifest

        mapping = {
            "ToolRegistry": ToolRegistry,
            "MCPServerTool": MCPServerTool,
            "ToolManifest": ToolManifest,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
