"""Standard tools and registry helpers for the Steuermann."""

from typing import Any

__all__ = [
    "ToolRegistry",
    "MCPServerTool",
    "ToolManifest",
    "ToolSandbox",
    "SandboxPolicy",
    "SandboxResult",
    "Permission",
    "get_sandbox",
    "reset_sandbox",
    "ToolRateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "get_rate_limiter",
    "reset_rate_limiter",
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

    if name in {
        "ToolSandbox",
        "SandboxPolicy",
        "SandboxResult",
        "Permission",
        "get_sandbox",
        "reset_sandbox",
    }:
        from .sandbox import (
            ToolSandbox,
            SandboxPolicy,
            SandboxResult,
            Permission,
            get_sandbox,
            reset_sandbox,
        )

        mapping = {
            "ToolSandbox": ToolSandbox,
            "SandboxPolicy": SandboxPolicy,
            "SandboxResult": SandboxResult,
            "Permission": Permission,
            "get_sandbox": get_sandbox,
            "reset_sandbox": reset_sandbox,
        }
        return mapping[name]

    if name in {
        "ToolRateLimiter",
        "RateLimitConfig",
        "RateLimitResult",
        "get_rate_limiter",
        "reset_rate_limiter",
    }:
        from .rate_limiter import (
            ToolRateLimiter,
            RateLimitConfig,
            RateLimitResult,
            get_rate_limiter,
            reset_rate_limiter,
        )

        mapping = {
            "ToolRateLimiter": ToolRateLimiter,
            "RateLimitConfig": RateLimitConfig,
            "RateLimitResult": RateLimitResult,
            "get_rate_limiter": get_rate_limiter,
            "reset_rate_limiter": reset_rate_limiter,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
