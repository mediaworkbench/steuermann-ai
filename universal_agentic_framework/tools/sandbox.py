"""Tool execution sandbox — security wrapper for tool execution.

Provides resource limits, timeout enforcement, and permission checking
to prevent tools from performing dangerous operations.
"""

import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger()


class Permission(str, Enum):
    """Tool permission categories."""
    SYSTEM_TIME = "system:time"
    SYSTEM_COMPUTE = "system:compute"
    FILESYSTEM_READ = "filesystem:read"
    FILESYSTEM_WRITE = "filesystem:write"
    NETWORK_HTTP = "network:http"
    NETWORK_INTERNAL = "network:internal"


@dataclass
class SandboxPolicy:
    """Security policy for tool execution."""

    # Allowed permissions
    allowed_permissions: Set[str] = field(default_factory=lambda: {
        Permission.SYSTEM_TIME.value,
        Permission.SYSTEM_COMPUTE.value,
        Permission.FILESYSTEM_READ.value,
        Permission.NETWORK_HTTP.value,
        Permission.NETWORK_INTERNAL.value,
    })

    # Maximum execution time per tool call (seconds)
    max_execution_time: float = 30.0

    # Maximum output size (bytes)
    max_output_size: int = 1_048_576  # 1 MB

    # Maximum concurrent tool executions
    max_concurrent: int = 5

    # Filesystem restrictions
    allowed_paths: List[str] = field(default_factory=list)  # Empty = no restriction
    denied_paths: List[str] = field(default_factory=lambda: [
        "/etc/shadow", "/etc/passwd", "/proc", "/sys",
        "~/.ssh", "~/.gnupg", "~/.aws", "~/.kube",
    ])

    # Network restrictions
    denied_hosts: List[str] = field(default_factory=lambda: [
        "169.254.169.254",  # Cloud metadata
        "metadata.google.internal",
    ])

    def has_permission(self, permission: str) -> bool:
        """Check if a permission is allowed by this policy."""
        return permission in self.allowed_permissions

    def is_path_allowed(self, path: str) -> bool:
        """Check if a filesystem path is allowed."""
        resolved = os.path.realpath(os.path.expanduser(path))
        # Check denied paths first
        for denied in self.denied_paths:
            denied_resolved = os.path.realpath(os.path.expanduser(denied))
            if resolved.startswith(denied_resolved):
                return False
        # If allowed_paths is set, path must be under one of them
        if self.allowed_paths:
            for allowed in self.allowed_paths:
                allowed_resolved = os.path.realpath(os.path.expanduser(allowed))
                if resolved.startswith(allowed_resolved):
                    return True
            return False
        return True


@dataclass
class SandboxResult:
    """Result of a sandboxed tool execution."""
    success: bool
    output: str
    execution_time_ms: float = 0.0
    timed_out: bool = False
    permission_denied: bool = False
    error: Optional[str] = None
    tool_name: str = ""


class ToolSandbox:
    """Sandboxed execution environment for tools.
    
    Enforces:
    - Permission checks before execution
    - Timeout limits via ThreadPoolExecutor
    - Output size limits
    - Filesystem path restrictions
    - Concurrent execution limits
    """

    def __init__(self, policy: Optional[SandboxPolicy] = None):
        self.policy = policy or SandboxPolicy()
        self._executor = ThreadPoolExecutor(max_workers=self.policy.max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()

    def check_permissions(self, tool_name: str, required_permissions: List[str]) -> Optional[str]:
        """Check if tool has all required permissions. Returns error message or None."""
        for perm in required_permissions:
            if not self.policy.has_permission(perm):
                msg = f"Permission denied: '{perm}' not allowed for tool '{tool_name}'"
                logger.warning("Sandbox permission denied", tool=tool_name, permission=perm)
                return msg
        return None

    def execute(
        self,
        tool_name: str,
        func: Callable[..., str],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        required_permissions: Optional[List[str]] = None,
        timeout_override: Optional[float] = None,
    ) -> SandboxResult:
        """Execute a tool function within the sandbox.
        
        Args:
            tool_name: Name of the tool being executed
            func: The callable to execute
            args: Positional arguments
            kwargs: Keyword arguments
            required_permissions: List of permission strings required
            timeout_override: Override the default timeout
            
        Returns:
            SandboxResult with execution outcome
        """
        kwargs = kwargs or {}
        required_permissions = required_permissions or []
        timeout = timeout_override or self.policy.max_execution_time

        # Check permissions
        perm_error = self.check_permissions(tool_name, required_permissions)
        if perm_error:
            return SandboxResult(
                success=False,
                output="",
                permission_denied=True,
                error=perm_error,
                tool_name=tool_name,
            )

        # Check concurrent limit
        with self._lock:
            if self._active_count >= self.policy.max_concurrent:
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"Concurrent execution limit reached ({self.policy.max_concurrent})",
                    tool_name=tool_name,
                )
            self._active_count += 1

        start = time.monotonic()
        try:
            # Execute with timeout
            future = self._executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
            except FuturesTimeoutError:
                future.cancel()
                elapsed = (time.monotonic() - start) * 1000
                logger.warning("Sandbox timeout", tool=tool_name, timeout=timeout, elapsed_ms=elapsed)
                return SandboxResult(
                    success=False,
                    output="",
                    execution_time_ms=elapsed,
                    timed_out=True,
                    error=f"Tool execution timed out after {timeout}s",
                    tool_name=tool_name,
                )

            elapsed = (time.monotonic() - start) * 1000
            output = str(result) if result is not None else ""

            # Enforce output size limit
            if len(output.encode("utf-8", errors="replace")) > self.policy.max_output_size:
                output = output[: self.policy.max_output_size] + "\n... [output truncated]"
                logger.warning("Sandbox output truncated", tool=tool_name, max_size=self.policy.max_output_size)

            logger.info(
                "Sandbox execution complete",
                tool=tool_name,
                execution_time_ms=round(elapsed, 2),
                output_length=len(output),
            )

            return SandboxResult(
                success=True,
                output=output,
                execution_time_ms=elapsed,
                tool_name=tool_name,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Sandbox execution error", tool=tool_name, error=str(e), elapsed_ms=elapsed)
            return SandboxResult(
                success=False,
                output="",
                execution_time_ms=elapsed,
                error=str(e),
                tool_name=tool_name,
            )
        finally:
            with self._lock:
                self._active_count -= 1

    def shutdown(self):
        """Shut down the executor."""
        self._executor.shutdown(wait=False)


# ── Module-level default sandbox ─────────────────────────────────────

_default_sandbox: Optional[ToolSandbox] = None


def get_sandbox(policy: Optional[SandboxPolicy] = None) -> ToolSandbox:
    """Get or create the default sandbox instance."""
    global _default_sandbox
    if _default_sandbox is None or policy is not None:
        _default_sandbox = ToolSandbox(policy=policy)
    return _default_sandbox


def reset_sandbox():
    """Reset the default sandbox (useful for testing)."""
    global _default_sandbox
    if _default_sandbox is not None:
        _default_sandbox.shutdown()
    _default_sandbox = None
