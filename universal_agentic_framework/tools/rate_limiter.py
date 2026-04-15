"""Tool rate limiter — prevents excessive tool invocations.

Implements a sliding window rate limiter with per-tool and global limits.
Thread-safe for concurrent access.
"""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    # Global limit: max tool calls across all tools per window
    global_max_calls: int = 100
    global_window_seconds: float = 60.0

    # Per-tool limit: max calls per individual tool per window
    per_tool_max_calls: int = 20
    per_tool_window_seconds: float = 60.0

    # Per-user limit: max tool calls per user per window
    per_user_max_calls: int = 30
    per_user_window_seconds: float = 60.0

    # Burst allowance: allow short bursts above the limit
    burst_multiplier: float = 1.5


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    reason: Optional[str] = None
    retry_after_seconds: Optional[float] = None
    remaining_calls: int = 0


class _SlidingWindow:
    """Thread-safe sliding window counter."""

    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def check(self) -> Tuple[bool, int, Optional[float]]:
        """Check if a call is allowed.
        
        Returns:
            (allowed, remaining_calls, retry_after_seconds)
        """
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if len(self._timestamps) >= self.max_calls:
                # Earliest timestamp + window = when a slot opens
                retry_after = self._timestamps[0] + self.window_seconds - now
                return False, 0, max(0.0, retry_after)
            return True, self.max_calls - len(self._timestamps), None

    def record(self) -> None:
        """Record a successful call."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            self._timestamps.append(now)

    def count(self) -> int:
        """Return current call count in window."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return len(self._timestamps)

    def reset(self) -> None:
        """Clear all recorded calls."""
        with self._lock:
            self._timestamps.clear()


class ToolRateLimiter:
    """Rate limiter for tool executions.
    
    Enforces:
    - Global call limit across all tools
    - Per-tool call limit
    - Per-user call limit
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # Global window
        self._global = _SlidingWindow(
            self.config.global_max_calls, self.config.global_window_seconds
        )

        # Per-tool windows: tool_name -> _SlidingWindow
        self._per_tool: Dict[str, _SlidingWindow] = defaultdict(
            lambda: _SlidingWindow(
                self.config.per_tool_max_calls, self.config.per_tool_window_seconds
            )
        )

        # Per-user windows: user_id -> _SlidingWindow
        self._per_user: Dict[str, _SlidingWindow] = defaultdict(
            lambda: _SlidingWindow(
                self.config.per_user_max_calls, self.config.per_user_window_seconds
            )
        )

    def check(self, tool_name: str, user_id: Optional[str] = None) -> RateLimitResult:
        """Check if a tool call is allowed under rate limits.
        
        Args:
            tool_name: The tool being invoked
            user_id: Optional user identifier
            
        Returns:
            RateLimitResult indicating whether the call is allowed
        """
        # Check global limit
        allowed, remaining, retry_after = self._global.check()
        if not allowed:
            logger.warning(
                "Rate limit exceeded (global)",
                tool=tool_name,
                user_id=user_id,
                retry_after=retry_after,
            )
            return RateLimitResult(
                allowed=False,
                reason=f"Global rate limit exceeded ({self.config.global_max_calls} calls/{self.config.global_window_seconds}s)",
                retry_after_seconds=retry_after,
                remaining_calls=0,
            )

        # Check per-tool limit
        allowed, remaining, retry_after = self._per_tool[tool_name].check()
        if not allowed:
            logger.warning(
                "Rate limit exceeded (per-tool)",
                tool=tool_name,
                user_id=user_id,
                retry_after=retry_after,
            )
            return RateLimitResult(
                allowed=False,
                reason=f"Per-tool rate limit exceeded for '{tool_name}' ({self.config.per_tool_max_calls} calls/{self.config.per_tool_window_seconds}s)",
                retry_after_seconds=retry_after,
                remaining_calls=0,
            )

        # Check per-user limit
        if user_id:
            allowed, remaining_user, retry_after = self._per_user[user_id].check()
            if not allowed:
                logger.warning(
                    "Rate limit exceeded (per-user)",
                    tool=tool_name,
                    user_id=user_id,
                    retry_after=retry_after,
                )
                return RateLimitResult(
                    allowed=False,
                    reason=f"Per-user rate limit exceeded ({self.config.per_user_max_calls} calls/{self.config.per_user_window_seconds}s)",
                    retry_after_seconds=retry_after,
                    remaining_calls=0,
                )
            remaining = min(remaining, remaining_user)

        return RateLimitResult(allowed=True, remaining_calls=remaining)

    def record(self, tool_name: str, user_id: Optional[str] = None) -> None:
        """Record a successful tool call."""
        self._global.record()
        self._per_tool[tool_name].record()
        if user_id:
            self._per_user[user_id].record()
        logger.info(
            "Tool call recorded",
            tool=tool_name,
            user_id=user_id,
            global_count=self._global.count(),
            tool_count=self._per_tool[tool_name].count(),
        )

    def get_stats(self) -> Dict[str, int]:
        """Return current rate limit statistics."""
        stats = {
            "global_calls": self._global.count(),
            "global_remaining": max(0, self.config.global_max_calls - self._global.count()),
        }
        for name, window in self._per_tool.items():
            stats[f"tool_{name}_calls"] = window.count()
        for uid, window in self._per_user.items():
            stats[f"user_{uid}_calls"] = window.count()
        return stats

    def reset(self):
        """Reset all windows."""
        self._global.reset()
        for w in self._per_tool.values():
            w.reset()
        for w in self._per_user.values():
            w.reset()


# ── Module-level default rate limiter ────────────────────────────────

_default_limiter: Optional[ToolRateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> ToolRateLimiter:
    """Get or create the default rate limiter instance."""
    global _default_limiter
    if _default_limiter is None or config is not None:
        _default_limiter = ToolRateLimiter(config=config)
    return _default_limiter


def reset_rate_limiter():
    """Reset the default rate limiter (useful for testing)."""
    global _default_limiter
    _default_limiter = None
