from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Awaitable, Callable, TypeVar


T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


def config_from_env(prefix: str, defaults: CircuitBreakerConfig) -> CircuitBreakerConfig:
    threshold = int(os.getenv(f"{prefix}_FAILURE_THRESHOLD", str(defaults.failure_threshold)))
    timeout = float(os.getenv(f"{prefix}_RECOVERY_TIMEOUT_SECONDS", str(defaults.recovery_timeout_seconds)))
    half_open_calls = int(os.getenv(f"{prefix}_HALF_OPEN_MAX_CALLS", str(defaults.half_open_max_calls)))
    return CircuitBreakerConfig(
        failure_threshold=max(1, threshold),
        recovery_timeout_seconds=max(1.0, timeout),
        half_open_max_calls=max(1, half_open_calls),
    )


class AsyncCircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_at: float | None = None
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and self._last_failure_at is not None:
                if (time.time() - self._last_failure_at) >= self.config.recovery_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
            return self._state

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
            "last_failure_at": self._last_failure_at,
        }

    def _allow_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            return False
        with self._lock:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def _record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._last_failure_at = None
            else:
                self._failure_count = 0

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_at = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._half_open_calls = 0
                return
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        if not self._allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. Retry after {self.config.recovery_timeout_seconds}s"
            )
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise
