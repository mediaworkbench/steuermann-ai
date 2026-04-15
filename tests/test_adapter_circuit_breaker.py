from __future__ import annotations

import asyncio

import pytest

from backend.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures() -> None:
    cb = AsyncCircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=10.0, half_open_max_calls=1),
    )

    async def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    with pytest.raises(RuntimeError):
        await cb.call(fail)

    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpenError):
        await cb.call(fail)


@pytest.mark.asyncio
async def test_circuit_transitions_half_open_and_closes_on_success() -> None:
    cb = AsyncCircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01, half_open_max_calls=1),
    )

    async def fail() -> None:
        raise RuntimeError("boom")

    async def succeed() -> str:
        return "ok"

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    assert cb.state == CircuitState.OPEN
    await asyncio.sleep(0.02)
    assert cb.state == CircuitState.HALF_OPEN

    result = await cb.call(succeed)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit() -> None:
    cb = AsyncCircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=0.01, half_open_max_calls=1),
    )

    async def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await cb.call(fail)

    await asyncio.sleep(0.02)
    assert cb.state == CircuitState.HALF_OPEN

    with pytest.raises(RuntimeError):
        await cb.call(fail)
    assert cb.state == CircuitState.OPEN
