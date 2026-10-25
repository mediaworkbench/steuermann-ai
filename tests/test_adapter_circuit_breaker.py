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


def _half_open_breaker() -> AsyncCircuitBreaker:
    cb = AsyncCircuitBreaker(
        name="test",
        config=CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=1.0, half_open_max_calls=1),
    )
    cb.record_failure()
    cb._last_failure_at -= 2  # recovery window elapsed → next state read is HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN
    return cb


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_releases_its_slot() -> None:
    cb = _half_open_breaker()

    async def cancelled() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await cb.call(cancelled)

    # Without the release the probe slot stays taken and every later call is rejected.
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request() is True


def test_release_frees_probe_slot_for_verdictless_stream() -> None:
    cb = _half_open_breaker()
    assert cb.allow_request() is True
    assert cb.allow_request() is False  # single probe slot taken
    cb.release()  # e.g. upstream answered 4xx / client stopped before the response
    assert cb.allow_request() is True


def test_release_is_a_noop_when_closed() -> None:
    cb = AsyncCircuitBreaker(name="test", config=CircuitBreakerConfig())
    cb.release()
    assert cb.state == CircuitState.CLOSED
    assert cb._half_open_calls == 0
