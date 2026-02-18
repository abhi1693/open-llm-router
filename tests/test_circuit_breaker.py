from __future__ import annotations

from open_llm_router.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)


def test_circuit_breaker_opens_after_threshold_and_recovers_to_half_open() -> None:
    breakers = CircuitBreakerRegistry(
        CircuitBreakerConfig(
            enabled=True,
            failure_threshold=2,
            recovery_timeout_seconds=0.0,
            half_open_max_requests=1,
        )
    )
    key = "acct-a:openai"

    assert breakers.allow_request(key) is True
    breakers.on_failure(key)
    assert breakers.allow_request(key) is True
    breakers.on_failure(key)

    # Once open, and with recovery timeout elapsed (0.0), the next call is a half-open probe.
    assert breakers.allow_request(key) is True
    snapshot = breakers.snapshot(key)
    assert snapshot["state"] == "half_open"
    assert snapshot["half_open_in_flight"] == 1


def test_circuit_breaker_half_open_success_closes_breaker() -> None:
    breakers = CircuitBreakerRegistry(
        CircuitBreakerConfig(
            enabled=True,
            failure_threshold=1,
            recovery_timeout_seconds=0.0,
            half_open_max_requests=1,
        )
    )
    key = "acct-b:openai"

    breakers.on_failure(key)
    assert (
        breakers.allow_request(key) is True
    )  # transitions to half-open and allows probe
    breakers.on_success(key)

    snapshot = breakers.snapshot(key)
    assert snapshot["state"] == "closed"
    assert snapshot["failure_count"] == 0
    assert breakers.allow_request(key) is True


def test_circuit_breaker_half_open_failure_reopens_breaker() -> None:
    breakers = CircuitBreakerRegistry(
        CircuitBreakerConfig(
            enabled=True,
            failure_threshold=1,
            recovery_timeout_seconds=0.0,
            half_open_max_requests=1,
        )
    )
    key = "acct-c:openai"

    breakers.on_failure(key)
    assert breakers.allow_request(key) is True
    breakers.on_failure(key)

    snapshot = breakers.snapshot(key)
    assert snapshot["state"] == "open"
