from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreakerConfig:
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_requests: int = 1


@dataclass(slots=True)
class _Circuit:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    opened_at_epoch: float = 0.0
    half_open_in_flight: int = 0


class CircuitBreakerRegistry:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._circuits: dict[str, _Circuit] = {}

    def allow_request(self, key: str) -> bool:
        if not self._config.enabled:
            return True

        circuit = self._circuits.setdefault(key, _Circuit())
        now = time.time()

        if circuit.state == CircuitState.OPEN:
            if now - circuit.opened_at_epoch >= self._config.recovery_timeout_seconds:
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_in_flight = 0
            else:
                return False

        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.half_open_in_flight >= self._config.half_open_max_requests:
                return False
            circuit.half_open_in_flight += 1
            return True

        return True

    def on_success(self, key: str) -> None:
        if not self._config.enabled:
            return

        circuit = self._circuits.setdefault(key, _Circuit())
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.half_open_in_flight = max(0, circuit.half_open_in_flight - 1)
            circuit.state = CircuitState.CLOSED
            circuit.failure_count = 0
            circuit.opened_at_epoch = 0.0
            return

        circuit.failure_count = 0
        circuit.state = CircuitState.CLOSED

    def on_failure(self, key: str) -> None:
        if not self._config.enabled:
            return

        circuit = self._circuits.setdefault(key, _Circuit())
        now = time.time()
        if circuit.state == CircuitState.HALF_OPEN:
            circuit.half_open_in_flight = max(0, circuit.half_open_in_flight - 1)
            circuit.state = CircuitState.OPEN
            circuit.opened_at_epoch = now
            circuit.failure_count = self._config.failure_threshold
            return

        circuit.failure_count += 1
        if circuit.failure_count >= self._config.failure_threshold:
            circuit.state = CircuitState.OPEN
            circuit.opened_at_epoch = now

    def snapshot(self, key: str) -> dict[str, int | float | str]:
        circuit = self._circuits.get(key) or _Circuit()
        return {
            "state": circuit.state.value,
            "failure_count": circuit.failure_count,
            "opened_at_epoch": round(circuit.opened_at_epoch, 3),
            "half_open_in_flight": circuit.half_open_in_flight,
        }
