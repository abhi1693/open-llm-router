from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ClassifierCalibrationSnapshot:
    secondary_total: int
    secondary_success: int
    secondary_non_success: int
    secondary_success_rate: float | None


class RouteDecisionTracker:
    def __init__(self, *, pending_capacity: int = 20000):
        self._pending_context: dict[str, bool] = {}
        self._pending_order: deque[str] = deque()
        self._pending_capacity = max(1, int(pending_capacity))
        self._secondary_total = 0
        self._secondary_success = 0
        self._secondary_non_success = 0

    def observe_route_decision(
        self,
        *,
        request_id: str,
        signals: dict[str, Any] | None,
    ) -> None:
        if not request_id or not isinstance(signals, dict):
            return
        secondary_used = bool(signals.get("secondary_classifier_used"))
        self._pending_context[request_id] = secondary_used
        self._pending_order.append(request_id)
        while len(self._pending_order) > self._pending_capacity:
            stale_request_id = self._pending_order.popleft()
            self._pending_context.pop(stale_request_id, None)

    def observe_proxy_terminal(self, *, request_id: str, outcome: str) -> None:
        if not request_id:
            return
        secondary_used = self._pending_context.pop(request_id, None)
        if secondary_used is None or not secondary_used:
            return
        self._secondary_total += 1
        if outcome.strip().lower() == "success":
            self._secondary_success += 1
        else:
            self._secondary_non_success += 1

    @property
    def snapshot(self) -> ClassifierCalibrationSnapshot:
        total = int(self._secondary_total)
        success = int(self._secondary_success)
        non_success = int(self._secondary_non_success)
        success_rate = (float(success) / float(total)) if total > 0 else None
        return ClassifierCalibrationSnapshot(
            secondary_total=total,
            secondary_success=success,
            secondary_non_success=non_success,
            secondary_success_rate=success_rate,
        )
