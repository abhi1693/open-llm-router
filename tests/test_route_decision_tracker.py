from __future__ import annotations

import pytest

from open_llm_router.route_decision_tracker import RouteDecisionTracker


def test_route_decision_tracker_records_secondary_outcomes() -> None:
    tracker = RouteDecisionTracker()

    tracker.observe_route_decision(
        request_id="req-1",
        signals={"secondary_classifier_used": True},
    )
    tracker.observe_proxy_terminal(request_id="req-1", outcome="success")

    tracker.observe_route_decision(
        request_id="req-2",
        signals={"secondary_classifier_used": True},
    )
    tracker.observe_proxy_terminal(request_id="req-2", outcome="error")

    snapshot = tracker.snapshot
    assert snapshot.secondary_total == 2
    assert snapshot.secondary_success == 1
    assert snapshot.secondary_non_success == 1
    assert snapshot.secondary_success_rate == pytest.approx(0.5)


def test_route_decision_tracker_ignores_missing_or_non_secondary_context() -> None:
    tracker = RouteDecisionTracker()

    tracker.observe_proxy_terminal(request_id="req-unknown", outcome="success")
    tracker.observe_route_decision(
        request_id="req-1",
        signals={"secondary_classifier_used": False},
    )
    tracker.observe_proxy_terminal(request_id="req-1", outcome="success")

    snapshot = tracker.snapshot
    assert snapshot.secondary_total == 0
    assert snapshot.secondary_success == 0
    assert snapshot.secondary_non_success == 0
    assert snapshot.secondary_success_rate is None


def test_route_decision_tracker_bounds_pending_request_capacity() -> None:
    tracker = RouteDecisionTracker(pending_capacity=2)

    tracker.observe_route_decision(
        request_id="req-1",
        signals={"secondary_classifier_used": True},
    )
    tracker.observe_route_decision(
        request_id="req-2",
        signals={"secondary_classifier_used": True},
    )
    tracker.observe_route_decision(
        request_id="req-3",
        signals={"secondary_classifier_used": True},
    )

    # req-1 should be evicted once capacity is exceeded.
    tracker.observe_proxy_terminal(request_id="req-1", outcome="success")
    tracker.observe_proxy_terminal(request_id="req-2", outcome="success")
    tracker.observe_proxy_terminal(request_id="req-3", outcome="error")

    snapshot = tracker.snapshot
    assert snapshot.secondary_total == 2
    assert snapshot.secondary_success == 1
    assert snapshot.secondary_non_success == 1
