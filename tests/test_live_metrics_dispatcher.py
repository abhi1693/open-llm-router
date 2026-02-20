from __future__ import annotations

import asyncio

from open_llm_router.runtime.live_metrics import InMemoryLiveMetricsStore
from open_llm_router.runtime.live_metrics_dispatcher import (
    EventContext,
    LiveMetricsEventDispatcher,
)
from open_llm_router.runtime.proxy_metrics import ProxyMetricsAccumulator
from open_llm_router.runtime.route_decision_tracker import RouteDecisionTracker


def _build_dispatcher() -> tuple[
    InMemoryLiveMetricsStore,
    ProxyMetricsAccumulator,
    RouteDecisionTracker,
    LiveMetricsEventDispatcher,
]:
    store = InMemoryLiveMetricsStore(alpha=1.0)
    proxy_metrics = ProxyMetricsAccumulator(
        connect_latency_window_size=16,
        connect_latency_alert_threshold_ms=50.0,
    )
    route_decision_tracker = RouteDecisionTracker()
    dispatcher = LiveMetricsEventDispatcher(
        store=store,
        proxy_metrics=proxy_metrics,
        route_decision_tracker=route_decision_tracker,
    )
    return store, proxy_metrics, route_decision_tracker, dispatcher


def test_event_context_from_event_normalizes_dimensions() -> None:
    context = EventContext.from_event(
        {
            "event": "proxy_retry",
            "model": " m1 ",
            "provider": " OpenAI ",
            "account": " Acct-A ",
        }
    )

    assert context.event_name == "proxy_retry"
    assert context.model == "m1"
    assert context.provider == "openai"
    assert context.account == "acct-a"
    assert context.target_key == ("openai", "acct-a")


def test_live_metrics_event_dispatcher_processes_route_calibration_events() -> None:
    async def _run() -> None:
        store, _, tracker, dispatcher = _build_dispatcher()

        await dispatcher.process(
            {
                "event": "route_decision",
                "selected_model": "m1",
                "request_id": "req-1",
                "task": "coding",
                "complexity": "high",
                "signals": {"secondary_classifier_used": True},
            }
        )
        await dispatcher.process(
            {
                "event": "proxy_terminal",
                "request_id": "req-1",
                "outcome": "success",
            }
        )

        snapshot = await store.snapshot_all()
        assert snapshot["m1"].route_decisions == 1
        calibration = tracker.snapshot
        assert calibration.secondary_total == 1
        assert calibration.secondary_success == 1
        assert calibration.secondary_non_success == 0

    asyncio.run(_run())


def test_live_metrics_event_dispatcher_ignores_model_scoped_events_without_model() -> (
    None
):
    async def _run() -> None:
        store, proxy_metrics, _, dispatcher = _build_dispatcher()

        await dispatcher.process({"event": "proxy_response", "status": 200})
        await dispatcher.process(
            {"event": "proxy_request_error", "error_type": "ConnectTimeout"}
        )
        await dispatcher.process({"event": "proxy_retry", "provider": "openai"})

        snapshot = await store.snapshot_all()
        assert snapshot == {}
        assert proxy_metrics.proxy_responses_by_status_class == {}
        assert proxy_metrics.proxy_errors_by_type == {}
        assert proxy_metrics.proxy_retries_total == 0

    asyncio.run(_run())


def test_live_metrics_event_dispatcher_processes_model_scoped_handlers() -> None:
    async def _run() -> None:
        store, proxy_metrics, _, dispatcher = _build_dispatcher()

        await dispatcher.process(
            {
                "event": "proxy_upstream_connected",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "connect_ms": 75.0,
            }
        )
        await dispatcher.process(
            {
                "event": "proxy_response",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "status": 503,
                "request_latency_ms": 10.0,
            }
        )
        await dispatcher.process(
            {
                "event": "proxy_retry",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
            }
        )
        await dispatcher.process(
            {
                "event": "proxy_request_error",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "error_type": "ConnectTimeout",
                "is_timeout": True,
                "attempt_latency_ms": 15.5,
            }
        )

        snapshot = await store.snapshot_all()
        assert snapshot["m1"].samples == 3
        assert snapshot["m1"].responses == 1
        assert snapshot["m1"].errors == 2

        assert proxy_metrics.proxy_retries_total == 1
        assert proxy_metrics.proxy_retries_by_target == {("openai", "acct-a"): 1}
        assert proxy_metrics.proxy_timeouts_total == 1
        assert proxy_metrics.proxy_timeouts_by_target == {("openai", "acct-a"): 1}
        assert proxy_metrics.proxy_responses_by_status_class == {"5xx": 1}
        assert proxy_metrics.proxy_errors_by_type == {"ConnectTimeout": 1}
        assert proxy_metrics.proxy_attempt_latency_count == 1

    asyncio.run(_run())
