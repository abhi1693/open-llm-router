from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from open_llm_router.runtime.proxy_metrics import ProxyMetricsAccumulator
from open_llm_router.runtime.route_decision_tracker import RouteDecisionTracker

if TYPE_CHECKING:
    from open_llm_router.runtime.live_metrics import LiveMetricsStore


@dataclass(slots=True)
class EventContext:
    event_name: str
    model: str
    provider: str | None
    account: str | None
    target_key: tuple[str, str]

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> EventContext:
        event_name = str(event.get("event") or "")
        model = str(event.get("model") or "").strip()
        provider = str(event.get("provider") or "").strip().lower() or None
        account = str(event.get("account") or "").strip().lower() or None
        target_key = (provider or "unknown", account or "unknown")
        return cls(
            event_name=event_name,
            model=model,
            provider=provider,
            account=account,
            target_key=target_key,
        )


class LiveMetricsEventDispatcher:
    def __init__(
        self,
        *,
        store: LiveMetricsStore,
        proxy_metrics: ProxyMetricsAccumulator,
        route_decision_tracker: RouteDecisionTracker,
    ) -> None:
        self._store = store
        self._proxy_metrics = proxy_metrics
        self._route_decision_tracker = route_decision_tracker
        self._model_handlers: dict[
            str,
            Callable[[dict[str, Any], EventContext], Awaitable[None]],
        ] = {
            "proxy_upstream_connected": self._handle_proxy_upstream_connected,
            "proxy_response": self._handle_proxy_response,
            "proxy_retry": self._handle_proxy_retry,
            "proxy_request_error": self._handle_proxy_request_error,
        }

    async def process(self, event: dict[str, Any]) -> None:
        context = EventContext.from_event(event)

        if context.event_name == "route_decision":
            await self._handle_route_decision(event)
            return

        if context.event_name == "proxy_terminal":
            self._handle_proxy_terminal(event)
            return

        if not context.model:
            return

        handler = self._model_handlers.get(context.event_name)
        if handler is not None:
            await handler(event, context)

    async def _handle_route_decision(self, event: dict[str, Any]) -> None:
        selected_model = str(event.get("selected_model") or "").strip()
        if selected_model:
            await self._store.record_route_decision(
                model=selected_model,
                task=str(event.get("task") or ""),
                complexity=str(event.get("complexity") or ""),
            )
        request_id = str(event.get("request_id") or "").strip()
        signals = event.get("signals")
        self._route_decision_tracker.observe_route_decision(
            request_id=request_id,
            signals=signals if isinstance(signals, dict) else None,
        )

    def _handle_proxy_terminal(self, event: dict[str, Any]) -> None:
        request_id = str(event.get("request_id") or "").strip()
        outcome = str(event.get("outcome") or "")
        self._route_decision_tracker.observe_proxy_terminal(
            request_id=request_id,
            outcome=outcome,
        )

    async def _handle_proxy_upstream_connected(
        self,
        event: dict[str, Any],
        context: EventContext,
    ) -> None:
        connect_ms = event.get("connect_ms")
        if not isinstance(connect_ms, (int, float)):
            return
        connect_value = max(0.0, float(connect_ms))
        await self._store.record_connect(
            model=context.model,
            connect_ms=connect_value,
            provider=context.provider,
            account=context.account,
        )
        connect_target_key = (
            context.provider or "unknown",
            context.account or "unknown",
            context.model,
        )
        self._proxy_metrics.record_connect(connect_target_key, connect_value)

    async def _handle_proxy_response(
        self,
        event: dict[str, Any],
        context: EventContext,
    ) -> None:
        status = event.get("status")
        if not isinstance(status, int):
            return
        self._proxy_metrics.record_response(status)
        request_latency_ms = event.get("request_latency_ms")
        latency_value = (
            float(request_latency_ms)
            if isinstance(request_latency_ms, (int, float))
            else None
        )
        await self._store.record_response(
            model=context.model,
            status=status,
            request_latency_ms=latency_value,
            provider=context.provider,
            account=context.account,
        )

    async def _handle_proxy_retry(
        self,
        _: dict[str, Any],
        context: EventContext,
    ) -> None:
        self._proxy_metrics.record_retry(context.target_key)

    async def _handle_proxy_request_error(
        self,
        event: dict[str, Any],
        context: EventContext,
    ) -> None:
        error_type = str(event.get("error_type") or "").strip() or "unknown"
        attempt_latency_ms = event.get("attempt_latency_ms")
        attempt_latency_value = (
            float(attempt_latency_ms)
            if isinstance(attempt_latency_ms, (int, float))
            else None
        )
        self._proxy_metrics.record_error(
            target_key=context.target_key,
            error_type=error_type,
            is_timeout=bool(event.get("is_timeout")),
            attempt_latency_ms=attempt_latency_value,
        )
        await self._store.record_error(
            model=context.model,
            error_type=error_type,
            provider=context.provider,
            account=context.account,
        )
