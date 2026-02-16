from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ModelMetricsSnapshot:
    model: str
    samples: int
    route_decisions: int
    responses: int
    errors: int
    ewma_connect_ms: float | None
    ewma_request_latency_ms: float | None
    ewma_failure_rate: float | None
    last_updated_epoch: float | None


class LiveMetricsStore(Protocol):
    async def record_route_decision(self, model: str, task: str, complexity: str) -> None: ...

    async def record_connect(self, model: str, connect_ms: float) -> None: ...

    async def record_response(
        self,
        model: str,
        status: int,
        request_latency_ms: float | None,
    ) -> None: ...

    async def record_error(self, model: str, error_type: str | None = None) -> None: ...

    async def snapshot_all(self) -> dict[str, ModelMetricsSnapshot]: ...


@dataclass(slots=True)
class _ModelMetricsState:
    model: str
    samples: int = 0
    route_decisions: int = 0
    responses: int = 0
    errors: int = 0
    ewma_connect_ms: float | None = None
    ewma_request_latency_ms: float | None = None
    ewma_failure_rate: float | None = None
    last_updated_epoch: float | None = None

    def to_snapshot(self) -> ModelMetricsSnapshot:
        return ModelMetricsSnapshot(
            model=self.model,
            samples=self.samples,
            route_decisions=self.route_decisions,
            responses=self.responses,
            errors=self.errors,
            ewma_connect_ms=self.ewma_connect_ms,
            ewma_request_latency_ms=self.ewma_request_latency_ms,
            ewma_failure_rate=self.ewma_failure_rate,
            last_updated_epoch=self.last_updated_epoch,
        )


def _ewma(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return (alpha * value) + ((1.0 - alpha) * previous)


def _is_failure_status(status: int) -> bool:
    return status == 429 or status >= 500


class InMemoryLiveMetricsStore:
    def __init__(self, *, alpha: float = 0.2) -> None:
        self._alpha = max(0.01, min(1.0, alpha))
        self._lock = asyncio.Lock()
        self._models: dict[str, _ModelMetricsState] = {}

    async def record_route_decision(self, model: str, task: str, complexity: str) -> None:
        del task, complexity
        key = model.strip()
        if not key:
            return
        now = time.time()
        async with self._lock:
            state = self._models.setdefault(key, _ModelMetricsState(model=key))
            state.route_decisions += 1
            state.last_updated_epoch = now

    async def record_connect(self, model: str, connect_ms: float) -> None:
        key = model.strip()
        if not key:
            return
        now = time.time()
        value = max(0.0, float(connect_ms))
        async with self._lock:
            state = self._models.setdefault(key, _ModelMetricsState(model=key))
            state.samples += 1
            state.ewma_connect_ms = _ewma(state.ewma_connect_ms, value, self._alpha)
            state.last_updated_epoch = now

    async def record_response(
        self,
        model: str,
        status: int,
        request_latency_ms: float | None,
    ) -> None:
        key = model.strip()
        if not key:
            return
        now = time.time()
        async with self._lock:
            state = self._models.setdefault(key, _ModelMetricsState(model=key))
            state.responses += 1
            state.samples += 1
            failure_flag = 1.0 if _is_failure_status(int(status)) else 0.0
            state.ewma_failure_rate = _ewma(state.ewma_failure_rate, failure_flag, self._alpha)
            if failure_flag > 0:
                state.errors += 1
            if request_latency_ms is not None:
                latency = max(0.0, float(request_latency_ms))
                state.ewma_request_latency_ms = _ewma(
                    state.ewma_request_latency_ms,
                    latency,
                    self._alpha,
                )
            state.last_updated_epoch = now

    async def record_error(self, model: str, error_type: str | None = None) -> None:
        del error_type
        key = model.strip()
        if not key:
            return
        now = time.time()
        async with self._lock:
            state = self._models.setdefault(key, _ModelMetricsState(model=key))
            state.errors += 1
            state.samples += 1
            state.ewma_failure_rate = _ewma(state.ewma_failure_rate, 1.0, self._alpha)
            state.last_updated_epoch = now

    async def snapshot_all(self) -> dict[str, ModelMetricsSnapshot]:
        async with self._lock:
            return {model: state.to_snapshot() for model, state in self._models.items()}


class RedisLiveMetricsStore:
    def __init__(
        self,
        redis_client: Any,
        *,
        alpha: float = 0.2,
        key_prefix: str = "router:live_metrics:model:",
        key_ttl_seconds: int = 14 * 24 * 3600,
    ) -> None:
        self._redis = redis_client
        self._alpha = max(0.01, min(1.0, alpha))
        self._key_prefix = key_prefix
        self._key_ttl_seconds = max(60, int(key_ttl_seconds))

    def _key_for_model(self, model: str) -> str:
        encoded = base64.urlsafe_b64encode(model.encode("utf-8")).decode("ascii").rstrip("=")
        return f"{self._key_prefix}{encoded}"

    @staticmethod
    def _parse_state(raw: dict[str, str]) -> _ModelMetricsState | None:
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            return None

        def _to_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except Exception:
                return default

        def _to_float(value: Any) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except Exception:
                return None

        return _ModelMetricsState(
            model=model,
            samples=_to_int(raw.get("samples"), 0),
            route_decisions=_to_int(raw.get("route_decisions"), 0),
            responses=_to_int(raw.get("responses"), 0),
            errors=_to_int(raw.get("errors"), 0),
            ewma_connect_ms=_to_float(raw.get("ewma_connect_ms")),
            ewma_request_latency_ms=_to_float(raw.get("ewma_request_latency_ms")),
            ewma_failure_rate=_to_float(raw.get("ewma_failure_rate")),
            last_updated_epoch=_to_float(raw.get("last_updated_epoch")),
        )

    async def _load_state(self, model: str) -> _ModelMetricsState:
        key = self._key_for_model(model)
        raw = await self._redis.hgetall(key)
        state = None
        if isinstance(raw, dict):
            normalized_raw = {
                str(k.decode("utf-8") if isinstance(k, bytes) else k): (
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                )
                for k, v in raw.items()
            }
            state = self._parse_state(normalized_raw)
        if state is None:
            state = _ModelMetricsState(model=model)
        return state

    async def _save_state(self, state: _ModelMetricsState) -> None:
        key = self._key_for_model(state.model)
        payload = {
            "model": state.model,
            "samples": str(state.samples),
            "route_decisions": str(state.route_decisions),
            "responses": str(state.responses),
            "errors": str(state.errors),
            "ewma_connect_ms": "" if state.ewma_connect_ms is None else f"{state.ewma_connect_ms:.6f}",
            "ewma_request_latency_ms": ""
            if state.ewma_request_latency_ms is None
            else f"{state.ewma_request_latency_ms:.6f}",
            "ewma_failure_rate": ""
            if state.ewma_failure_rate is None
            else f"{state.ewma_failure_rate:.6f}",
            "last_updated_epoch": ""
            if state.last_updated_epoch is None
            else f"{state.last_updated_epoch:.6f}",
        }
        await self._redis.hset(key, mapping=payload)
        await self._redis.expire(key, self._key_ttl_seconds)

    async def record_route_decision(self, model: str, task: str, complexity: str) -> None:
        del task, complexity
        key = model.strip()
        if not key:
            return
        state = await self._load_state(key)
        state.route_decisions += 1
        state.last_updated_epoch = time.time()
        await self._save_state(state)

    async def record_connect(self, model: str, connect_ms: float) -> None:
        key = model.strip()
        if not key:
            return
        value = max(0.0, float(connect_ms))
        state = await self._load_state(key)
        state.samples += 1
        state.ewma_connect_ms = _ewma(state.ewma_connect_ms, value, self._alpha)
        state.last_updated_epoch = time.time()
        await self._save_state(state)

    async def record_response(
        self,
        model: str,
        status: int,
        request_latency_ms: float | None,
    ) -> None:
        key = model.strip()
        if not key:
            return
        state = await self._load_state(key)
        state.responses += 1
        state.samples += 1
        failure_flag = 1.0 if _is_failure_status(int(status)) else 0.0
        state.ewma_failure_rate = _ewma(state.ewma_failure_rate, failure_flag, self._alpha)
        if failure_flag > 0:
            state.errors += 1
        if request_latency_ms is not None:
            latency = max(0.0, float(request_latency_ms))
            state.ewma_request_latency_ms = _ewma(
                state.ewma_request_latency_ms,
                latency,
                self._alpha,
            )
        state.last_updated_epoch = time.time()
        await self._save_state(state)

    async def record_error(self, model: str, error_type: str | None = None) -> None:
        del error_type
        key = model.strip()
        if not key:
            return
        state = await self._load_state(key)
        state.errors += 1
        state.samples += 1
        state.ewma_failure_rate = _ewma(state.ewma_failure_rate, 1.0, self._alpha)
        state.last_updated_epoch = time.time()
        await self._save_state(state)

    async def snapshot_all(self) -> dict[str, ModelMetricsSnapshot]:
        output: dict[str, ModelMetricsSnapshot] = {}
        cursor: Any = 0
        pattern = f"{self._key_prefix}*"
        while True:
            cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=200)
            for key in keys:
                raw = await self._redis.hgetall(key)
                normalized_raw = {
                    str(k.decode("utf-8") if isinstance(k, bytes) else k): (
                        v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    )
                    for k, v in raw.items()
                }
                state = self._parse_state(normalized_raw)
                if state is None:
                    continue
                output[state.model] = state.to_snapshot()
            if cursor in (0, "0"):
                break
        return output


def build_live_metrics_store(
    *,
    redis_url: str | None,
    logger: logging.Logger | None = None,
    alpha: float = 0.2,
) -> LiveMetricsStore:
    if not redis_url:
        return InMemoryLiveMetricsStore(alpha=alpha)
    try:
        from redis.asyncio import from_url
    except Exception as exc:  # pragma: no cover - fallback behavior tested via builder.
        if logger is not None:
            logger.warning("live_metrics_redis_unavailable reason=%s fallback=in_memory", str(exc))
        return InMemoryLiveMetricsStore(alpha=alpha)

    try:
        client = from_url(redis_url, decode_responses=False)
        return RedisLiveMetricsStore(redis_client=client, alpha=alpha)
    except Exception as exc:
        if logger is not None:
            logger.warning("live_metrics_redis_connect_failed reason=%s fallback=in_memory", str(exc))
        return InMemoryLiveMetricsStore(alpha=alpha)


class LiveMetricsCollector:
    def __init__(
        self,
        *,
        store: LiveMetricsStore,
        logger: logging.Logger | None = None,
        enabled: bool = True,
        queue_size: int = 8192,
    ) -> None:
        self._store = store
        self._logger = logger
        self._enabled = enabled
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max(1, queue_size))
        self._worker_task: asyncio.Task[None] | None = None
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    async def start(self) -> None:
        if not self._enabled or self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._run(), name="live-metrics-collector")

    async def close(self) -> None:
        if self._worker_task is None:
            return
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    def ingest(self, event: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(dict(event))
        except asyncio.QueueFull:
            self._dropped_events += 1

    async def snapshot(self) -> dict[str, ModelMetricsSnapshot]:
        return await self._store.snapshot_all()

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                self._queue.task_done()
                break
            try:
                await self._process_event(event)
            except Exception as exc:
                if self._logger is not None:
                    self._logger.debug("live_metrics_process_failed error=%s", str(exc))
            finally:
                self._queue.task_done()

    async def _process_event(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "")
        model = str(event.get("model") or "").strip()

        if event_name == "route_decision":
            selected_model = str(event.get("selected_model") or "").strip()
            if selected_model:
                await self._store.record_route_decision(
                    model=selected_model,
                    task=str(event.get("task") or ""),
                    complexity=str(event.get("complexity") or ""),
                )
            return

        if not model:
            return

        if event_name == "proxy_upstream_connected":
            connect_ms = event.get("connect_ms")
            if isinstance(connect_ms, (int, float)):
                await self._store.record_connect(model=model, connect_ms=float(connect_ms))
            return

        if event_name == "proxy_response":
            status = event.get("status")
            if not isinstance(status, int):
                return
            request_latency_ms = event.get("request_latency_ms")
            latency_value = float(request_latency_ms) if isinstance(request_latency_ms, (int, float)) else None
            await self._store.record_response(
                model=model,
                status=status,
                request_latency_ms=latency_value,
            )
            return

        if event_name == "proxy_request_error":
            await self._store.record_error(
                model=model,
                error_type=(str(event.get("error_type")) if event.get("error_type") else None),
            )


def snapshot_to_dict(snapshot: dict[str, ModelMetricsSnapshot]) -> dict[str, dict[str, Any]]:
    return {
        model: {
            "samples": value.samples,
            "route_decisions": value.route_decisions,
            "responses": value.responses,
            "errors": value.errors,
            "ewma_connect_ms": value.ewma_connect_ms,
            "ewma_request_latency_ms": value.ewma_request_latency_ms,
            "ewma_failure_rate": value.ewma_failure_rate,
            "last_updated_epoch": value.last_updated_epoch,
        }
        for model, value in sorted(snapshot.items())
    }


def snapshot_to_json(snapshot: dict[str, ModelMetricsSnapshot]) -> str:
    return json.dumps(snapshot_to_dict(snapshot), ensure_ascii=True, separators=(",", ":"), default=str)
