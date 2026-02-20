from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from fastapi.responses import Response

_redis_from_url: Any | None

try:
    from redis.asyncio import from_url as _redis_from_url
except ImportError:  # pragma: no cover - optional dependency.
    _redis_from_url = None

if TYPE_CHECKING:
    import logging


@dataclass(slots=True)
class IdempotencyConfig:
    enabled: bool = True
    ttl_seconds: int = 120
    wait_timeout_seconds: float = 30.0


@dataclass(slots=True)
class CachedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    created_at_epoch: float

    def to_fastapi_response(self) -> Response:
        response = Response(
            content=self.body,
            status_code=self.status_code,
            headers=self.headers,
        )
        response.headers["x-router-idempotency-status"] = "replayed"
        return response


@dataclass(slots=True)
class BeginResult:
    mode: str
    key: str
    cached: CachedResponse | None = None
    waiter: asyncio.Event | None = None


class IdempotencyBackend(Protocol):
    async def begin(self, key: str) -> BeginResult: ...

    async def wait_for_existing(self, result: BeginResult) -> CachedResponse | None: ...

    async def store(
        self,
        key: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None: ...

    async def release_without_store(self, key: str) -> None: ...


class AsyncKeyValueStore(Protocol):
    async def get(self, key: str) -> bytes | str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...


class KeyValueStoreFactory(Protocol):
    def __call__(self, redis_url: str) -> AsyncKeyValueStore: ...


class IdempotencyStore:
    def __init__(self, config: IdempotencyConfig) -> None:
        self._config = config
        self._lock = asyncio.Lock()
        self._cache: dict[str, tuple[float, CachedResponse]] = {}
        self._inflight: dict[str, asyncio.Event] = {}

    async def begin(self, key: str) -> BeginResult:
        if not self._config.enabled:
            return BeginResult(mode="bypass", key=key)

        async with self._lock:
            self._prune_locked()
            cached = self._cache.get(key)
            if cached is not None:
                _, response = cached
                return BeginResult(mode="replay", key=key, cached=response)

            waiter = self._inflight.get(key)
            if waiter is not None:
                return BeginResult(mode="wait", key=key, waiter=waiter)

            waiter = asyncio.Event()
            self._inflight[key] = waiter
            return BeginResult(mode="leader", key=key, waiter=waiter)

    async def wait_for_existing(self, result: BeginResult) -> CachedResponse | None:
        if result.mode != "wait" or result.waiter is None:
            return None
        try:
            await asyncio.wait_for(
                result.waiter.wait(),
                timeout=self._config.wait_timeout_seconds,
            )
        except TimeoutError:
            return None
        async with self._lock:
            cached = self._cache.get(result.key)
            if cached is None:
                return None
            _, response = cached
            return response

    async def store(
        self,
        key: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        expires_at = time.time() + self._config.ttl_seconds
        response = CachedResponse(
            status_code=status_code,
            headers=dict(headers),
            body=body,
            created_at_epoch=time.time(),
        )
        async with self._lock:
            self._cache[key] = (expires_at, response)
            waiter = self._inflight.pop(key, None)
            if waiter is not None:
                waiter.set()

    async def release_without_store(self, key: str) -> None:
        async with self._lock:
            waiter = self._inflight.pop(key, None)
            if waiter is not None:
                waiter.set()

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [key for key, (until, _) in self._cache.items() if now >= until]
        for key in expired:
            self._cache.pop(key, None)


class KeyValueIdempotencyStore:
    def __init__(self, config: IdempotencyConfig, kv_store: AsyncKeyValueStore) -> None:
        self._config = config
        self._kv_store = kv_store

    async def begin(self, key: str) -> BeginResult:
        if not self._config.enabled:
            return BeginResult(mode="bypass", key=key)

        raw = await self._kv_store.get(key)
        if not raw:
            return BeginResult(mode="leader", key=key)

        cached = _deserialize_cached_response(raw)
        if cached is None:
            return BeginResult(mode="leader", key=key)
        return BeginResult(mode="replay", key=key, cached=cached)

    async def wait_for_existing(
        self,
        result: BeginResult,
    ) -> CachedResponse | None:
        _ = result
        # Redis backend currently supports replay across requests/pods, but not distributed
        # in-flight leader election/waiting for concurrent duplicates.
        return None

    async def store(
        self,
        key: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        serialized = _serialize_cached_response(
            status_code=status_code,
            headers=headers,
            body=body,
        )
        await self._kv_store.set(
            key,
            serialized,
            ttl_seconds=self._config.ttl_seconds,
        )

    async def release_without_store(self, key: str) -> None:
        _ = key


def build_idempotency_store(
    config: IdempotencyConfig,
    redis_url: str | None = None,
    logger: logging.Logger | None = None,
    create_key_value_store: KeyValueStoreFactory | None = None,
) -> IdempotencyBackend:
    if not redis_url:
        return IdempotencyStore(config=config)

    factory = create_key_value_store or build_redis_key_value_store
    try:
        return KeyValueIdempotencyStore(config=config, kv_store=factory(redis_url))
    except RuntimeError as exc:
        if logger is not None:
            logger.warning(
                "idempotency_redis_unavailable reason=%s fallback=in_memory",
                str(exc),
            )
        return IdempotencyStore(config=config)


class RedisAsyncKeyValueStore:
    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def get(self, key: str) -> bytes | str | None:
        value = await self._redis.get(key)
        if value is None:
            return None
        if isinstance(value, (bytes, str)):
            return value
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        await self._redis.set(key, value, ex=ttl_seconds)


def build_redis_key_value_store(redis_url: str) -> AsyncKeyValueStore:
    if _redis_from_url is None:  # pragma: no cover - covered by fallback tests.
        msg = "redis package is not installed"
        raise RuntimeError(msg)
    client = _redis_from_url(redis_url, decode_responses=False)
    return RedisAsyncKeyValueStore(redis_client=client)


def _serialize_cached_response(
    status_code: int,
    headers: dict[str, str],
    body: bytes,
) -> str:
    payload = {
        "status_code": int(status_code),
        "headers": dict(headers),
        "body_b64": base64.b64encode(body).decode("ascii"),
        "created_at_epoch": time.time(),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _deserialize_cached_response(raw: bytes | str) -> CachedResponse | None:
    try:
        decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(decoded)
        status_code = int(payload["status_code"])
        headers = payload.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        body_b64 = payload.get("body_b64", "")
        if not isinstance(body_b64, str):
            return None
        body = base64.b64decode(body_b64.encode("ascii"))
        created_at = float(payload.get("created_at_epoch", 0.0))
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None

    normalized_headers: dict[str, str] = {}
    for key, value in headers.items():
        normalized_headers[str(key)] = str(value)
    return CachedResponse(
        status_code=status_code,
        headers=normalized_headers,
        body=body,
        created_at_epoch=created_at,
    )


def build_idempotency_cache_key(
    idempotency_key: str,
    tenant_id: str,
    path: str,
    payload: dict[str, Any],
) -> str:
    # Use a fixed-size fingerprint to avoid oversized cache keys and reduce key serialization overhead.
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    payload_fingerprint = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    return f"{tenant_id}|{path}|{idempotency_key}|sha256:{payload_fingerprint}"
