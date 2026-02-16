from __future__ import annotations

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from open_llm_router.idempotency import (
    IdempotencyConfig,
    IdempotencyStore,
    build_idempotency_store,
)
from open_llm_router.main import app
from open_llm_router.proxy import BackendProxy
from open_llm_router.settings import get_settings


def _set_default_test_env(monkeypatch) -> None:
    monkeypatch.setenv("ROUTING_CONFIG_PATH", "router.profile.yaml")


def _build_client(monkeypatch, **env):
    _set_default_test_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return TestClient(app)


def test_settings_redis_url_is_loaded(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://generic:6379/0")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.redis_url == "redis://generic:6379/0"


def test_non_stream_requests_replay_with_same_idempotency_key(monkeypatch):
    calls = {"count": 0}

    async def fake_forward_with_fallback(
        self,
        path,
        payload,
        incoming_headers,
        route_decision,
        stream,
        request_id=None,
    ):
        calls["count"] += 1
        return JSONResponse(
            content={"ok": True, "call_count": calls["count"]},
            headers={"x-router-model": "fake-model", "x-router-provider": "fake-provider"},
        )

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    with _build_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        headers = {"Idempotency-Key": "abc-123"}
        first = client.post("/v1/chat/completions", json=payload, headers=headers)
        second = client.post("/v1/chat/completions", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["call_count"] == 1
    assert second.json()["call_count"] == 1
    assert first.headers.get("x-router-idempotency-status") == "stored"
    assert second.headers.get("x-router-idempotency-status") == "replayed"
    assert calls["count"] == 1


def test_streaming_requests_are_not_cached_by_idempotency(monkeypatch):
    calls = {"count": 0}

    async def fake_forward_with_fallback(
        self,
        path,
        payload,
        incoming_headers,
        route_decision,
        stream,
        request_id=None,
    ):
        calls["count"] += 1
        return JSONResponse(content={"ok": True, "call_count": calls["count"]})

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    with _build_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        headers = {"Idempotency-Key": "abc-456"}
        first = client.post("/v1/chat/completions", json=payload, headers=headers)
        second = client.post("/v1/chat/completions", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["call_count"] == 1
    assert second.json()["call_count"] == 2
    assert first.headers.get("x-router-idempotency-status") is None
    assert second.headers.get("x-router-idempotency-status") is None
    assert calls["count"] == 2


def test_build_idempotency_store_falls_back_to_in_memory_when_redis_factory_fails():
    config = IdempotencyConfig()

    def failing_factory(_url):
        raise RuntimeError("redis unavailable")

    store = build_idempotency_store(
        config=config,
        redis_url="redis://localhost:6379/0",
        create_key_value_store=failing_factory,
    )
    assert isinstance(store, IdempotencyStore)


def test_build_idempotency_store_uses_redis_factory_when_available():
    config = IdempotencyConfig()

    class _FakeKVStore:
        async def get(self, key):
            raise NotImplementedError

        async def set(self, key, value, ttl_seconds):
            raise NotImplementedError

    fake_store = _FakeKVStore()

    def factory(_url):
        return fake_store

    store = build_idempotency_store(
        config=config,
        redis_url="redis://localhost:6379/0",
        create_key_value_store=factory,
    )
    assert store.__class__.__name__ == "KeyValueIdempotencyStore"
