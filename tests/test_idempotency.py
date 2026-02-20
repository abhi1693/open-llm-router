from __future__ import annotations

from typing import Any

import pytest
from fastapi.responses import JSONResponse

from open_llm_router.config.settings import get_settings
from open_llm_router.gateway.idempotency import (
    IdempotencyConfig,
    IdempotencyStore,
    build_idempotency_cache_key,
    build_idempotency_store,
)
from open_llm_router.gateway.proxy import BackendProxy
from open_llm_router.server.main import app
from tests.client_test_utils import build_test_client


def test_settings_redis_url_is_loaded(monkeypatch: Any) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://generic:6379/0")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.redis_url == "redis://generic:6379/0"


def test_non_stream_requests_replay_with_same_idempotency_key(monkeypatch: Any) -> None:
    calls = {"count": 0}

    async def fake_forward_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        calls["count"] += 1
        return JSONResponse(
            content={"ok": True, "call_count": calls["count"]},
            headers={
                "x-router-model": "fake-model",
                "x-router-provider": "fake-provider",
            },
        )

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    with build_test_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
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


def test_streaming_requests_are_not_cached_by_idempotency(monkeypatch: Any) -> None:
    calls = {"count": 0}

    async def fake_forward_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        calls["count"] += 1
        return JSONResponse(content={"ok": True, "call_count": calls["count"]})

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    with build_test_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
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


def test_build_idempotency_store_falls_back_to_in_memory_when_redis_factory_fails() -> (
    None
):
    config = IdempotencyConfig()

    def failing_factory(redis_url: str) -> Any:
        _ = redis_url
        msg = "redis unavailable"
        raise RuntimeError(msg)

    store = build_idempotency_store(
        config=config,
        redis_url="redis://localhost:6379/0",
        create_key_value_store=failing_factory,
    )
    assert isinstance(store, IdempotencyStore)


def test_build_idempotency_store_uses_redis_factory_when_available() -> None:
    config = IdempotencyConfig()

    class _FakeKVStore:
        async def get(self, key: Any) -> Any:
            raise NotImplementedError

        async def set(self, key: Any, value: Any, ttl_seconds: Any) -> Any:
            raise NotImplementedError

    fake_store = _FakeKVStore()

    def factory(redis_url: str) -> Any:
        _ = redis_url
        return fake_store

    store = build_idempotency_store(
        config=config,
        redis_url="redis://localhost:6379/0",
        create_key_value_store=factory,
    )
    assert store.__class__.__name__ == "KeyValueIdempotencyStore"


def test_build_idempotency_cache_key_is_hashed_and_stable() -> None:
    payload_a = {"messages": [{"role": "user", "content": "hello"}], "stream": False}
    payload_b = {"stream": False, "messages": [{"content": "hello", "role": "user"}]}

    key_a = build_idempotency_cache_key(
        idempotency_key="abc",
        tenant_id="tenant-1",
        path="/v1/chat/completions",
        payload=payload_a,
    )
    key_b = build_idempotency_cache_key(
        idempotency_key="abc",
        tenant_id="tenant-1",
        path="/v1/chat/completions",
        payload=payload_b,
    )

    assert key_a == key_b
    assert "|sha256:" in key_a
    assert len(key_a.rsplit("|", 1)[-1]) == len("sha256:") + 64


def test_proxy_terminal_event_emitted_for_success(monkeypatch: Any) -> None:
    async def fake_forward_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        return JSONResponse(content={"ok": True}, status_code=200)

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    captured: list[dict[str, Any]] = []
    with build_test_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
        previous_hook = app.state.audit_event_hook
        app.state.audit_event_hook = captured.append
        app.state.audit_payload_summary_enabled = False
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            app.state.audit_event_hook = previous_hook

    assert response.status_code == 200
    terminal_events = [
        event for event in captured if event.get("event") == "proxy_terminal"
    ]
    assert len(terminal_events) == 1
    event = terminal_events[0]
    assert event["outcome"] == "success"
    assert event["status"] == 200
    assert event["path"] == "/v1/chat/completions"


def test_proxy_terminal_event_emitted_for_exhausted(monkeypatch: Any) -> None:
    async def fake_forward_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "routing_exhausted",
                    "message": "All model/account targets failed.",
                },
            },
        )

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    captured: list[dict[str, Any]] = []
    with build_test_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
        previous_hook = app.state.audit_event_hook
        app.state.audit_event_hook = captured.append
        app.state.audit_payload_summary_enabled = False
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        finally:
            app.state.audit_event_hook = previous_hook

    assert response.status_code == 502
    terminal_events = [
        event for event in captured if event.get("event") == "proxy_terminal"
    ]
    assert len(terminal_events) == 1
    event = terminal_events[0]
    assert event["outcome"] == "exhausted"
    assert event["status"] == 502
    assert event["error_type"] == "routing_exhausted"


def test_proxy_terminal_event_emitted_for_unhandled_error(monkeypatch: Any) -> None:
    async def fake_forward_with_fallback(*_args: Any, **_kwargs: Any) -> Any:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        BackendProxy,
        "forward_with_fallback",
        fake_forward_with_fallback,
    )

    captured: list[dict[str, Any]] = []
    with build_test_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
        previous_hook = app.state.audit_event_hook
        app.state.audit_event_hook = captured.append
        app.state.audit_payload_summary_enabled = False
        try:
            with pytest.raises(RuntimeError):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
        finally:
            app.state.audit_event_hook = previous_hook

    terminal_events = [
        event for event in captured if event.get("event") == "proxy_terminal"
    ]
    assert len(terminal_events) == 1
    event = terminal_events[0]
    assert event["outcome"] == "error"
    assert event["status"] == 500
    assert event["error_type"] == "RuntimeError"
