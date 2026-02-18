from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi.testclient import TestClient

from open_llm_router.main import app
from open_llm_router.settings import get_settings


def _set_default_test_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("ROUTING_CONFIG_PATH", "router.profile.yaml")


def _build_client(monkeypatch: Any, **env: Any) -> Any:
    _set_default_test_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return TestClient(app)


def test_v1_models_allows_when_auth_disabled(monkeypatch: Any) -> None:
    with _build_client(monkeypatch, INGRESS_AUTH_REQUIRED="false") as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        ids = [item["id"] for item in body.get("data", [])]
        assert "auto" in ids
        assert len(ids) >= 1


def test_v1_models_rejects_without_token_when_auth_required(monkeypatch: Any) -> None:
    with _build_client(
        monkeypatch,
        INGRESS_AUTH_REQUIRED="true",
        INGRESS_API_KEYS="router-key-1",
    ) as client:
        response = client.get("/v1/models")
        assert response.status_code == 401


def test_v1_models_accepts_valid_api_key(monkeypatch: Any) -> None:
    with _build_client(
        monkeypatch,
        INGRESS_AUTH_REQUIRED="true",
        INGRESS_API_KEYS="router-key-1,router-key-2",
    ) as client:
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer router-key-2"}
        )
        assert response.status_code == 200


def test_v1_models_accepts_valid_oauth_token(monkeypatch: Any) -> None:
    secret = "local-test-secret-with-32-bytes-minimum"
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": "user-123",
            "iss": "https://chatgpt.com",
            "aud": "open-llm-router",
            "scope": "openid profile",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        secret,
        algorithm="HS256",
    )

    with _build_client(
        monkeypatch,
        INGRESS_AUTH_REQUIRED="true",
        INGRESS_API_KEYS="",
        OAUTH_ENABLED="true",
        OAUTH_ISSUER="https://chatgpt.com",
        OAUTH_AUDIENCE="open-llm-router",
        OAUTH_ALGORITHMS="HS256",
        OAUTH_JWT_SECRET=secret,
    ) as client:
        response = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
