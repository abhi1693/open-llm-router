import asyncio
import base64
import json
import time
from datetime import datetime, timedelta, timezone

import httpx
from starlette.datastructures import Headers

from open_llm_router.config import BackendAccount
from open_llm_router.proxy import (
    BackendProxy,
    _build_upstream_headers,
    _parse_retry_after_seconds,
    _prepare_upstream_request,
)
from open_llm_router.router_engine import RouteDecision


def _decision(
    selected: str,
    fallbacks: list[str],
    source: str = "auto",
) -> RouteDecision:
    return RouteDecision(
        selected_model=selected,
        source=source,
        task="general",
        complexity="low",
        requested_model="auto",
        fallback_models=fallbacks,
        signals={},
    )


def _jwt_with_account_id(account_id: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode("ascii").rstrip("=")
    payload_obj = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    payload = base64.urlsafe_b64encode(json.dumps(payload_obj).encode("utf-8")).decode("ascii").rstrip("=")
    return f"{header}.{payload}.sig"


def test_build_targets_across_accounts_and_fallback_models():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-a",
                provider="openai",
                base_url="http://provider-a",
                api_key="key-a",
                models=["m1", "m2"],
            ),
            BackendAccount(
                name="acct-b",
                provider="openai",
                base_url="http://provider-b",
                api_key="key-b",
                models=["m1", "m3"],
            ),
        ],
    )
    targets = proxy._build_candidate_targets(_decision("m1", ["m2", "m3"]))
    assert [target.label for target in targets] == [
        "acct-a:m1",
        "acct-b:m1",
        "acct-a:m2",
        "acct-b:m3",
    ]
    asyncio.run(proxy.close())


def test_build_targets_request_source_ignores_fallbacks():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-a",
                provider="openai",
                base_url="http://provider-a",
                api_key="key-a",
                models=["m1", "m2"],
            ),
            BackendAccount(
                name="acct-b",
                provider="openai",
                base_url="http://provider-b",
                api_key="key-b",
                models=["m1", "m3"],
            ),
        ],
    )
    targets = proxy._build_candidate_targets(_decision("m1", ["m2", "m3"], source="request"))
    assert [target.label for target in targets] == ["acct-a:m1", "acct-b:m1"]
    asyncio.run(proxy.close())


def test_build_targets_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("ACCOUNT_A_KEY", "env-key-a")
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-a",
                provider="openai",
                base_url="http://provider-a",
                api_key_env="ACCOUNT_A_KEY",
                models=["m1"],
            )
        ],
    )
    targets = proxy._build_candidate_targets(_decision("m1", []))
    token = asyncio.run(proxy._resolve_bearer_token(targets[0].account))
    assert token == "env-key-a"
    asyncio.run(proxy.close())


def test_legacy_backend_used_when_accounts_absent():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[],
    )
    targets = proxy._build_candidate_targets(_decision("m1", ["m2"]))
    assert [target.label for target in targets] == ["default:m1", "default:m2"]
    token = asyncio.run(proxy._resolve_bearer_token(targets[0].account))
    assert token == "legacy-key"
    asyncio.run(proxy.close())


def test_oauth_account_uses_access_token():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key=None,
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-oauth",
                provider="openai-codex",
                base_url="https://chatgpt.com/backend-api",
                auth_mode="oauth",
                oauth_access_token="oauth-access-1",
                oauth_refresh_token="oauth-refresh-1",
                oauth_expires_at=int(time.time()) + 3600,
                models=["gpt-5.2-codex"],
            )
        ],
    )

    targets = proxy._build_candidate_targets(_decision("gpt-5.2-codex", []))
    token = asyncio.run(proxy._resolve_bearer_token(targets[0].account))
    assert token == "oauth-access-1"
    asyncio.run(proxy.close())


def test_oauth_account_refreshes_when_expired():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key=None,
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-oauth",
                provider="openai-codex",
                base_url="https://chatgpt.com/backend-api",
                auth_mode="oauth",
                oauth_access_token="expired-token",
                oauth_refresh_token="refresh-token-1",
                oauth_expires_at=int(time.time()) - 60,
                oauth_client_id="client-id-1",
                models=["gpt-5.2-codex"],
            )
        ],
    )

    asyncio.run(proxy.client.aclose())

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://auth.openai.com/oauth/token"
        return httpx.Response(
            status_code=200,
            json={
                "access_token": "new-access-token",
                "refresh_token": "refresh-token-2",
                "expires_in": 1800,
            },
        )

    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)

    targets = proxy._build_candidate_targets(_decision("gpt-5.2-codex", []))
    token = asyncio.run(proxy._resolve_bearer_token(targets[0].account))
    assert token == "new-access-token"
    asyncio.run(proxy.close())


def test_oauth_account_resolves_chatgpt_account_id_from_token():
    token = _jwt_with_account_id("acct_abc123")
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key=None,
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-oauth",
                provider="openai-codex",
                base_url="https://chatgpt.com/backend-api",
                auth_mode="oauth",
                oauth_access_token=token,
                oauth_refresh_token="refresh-token-1",
                oauth_expires_at=int(time.time()) + 3600,
                models=["gpt-5.2-codex"],
            )
        ],
    )

    targets = proxy._build_candidate_targets(_decision("gpt-5.2-codex", []))
    resolved = asyncio.run(proxy._resolve_bearer_token(targets[0].account))
    assert resolved == token
    assert proxy._resolve_oauth_account_id(targets[0].account) == "acct_abc123"
    asyncio.run(proxy.close())


def test_build_upstream_headers_adds_codex_account_headers():
    headers = _build_upstream_headers(
        incoming_headers=Headers({"content-type": "application/json"}),
        bearer_token="token-1",
        provider="openai-codex",
        oauth_account_id="acct_xyz",
        organization=None,
        project=None,
    )

    assert headers["Authorization"] == "Bearer token-1"
    assert headers["chatgpt-account-id"] == "acct_xyz"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["originator"] == "pi"


def test_prepare_upstream_request_maps_chat_completions_to_codex_responses():
    payload = {
        "model": "gpt-5.2-codex",
        "stream": True,
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        "temperature": 0.2,
    }

    prepared = _prepare_upstream_request(
        path="/v1/chat/completions",
        payload=payload,
        provider="openai-codex",
        stream=True,
    )

    assert prepared.path == "/codex/responses"
    assert prepared.stream is True
    assert prepared.adapter == "chat_completions"
    assert prepared.payload["model"] == "gpt-5.2-codex"
    assert prepared.payload["store"] is False
    assert prepared.payload["stream"] is True
    assert prepared.payload["instructions"] == "Be concise."
    assert prepared.payload["temperature"] == 0.2
    assert prepared.payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello."}],
        }
    ]


def test_prepare_upstream_request_converts_chat_multimodal_content():
    payload = {
        "model": "gpt-5.2-codex",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    }

    prepared = _prepare_upstream_request(
        path="/v1/chat/completions",
        payload=payload,
        provider="openai-codex",
        stream=False,
    )

    assert prepared.path == "/codex/responses"
    assert prepared.stream is True
    assert prepared.payload["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe image"},
                {"type": "input_image", "image_url": "https://example.com/cat.png"},
            ],
        }
    ]


def test_prepare_upstream_request_normalizes_assistant_and_tool_roles():
    payload = {
        "model": "gpt-5.2-codex",
        "messages": [
            {"role": "system", "content": "Use concise answers."},
            {"role": "assistant", "content": "Previous answer."},
            {"role": "tool", "content": "filesystem result"},
            {"role": "user", "content": "Continue."},
        ],
    }

    prepared = _prepare_upstream_request(
        path="/v1/chat/completions",
        payload=payload,
        provider="openai-codex",
        stream=True,
    )

    assert prepared.path == "/codex/responses"
    assert prepared.payload["instructions"] == "Use concise answers."
    assert prepared.payload["input"][0] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Previous answer."}],
    }
    assert prepared.payload["input"][1] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Tool output:\nfilesystem result"}],
    }
    assert prepared.payload["input"][2] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Continue."}],
    }


def test_prepare_upstream_request_maps_chat_tools_for_responses_api():
    payload = {
        "model": "gpt-5.2-codex",
        "messages": [{"role": "user", "content": "Run listing tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in cwd",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "list_files"}},
    }

    prepared = _prepare_upstream_request(
        path="/v1/chat/completions",
        payload=payload,
        provider="openai-codex",
        stream=True,
    )

    assert prepared.payload["tools"] == [
        {
            "type": "function",
            "name": "list_files",
            "description": "List files in cwd",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]
    assert prepared.payload["tool_choice"] == {"type": "function", "name": "list_files"}


def test_parse_retry_after_seconds_numeric():
    headers = httpx.Headers({"Retry-After": "12"})
    assert _parse_retry_after_seconds(headers, default_seconds=30.0) == 12.0


def test_parse_retry_after_seconds_http_date():
    future = datetime.now(timezone.utc) + timedelta(seconds=8)
    headers = httpx.Headers({"Retry-After": future.strftime("%a, %d %b %Y %H:%M:%S GMT")})
    value = _parse_retry_after_seconds(headers, default_seconds=30.0)
    assert 0 < value <= 8.5
