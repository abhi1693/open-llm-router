import asyncio
import base64
import json
import time
from datetime import datetime, timedelta, timezone

import httpx
from starlette.datastructures import Headers

from open_llm_router.config import BackendAccount
from open_llm_router.proxy import (
    BackendTarget,
    BackendProxy,
    _build_upstream_headers,
    _parse_retry_after_seconds,
    _prepare_upstream_request,
    _request_error_details,
)
from open_llm_router.router_engine import RouteDecision


def _decision(
    selected: str,
    fallbacks: list[str],
    source: str = "auto",
    provider_preferences: dict | None = None,
) -> RouteDecision:
    return RouteDecision(
        selected_model=selected,
        source=source,
        task="general",
        complexity="low",
        requested_model="auto",
        fallback_models=fallbacks,
        signals={},
        provider_preferences=provider_preferences or {},
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


def test_build_targets_applies_provider_order_preference():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            [],
            provider_preferences={"order": ["gemini", "openai"]},
        )
    )
    assert [target.account_name for target in targets] == ["acct-gemini", "acct-openai"]
    asyncio.run(proxy.close())


def test_build_targets_applies_provider_only_filter():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            [],
            provider_preferences={"only": ["gemini"]},
        )
    )
    assert [target.account_name for target in targets] == ["acct-gemini"]
    asyncio.run(proxy.close())


def test_build_targets_applies_provider_ignore_filter():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            [],
            provider_preferences={"ignore": ["openai"]},
        )
    )
    assert [target.account_name for target in targets] == ["acct-gemini"]
    asyncio.run(proxy.close())


def test_build_targets_applies_provider_sort_price():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai/m1": {
                "costs": {"input_per_1k": 0.0020, "output_per_1k": 0.0040},
                "priors": {"latency_ms": 700},
            },
            "gemini/m1": {
                "costs": {"input_per_1k": 0.0002, "output_per_1k": 0.0008},
                "priors": {"latency_ms": 900},
            },
        },
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            [],
            provider_preferences={"sort": "price"},
        )
    )
    assert [target.account_name for target in targets] == ["acct-gemini", "acct-openai"]
    asyncio.run(proxy.close())


def test_build_targets_applies_provider_sort_latency():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai/m1": {
                "costs": {"input_per_1k": 0.0020, "output_per_1k": 0.0040},
                "priors": {"latency_ms": 450},
            },
            "gemini/m1": {
                "costs": {"input_per_1k": 0.0002, "output_per_1k": 0.0008},
                "priors": {"latency_ms": 900},
            },
        },
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            [],
            provider_preferences={"sort": "latency"},
        )
    )
    assert [target.account_name for target in targets] == ["acct-openai", "acct-gemini"]
    asyncio.run(proxy.close())


def test_build_targets_partition_model_keeps_model_grouping():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai/m1": {"priors": {"latency_ms": 700}},
            "gemini/m1": {"priors": {"latency_ms": 900}},
            "openai/m2": {"priors": {"latency_ms": 300}},
            "gemini/m2": {"priors": {"latency_ms": 500}},
        },
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1", "openai/m2"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1", "gemini/m2"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            ["m2"],
            provider_preferences={"sort": "latency", "partition": "model"},
        )
    )
    assert [(target.model, target.account_name) for target in targets] == [
        ("m1", "acct-openai"),
        ("m1", "acct-gemini"),
        ("m2", "acct-openai"),
        ("m2", "acct-gemini"),
    ]
    asyncio.run(proxy.close())


def test_build_targets_partition_none_applies_global_sort():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai/m1": {"priors": {"latency_ms": 700}},
            "gemini/m1": {"priors": {"latency_ms": 900}},
            "openai/m2": {"priors": {"latency_ms": 300}},
            "gemini/m2": {"priors": {"latency_ms": 500}},
        },
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1", "openai/m2"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1", "gemini/m2"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            ["m2"],
            provider_preferences={"sort": "latency", "partition": "none"},
        )
    )
    assert [(target.model, target.account_name) for target in targets] == [
        ("m2", "acct-openai"),
        ("m2", "acct-gemini"),
        ("m1", "acct-openai"),
        ("m1", "acct-gemini"),
    ]
    asyncio.run(proxy.close())


def test_filter_targets_by_parameter_support_keeps_compatible_targets():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai/m1": {
                "supported_parameters": ["temperature", "response_format"],
            },
            "gemini/m1": {
                "supported_parameters": ["temperature"],
            },
        },
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(
        _decision(
            "m1",
            [],
            provider_preferences={"require_parameters": True},
        )
    )
    accepted, rejected, requested = proxy._filter_targets_by_parameter_support(
        candidate_targets=targets,
        payload={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
    )

    assert requested == ["response_format", "temperature"]
    assert [target.account_name for target in accepted] == ["acct-openai"]
    assert "acct-gemini:m1" in rejected
    asyncio.run(proxy.close())


def test_forward_with_fallback_returns_400_when_require_parameters_excludes_all_targets():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai/m1": {
                "supported_parameters": ["temperature"],
            },
            "gemini/m1": {
                "supported_parameters": ["temperature"],
            },
        },
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )

    response = asyncio.run(
        proxy.forward_with_fallback(
            path="/v1/chat/completions",
            payload={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "json_object"},
            },
            incoming_headers=Headers({"content-type": "application/json"}),
            route_decision=_decision(
                "m1",
                [],
                provider_preferences={"require_parameters": True},
            ),
            stream=False,
            request_id="req-require-params",
        )
    )

    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"]["constraint"] == "require_parameters"
    asyncio.run(proxy.close())


def test_forward_with_fallback_returns_400_when_provider_filters_exclude_all_targets():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
        ],
    )

    response = asyncio.run(
        proxy.forward_with_fallback(
            path="/v1/chat/completions",
            payload={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
            },
            incoming_headers=Headers({"content-type": "application/json"}),
            route_decision=_decision(
                "m1",
                [],
                provider_preferences={"only": ["gemini"], "ignore": ["openai"]},
            ),
            stream=False,
            request_id="req-provider-filters",
        )
    )

    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"]["constraint"] == "provider"
    assert body["error"]["details"]["only"] == ["gemini"]
    assert body["error"]["details"]["ignore"] == ["openai"]
    asyncio.run(proxy.close())


def test_forward_with_fallback_retries_next_target_when_fallbacks_enabled():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )
    asyncio.run(proxy.client.aclose())

    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(
                status_code=500,
                headers={"Content-Type": "application/json"},
                json={"error": "first target failed"},
                request=request,
            )
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            json={"ok": True},
            request=request,
        )

    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)

    response = asyncio.run(
        proxy.forward_with_fallback(
            path="/v1/chat/completions",
            payload={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
            },
            incoming_headers=Headers({"content-type": "application/json"}),
            route_decision=_decision(
                "m1",
                [],
                provider_preferences={
                    "allow_fallbacks": True,
                    "order": ["openai", "gemini"],
                },
            ),
            stream=False,
            request_id="req-fallbacks-on",
        )
    )

    assert state["calls"] == 2
    assert response.status_code == 200
    assert response.headers["x-router-account"] == "acct-gemini"
    body = json.loads(response.body.decode("utf-8"))
    assert body["ok"] is True
    assert body["_router"]["attempted_targets"] == ["acct-openai:m1", "acct-gemini:m1"]
    asyncio.run(proxy.close())


def test_proxy_request_error_audit_contains_attempt_and_status_fields():
    captured_events: list[dict] = []

    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[500],
        audit_hook=lambda event: captured_events.append(event),
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )
    asyncio.run(proxy.client.aclose())

    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            raise httpx.ConnectTimeout("connect timeout", request=request)
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            json={"ok": True},
            request=request,
        )

    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)

    response = asyncio.run(
        proxy.forward_with_fallback(
            path="/v1/chat/completions",
            payload={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
            },
            incoming_headers=Headers({"content-type": "application/json"}),
            route_decision=_decision(
                "m1",
                [],
                provider_preferences={
                    "allow_fallbacks": True,
                    "order": ["openai", "gemini"],
                },
            ),
            stream=False,
            request_id="req-error-audit-fields",
        )
    )

    assert response.status_code == 200
    error_events = [event for event in captured_events if event.get("event") == "proxy_request_error"]
    assert len(error_events) == 1

    event = error_events[0]
    assert event["attempt"] == 1
    assert event["total_attempts"] == 2
    assert event["status_code"] is None
    assert event["error_type"] == "ConnectTimeout"
    assert event["is_timeout"] is True
    assert event["error"]
    asyncio.run(proxy.close())


def test_forward_with_fallback_does_not_retry_when_fallbacks_disabled():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-openai",
                models=["openai/m1"],
            ),
            BackendAccount(
                name="acct-gemini",
                provider="gemini",
                base_url="http://provider-gemini",
                api_key="key-gemini",
                models=["gemini/m1"],
            ),
        ],
    )
    asyncio.run(proxy.client.aclose())

    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            json={"error": "first target failed"},
            request=request,
        )

    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)

    response = asyncio.run(
        proxy.forward_with_fallback(
            path="/v1/chat/completions",
            payload={
                "model": "auto",
                "messages": [{"role": "user", "content": "hello"}],
            },
            incoming_headers=Headers({"content-type": "application/json"}),
            route_decision=_decision(
                "m1",
                [],
                provider_preferences={
                    "allow_fallbacks": False,
                    "order": ["openai", "gemini"],
                },
            ),
            stream=False,
            request_id="req-fallbacks-off",
        )
    )

    assert state["calls"] == 1
    assert response.status_code == 500
    assert response.headers["x-router-account"] == "acct-openai"
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"] == "first target failed"
    assert body["_router"]["attempted_targets"] == ["acct-openai:m1"]
    asyncio.run(proxy.close())


def test_build_targets_resolves_provider_qualified_model_to_matching_account():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-openai",
                provider="openai",
                base_url="http://provider-openai",
                api_key="key-a",
                models=["openai/gpt-5.2", "gpt-5.2-codex"],
            ),
            BackendAccount(
                name="acct-codex",
                provider="openai-codex",
                base_url="http://provider-codex",
                api_key="key-b",
                models=["gpt-5.2"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(_decision("openai/gpt-5.2", []))
    assert [target.label for target in targets] == ["acct-openai:gpt-5.2"]
    asyncio.run(proxy.close())


def test_build_targets_uses_model_registry_id_for_upstream_model():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        model_registry={
            "openai-codex/gpt-5.3-codex": {
                "id": "gpt-5.3-codex-upstream",
                "name": "GPT 5.3 Codex",
            }
        },
        accounts=[
            BackendAccount(
                name="acct-codex",
                provider="openai-codex",
                base_url="http://provider-codex",
                api_key="key-b",
                models=["openai-codex/gpt-5.3-codex"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(_decision("openai-codex/gpt-5.3-codex", []))
    assert [target.label for target in targets] == ["acct-codex:openai-codex/gpt-5.3-codex"]
    assert targets[0].upstream_model == "gpt-5.3-codex-upstream"
    asyncio.run(proxy.close())


def test_backend_proxy_applies_explicit_timeout_configuration():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[],
        connect_timeout_seconds=1.5,
        read_timeout_seconds=12.0,
        write_timeout_seconds=8.0,
        pool_timeout_seconds=2.5,
    )

    timeout = proxy.client.timeout
    assert timeout.connect == 1.5
    assert timeout.read == 12.0
    assert timeout.write == 8.0
    assert timeout.pool == 2.5
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


def test_build_upstream_headers_defaults_accept_to_sse_for_streaming_requests():
    headers = _build_upstream_headers(
        incoming_headers=Headers({"content-type": "application/json"}),
        bearer_token="token-1",
        provider="nvidia",
        oauth_account_id=None,
        organization=None,
        project=None,
        stream=True,
    )

    assert headers["Accept"] == "text/event-stream"


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


def test_build_targets_resolves_codex_model_key_to_upstream_model_id():
    proxy = BackendProxy(
        base_url="http://legacy",
        timeout_seconds=30,
        backend_api_key="legacy-key",
        retry_statuses=[429, 500],
        accounts=[
            BackendAccount(
                name="acct-codex",
                provider="openai-codex",
                base_url="http://provider-codex",
                api_key="key-b",
                models=["gpt-5.2-codex", "openai-codex/gpt-5.2"],
            ),
        ],
    )

    targets = proxy._build_candidate_targets(_decision("openai-codex/gpt-5.2", []))
    assert targets[0].model == "openai-codex/gpt-5.2"
    assert targets[0].upstream_model == "gpt-5.2"
    asyncio.run(proxy.close())


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


def test_request_error_details_include_type_repr_and_request_metadata():
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    exc = httpx.ConnectTimeout("", request=request)

    details = _request_error_details(exc)

    assert details["error_type"] == "ConnectTimeout"
    assert details["error"]
    assert details["error_repr"]
    assert details["is_timeout"] is True
    assert details["status_code"] is None
    assert details["request_method"] == "POST"
    assert details["request_url"] == "https://example.com/v1/chat/completions"


def test_prepare_upstream_request_sanitizes_gemini_chat_payload():
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "tool_choice": None,
        "reasoning_effort": "high",
        "parallel_tool_calls": True,
        "max_output_tokens": 512,
    }

    prepared = _prepare_upstream_request(
        path="/v1/chat/completions",
        payload=payload,
        provider="gemini",
        stream=True,
    )

    assert prepared.path == "/v1/chat/completions"
    assert prepared.stream is True
    assert prepared.payload["model"] == "gemini-2.5-flash"
    assert prepared.payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert prepared.payload["stream"] is True
    assert prepared.payload["max_tokens"] == 512
    assert "tool_choice" not in prepared.payload
    assert "reasoning_effort" not in prepared.payload
    assert "parallel_tool_calls" not in prepared.payload


def test_prepare_upstream_request_keeps_nvidia_chat_payload_passthrough():
    payload = {
        "model": "z-ai/glm5",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 16384,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False},
    }

    prepared = _prepare_upstream_request(
        path="/v1/chat/completions",
        payload=payload,
        provider="nvidia",
        stream=True,
    )

    assert prepared.path == "/v1/chat/completions"
    assert prepared.stream is True
    assert prepared.payload == payload


def test_parse_retry_after_seconds_numeric():
    headers = httpx.Headers({"Retry-After": "12"})
    assert _parse_retry_after_seconds(headers, default_seconds=30.0) == 12.0


def test_parse_retry_after_seconds_http_date():
    future = datetime.now(timezone.utc) + timedelta(seconds=8)
    headers = httpx.Headers({"Retry-After": future.strftime("%a, %d %b %Y %H:%M:%S GMT")})
    value = _parse_retry_after_seconds(headers, default_seconds=30.0)
    assert 0 < value <= 8.5


def test_to_fastapi_response_injects_router_diagnostics_into_json_payload():
    account = BackendAccount(
        name="acct-openai",
        provider="openai",
        base_url="http://provider-openai",
        api_key="key-openai",
        models=["openai/m1"],
    )
    target = BackendTarget(
        account=account,
        account_name="acct-openai",
        provider="openai",
        base_url="http://provider-openai",
        model="openai/m1",
        auth_mode="api_key",
        organization=None,
        project=None,
        upstream_model="m1",
    )
    route_decision = RouteDecision(
        selected_model="openai/m1",
        source="auto",
        task="general",
        complexity="low",
        requested_model="auto",
        fallback_models=["openai/m2"],
        signals={},
        ranked_models=["openai/m1", "openai/m2"],
        candidate_scores=[{"model": "openai/m1", "utility": 0.82}],
        provider_preferences={"sort": "latency", "partition": "model"},
    )
    upstream = httpx.Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        json={"id": "resp_1", "object": "response", "model": "m1"},
        request=httpx.Request("POST", "https://provider-openai/v1/responses"),
    )

    response = asyncio.run(
        BackendProxy._to_fastapi_response(
            upstream=upstream,
            stream=False,
            upstream_stream=False,
            target=target,
            attempted_targets=["acct-openai:m1"],
            attempted_upstream_models=["m1"],
            request_latency_ms=123.4567,
            route_decision=route_decision,
            request_id="req_diag_1",
            adapter="passthrough",
            audit_hook=None,
        )
    )

    body = json.loads(response.body.decode("utf-8"))
    assert body["_router"]["request_id"] == "req_diag_1"
    assert body["_router"]["selected_model"] == "openai/m1"
    assert body["_router"]["provider"] == "openai"
    assert body["_router"]["account"] == "acct-openai"
    assert body["_router"]["request_latency_ms"] == 123.457
    assert body["_router"]["ranked_models"] == ["openai/m1", "openai/m2"]
    assert body["_router"]["top_utility"] == 0.82


def test_to_fastapi_response_keeps_non_json_body_unchanged():
    account = BackendAccount(
        name="acct-openai",
        provider="openai",
        base_url="http://provider-openai",
        api_key="key-openai",
        models=["openai/m1"],
    )
    target = BackendTarget(
        account=account,
        account_name="acct-openai",
        provider="openai",
        base_url="http://provider-openai",
        model="openai/m1",
        auth_mode="api_key",
        organization=None,
        project=None,
        upstream_model="m1",
    )
    upstream = httpx.Response(
        status_code=200,
        headers={"Content-Type": "text/plain"},
        content=b"plain text body",
        request=httpx.Request("POST", "https://provider-openai/v1/responses"),
    )

    response = asyncio.run(
        BackendProxy._to_fastapi_response(
            upstream=upstream,
            stream=False,
            upstream_stream=False,
            target=target,
            attempted_targets=["acct-openai:m1"],
            attempted_upstream_models=["m1"],
            request_latency_ms=5.0,
            route_decision=_decision("openai/m1", []),
            request_id="req_diag_text",
            adapter="passthrough",
            audit_hook=None,
        )
    )

    assert response.body == b"plain text body"
    assert b"_router" not in response.body


def test_chat_completions_adapter_includes_router_diagnostics_in_json_response():
    account = BackendAccount(
        name="acct-codex",
        provider="openai-codex",
        base_url="http://provider-codex",
        api_key="key-codex",
        models=["openai-codex/gpt-5.2"],
    )
    target = BackendTarget(
        account=account,
        account_name="acct-codex",
        provider="openai-codex",
        base_url="http://provider-codex",
        model="openai-codex/gpt-5.2",
        auth_mode="api_key",
        organization=None,
        project=None,
        upstream_model="gpt-5.2",
    )
    sse_payload = "\n".join(
        [
            'data: {"type":"response.created","response":{"id":"resp_1","created_at":1700000000}}',
            'data: {"type":"response.output_text.done","text":"Hello from codex"}',
            'data: {"type":"response.completed"}',
            "data: [DONE]",
        ]
    )
    upstream = httpx.Response(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=sse_payload.encode("utf-8"),
        request=httpx.Request("POST", "https://provider-codex/codex/responses"),
    )

    response = asyncio.run(
        BackendProxy._to_fastapi_response(
            upstream=upstream,
            stream=False,
            upstream_stream=True,
            target=target,
            attempted_targets=["acct-codex:gpt-5.2"],
            attempted_upstream_models=["gpt-5.2"],
            request_latency_ms=42.3,
            route_decision=_decision("openai-codex/gpt-5.2", []),
            request_id="req_diag_chat",
            adapter="chat_completions",
            audit_hook=None,
        )
    )

    body = json.loads(response.body.decode("utf-8"))
    assert body["model"] == "openai-codex/gpt-5.2"
    assert body["choices"][0]["message"]["content"] == "Hello from codex"
    assert body["_router"]["request_id"] == "req_diag_chat"
    assert body["_router"]["provider"] == "openai-codex"
    assert body["_router"]["selected_model"] == "openai-codex/gpt-5.2"
