from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from open_llm_router.main import app
from open_llm_router.settings import get_settings


def _build_client(monkeypatch: Any, **env: Any) -> Any:
    monkeypatch.setenv("ROUTING_CONFIG_PATH", "router.profile.yaml")
    monkeypatch.setenv("INGRESS_AUTH_REQUIRED", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return TestClient(app)


def test_router_live_metrics_endpoint_returns_snapshot(monkeypatch: Any) -> None:
    with _build_client(monkeypatch) as client:
        response = client.get("/v1/router/live-metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "router.live_metrics"
    assert "dropped_events" in payload
    assert "queue_depth" in payload
    assert "proxy_retries_total" in payload
    assert "proxy_connect_latency_quantiles_by_target" in payload
    assert "proxy_connect_latency_slo_violations_total" in payload
    assert isinstance(payload["models"], dict)


def test_router_policy_endpoint_returns_runtime_profile_state(monkeypatch: Any) -> None:
    with _build_client(monkeypatch) as client:
        response = client.get("/v1/router/policy")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "router.policy"
    assert "updater" in payload
    assert "model_profiles" in payload
    assert isinstance(payload["model_profiles"], dict)


def test_metrics_endpoint_returns_prometheus_payload(monkeypatch: Any) -> None:
    with _build_client(monkeypatch) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "router_live_metrics_events_dropped_total" in body
    assert "router_proxy_retries_total" in body
    assert "router_proxy_connect_latency_slo_violations_total" in body
