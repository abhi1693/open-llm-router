from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
import yaml

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
    assert "classifier_calibration" in payload
    assert isinstance(payload["model_profiles"], dict)


def test_metrics_endpoint_returns_prometheus_payload(monkeypatch: Any) -> None:
    with _build_client(monkeypatch) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "router_live_metrics_events_dropped_total" in body
    assert "router_proxy_retries_total" in body
    assert "router_proxy_connect_latency_slo_violations_total" in body


def test_startup_prefetches_local_semantic_model_when_enabled(
    monkeypatch: Any, tmp_path: Any
) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_model": "openai-codex/gpt-5.2-codex",
                "task_routes": {
                    "general": {
                        "default": ["openai-codex/gpt-5.2-codex"],
                    }
                },
                "semantic_classifier": {
                    "enabled": True,
                    "backend": "local_embedding",
                    "local_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "local_files_only": False,
                    "local_max_length": 256,
                    "min_confidence": 0.2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, bool]] = []

    def _fake_load_local_embedding_runtime(
        *, model_name: str, local_files_only: bool
    ) -> tuple[str, str, str] | None:
        calls.append((model_name, local_files_only))
        return ("tokenizer", "model", "torch")

    monkeypatch.setattr(
        "open_llm_router.main._load_local_embedding_runtime",
        _fake_load_local_embedding_runtime,
    )

    with _build_client(monkeypatch, ROUTING_CONFIG_PATH=str(config_path)) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", False)]


def test_startup_prefetches_local_route_reranker_model_when_enabled(
    monkeypatch: Any, tmp_path: Any
) -> None:
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_model": "openai-codex/gpt-5.2-codex",
                "task_routes": {
                    "general": {
                        "default": ["openai-codex/gpt-5.2-codex"],
                    }
                },
                "route_reranker": {
                    "enabled": True,
                    "backend": "local_embedding",
                    "local_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "local_files_only": False,
                    "local_max_length": 256,
                    "similarity_weight": 0.35,
                    "min_similarity": 0.0,
                    "model_hints": {},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, bool]] = []

    def _fake_load_local_embedding_runtime(
        *, model_name: str, local_files_only: bool
    ) -> tuple[str, str, str] | None:
        calls.append((model_name, local_files_only))
        return ("tokenizer", "model", "torch")

    monkeypatch.setattr(
        "open_llm_router.main._load_local_embedding_runtime",
        _fake_load_local_embedding_runtime,
    )

    with _build_client(monkeypatch, ROUTING_CONFIG_PATH=str(config_path)) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", False)]
