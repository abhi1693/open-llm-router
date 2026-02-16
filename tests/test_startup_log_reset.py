from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from open_llm_router.main import app
from open_llm_router.settings import get_settings


def _build_client(monkeypatch, **env):
    monkeypatch.setenv("ROUTING_CONFIG_PATH", "router.profile.yaml")
    monkeypatch.setenv("INGRESS_AUTH_REQUIRED", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return TestClient(app)


def test_startup_clears_all_files_in_configured_logs_directory(
    monkeypatch, tmp_path: Path
) -> None:
    logs_dir = tmp_path / "logs"
    nested_dir = logs_dir / "archive"
    nested_dir.mkdir(parents=True, exist_ok=True)

    audit_path = logs_dir / "router_decisions.jsonl"
    overrides_path = logs_dir / "router.runtime.overrides.yaml"
    extra_json = logs_dir / "perf-summary.json"
    nested_old_log = nested_dir / "old.log"

    audit_path.write_text("old audit content\n", encoding="utf-8")
    overrides_path.write_text("old overrides\n", encoding="utf-8")
    extra_json.write_text('{"stale": true}\n', encoding="utf-8")
    nested_old_log.write_text("old nested log\n", encoding="utf-8")

    with _build_client(
        monkeypatch,
        ROUTER_AUDIT_LOG_PATH=str(audit_path),
        ROUTER_RUNTIME_OVERRIDES_PATH=str(overrides_path),
    ) as client:
        response = client.get("/health")
        assert response.status_code == 200

    # Startup should have cleared all existing files in logs/ before serving.
    assert audit_path.exists()
    assert audit_path.read_text(encoding="utf-8") == ""
    assert overrides_path.exists()
    overrides_body = overrides_path.read_text(encoding="utf-8")
    assert "old overrides" not in overrides_body
    assert not extra_json.exists()
    assert not nested_old_log.exists()
