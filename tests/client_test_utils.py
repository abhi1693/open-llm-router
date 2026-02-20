from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from open_llm_router.main import app
from open_llm_router.settings import get_settings

TEST_ROUTING_CONFIG_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "router.profile.yaml"
)


def set_default_test_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("ROUTING_CONFIG_PATH", str(TEST_ROUTING_CONFIG_PATH))


def build_test_client(monkeypatch: Any, **env: Any) -> TestClient:
    set_default_test_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return TestClient(app)
