from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_llm_router.catalog import CatalogValidationError
from open_llm_router.config import load_routing_config


def _write(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_load_routing_config_rejects_unknown_model(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "openai-codex/not-a-real-model",
            "task_routes": {"general": {"default": ["openai-codex/not-a-real-model"]}},
        },
    )

    with pytest.raises(CatalogValidationError) as exc:
        load_routing_config(str(config_path))

    message = str(exc.value)
    assert "default_model" in message
    assert "Suggested canonical ids" in message


def test_load_routing_config_rejects_unknown_provider(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "openai-codex/gpt-5.2",
            "task_routes": {"general": {"default": ["openai-codex/gpt-5.2"]}},
            "accounts": [
                {
                    "name": "bad-account",
                    "provider": "not-supported",
                    "base_url": "https://example.com",
                }
            ],
        },
    )

    with pytest.raises(CatalogValidationError) as exc:
        load_routing_config(str(config_path))

    message = str(exc.value)
    assert "accounts[0].provider" in message
    assert "Suggested canonical ids" in message
