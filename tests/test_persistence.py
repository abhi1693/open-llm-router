from __future__ import annotations

from typing import TYPE_CHECKING

from open_llm_router.utils.persistence import YamlFileStore

if TYPE_CHECKING:
    from pathlib import Path


def test_yaml_file_store_load_returns_default_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    store = YamlFileStore(path)
    payload = store.load(default={"ok": True})
    assert payload == {"ok": True}


def test_yaml_file_store_write_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    store = YamlFileStore(path)
    original = {"default_model": "openai/gpt-5.1", "accounts": [{"name": "main"}]}
    store.write(original, sort_keys=False)

    loaded = store.load(default={})
    assert loaded == original
    assert path.exists()
