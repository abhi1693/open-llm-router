from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from open_llm_router.persistence import YamlFileStore


def load_yaml_dict(
    path: str | Path,
    *,
    error_message: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, dict):
        return payload
    if error_message is not None:
        raise ValueError(error_message)
    raise ValueError(f"Expected YAML object in '{resolved}'.")


def write_yaml_dict(
    path: str | Path,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
) -> None:
    YamlFileStore(Path(path)).write(payload, sort_keys=sort_keys)
