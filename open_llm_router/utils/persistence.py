from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


class YamlFileStore:
    """Shared YAML persistence helper with atomic write support."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, *, default: Any = None) -> Any:
        if not self.path.exists():
            return default
        with self.path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if payload is None:
            return default
        return payload

    def write(
        self,
        payload: Any,
        *,
        sort_keys: bool = False,
        atomic: bool = True,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not atomic:
            with self.path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=sort_keys)
            return

        temp_path = self._temp_path()
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=sort_keys)
            temp_path.replace(self.path)
        except Exception:
            with contextlib.suppress(Exception):
                temp_path.unlink(missing_ok=True)
            raise

    def _temp_path(self) -> Path:
        token = uuid4().hex
        return self.path.with_name(f".{self.path.name}.{token}.tmp")
