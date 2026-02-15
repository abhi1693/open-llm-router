from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
import time
from typing import Any


class JsonlAuditLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._lock = Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        record = {"ts": int(time.time()), **event}
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def close(self) -> None:
        # File is opened per-write, so there is nothing to flush/close here.
        return None
