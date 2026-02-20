from __future__ import annotations

import json
import time
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread
from typing import Any


class JsonlAuditLogger:
    def __init__(
        self,
        path: str,
        enabled: bool = True,
        max_queue_size: int = 8192,
    ) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._lock = Lock()
        self._queue: Queue[str | None] | None = None
        self._worker: Thread | None = None
        self._dropped_records = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._queue = Queue(maxsize=max_queue_size)
            self._worker = Thread(
                target=self._drain_queue, name="router-audit-writer", daemon=True
            )
            self._worker.start()

    def log(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        record = {"ts": int(time.time()), **event}
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":"), default=str)
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(line)
        except Full:
            with self._lock:
                self._dropped_records += 1

    def close(self) -> None:
        if not self.enabled:
            return None
        queue = self._queue
        worker = self._worker
        if queue is None or worker is None:
            return None
        queue.put(None)
        worker.join(timeout=2.0)
        return None

    def _drain_queue(self) -> None:
        queue = self._queue
        if queue is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            while True:
                item = queue.get()
                if item is None:
                    queue.task_done()
                    break
                handle.write(item + "\n")
                handle.flush()
                queue.task_done()
            dropped = 0
            with self._lock:
                dropped = self._dropped_records
                self._dropped_records = 0
            if dropped > 0:
                fallback_record = {
                    "ts": int(time.time()),
                    "event": "audit_logger_dropped_records",
                    "dropped_count": dropped,
                }
                handle.write(
                    json.dumps(
                        fallback_record,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )
                handle.flush()
