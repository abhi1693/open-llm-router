from __future__ import annotations

import json
import time
from pathlib import Path

from open_llm_router.audit import JsonlAuditLogger


def test_audit_logger_writes_records_before_close(tmp_path: Path) -> None:
    log_path = tmp_path / "router_decisions.jsonl"
    logger = JsonlAuditLogger(path=str(log_path), enabled=True)
    try:
        logger.log({"event": "route_decision", "request_id": "req-1"})

        deadline = time.time() + 1.0
        content = ""
        while time.time() < deadline:
            if log_path.exists():
                content = log_path.read_text(encoding="utf-8")
                if content.strip():
                    break
            time.sleep(0.02)

        assert content.strip()
        payload = json.loads(content.strip().splitlines()[0])
        assert payload["event"] == "route_decision"
        assert payload["request_id"] == "req-1"
    finally:
        logger.close()
