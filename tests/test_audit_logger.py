from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from open_llm_router.gateway.audit import JsonlAuditLogger
from open_llm_router.server.main import _sanitize_audit_event


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


def test_sanitize_audit_event_redacts_text_previews() -> None:
    event: dict[str, Any] = {
        "event": "route_decision",
        "signals": {
            "text_preview": "my private prompt",
            "text_preview_total": "full private prompt",
            "task": "general",
        },
        "tool_call_summary": [
            {
                "name": "search",
                "arguments_preview": '{"q":"secret term"}',
            }
        ],
    }

    sanitized = _sanitize_audit_event(event)

    assert sanitized["signals"]["text_preview"] == "[redacted]"
    assert sanitized["signals"]["text_preview_total"] == "[redacted]"
    assert sanitized["signals"]["task"] == "general"
    tool_call_summary = sanitized["tool_call_summary"]
    assert isinstance(tool_call_summary, list)
    assert tool_call_summary[0]["arguments_preview"] == "[redacted]"
    assert event["signals"]["text_preview"] == "my private prompt"
