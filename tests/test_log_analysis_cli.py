from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from open_llm_router.cli.log_analysis_cli import main, summarize_log


def _write_log(
    path: Path,
    objects: list[dict[str, Any]],
    *,
    pretty_first: bool = True,
) -> None:
    parts: list[str] = []
    for idx, obj in enumerate(objects):
        if idx == 0 and pretty_first:
            parts.append(json.dumps(obj, indent=2))
        else:
            parts.append(json.dumps(obj, separators=(",", ":")))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def test_summarize_log_aggregates_performance_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "router_decisions.jsonl"
    _write_log(
        log_path,
        [
            {
                "ts": 100,
                "event": "route_decision",
                "request_id": "r1",
                "selected_model": "openai-codex/gpt-5.2-codex",
                "task": "coding",
                "complexity": "high",
                "payload_summary": {"stream": True, "has_tools": True},
                "candidate_scores": [
                    {"model": "openai-codex/gpt-5.2-codex", "utility": 0.50},
                    {"model": "openai-codex/gpt-5.2", "utility": 0.45},
                ],
            },
            {"ts": 101, "event": "proxy_start", "request_id": "r1"},
            {
                "ts": 101,
                "event": "proxy_attempt",
                "request_id": "r1",
                "attempt": 1,
                "target": "acct-a:gpt-5.2-codex",
            },
            {
                "ts": 102,
                "event": "proxy_upstream_connected",
                "request_id": "r1",
                "target": "acct-a:gpt-5.2-codex",
                "connect_ms": 300,
            },
            {
                "ts": 103,
                "event": "proxy_response",
                "request_id": "r1",
                "model": "openai-codex/gpt-5.2-codex",
                "status": 200,
                "attempts": 1,
            },
            {
                "ts": 106,
                "event": "proxy_chat_result",
                "request_id": "r1",
                "finish_reason": "stop",
            },
            {
                "ts": 110,
                "event": "route_decision",
                "request_id": "r2",
                "selected_model": "gemini/gemini-2.5-flash-lite",
                "task": "general",
                "complexity": "low",
                "payload_summary": {"stream": False, "has_tools": False},
                "candidate_scores": [
                    {"model": "gemini/gemini-2.5-flash-lite", "utility": 0.20},
                    {"model": "gemini/gemini-2.5-flash", "utility": 0.19},
                ],
            },
            {"ts": 110, "event": "proxy_start", "request_id": "r2"},
            {
                "ts": 110,
                "event": "proxy_attempt",
                "request_id": "r2",
                "attempt": 1,
                "target": "acct-b:gpt-5.2-codex",
            },
            {
                "ts": 111,
                "event": "proxy_request_error",
                "request_id": "r2",
                "target": "acct-b:gpt-5.2-codex",
                "upstream_model": "gpt-5.2-codex",
                "error_type": "ConnectTimeout",
                "is_timeout": True,
                "error": "timeout",
            },
            {
                "ts": 111,
                "event": "proxy_attempt",
                "request_id": "r2",
                "attempt": 2,
                "target": "acct-c:gemini-2.5-flash-lite",
            },
            {
                "ts": 112,
                "event": "proxy_response",
                "request_id": "r2",
                "model": "gemini/gemini-2.5-flash-lite",
                "status": 200,
                "attempts": 2,
            },
        ],
    )

    summary = summarize_log(log_path, top_n=5)

    assert summary["parse"]["parse_errors"] == 0
    assert summary["events"]["counts"]["route_decision"] == 2
    assert summary["routing"]["total_decisions"] == 2

    assert summary["routing"]["task_counts"]["coding"] == 1
    assert summary["routing"]["task_counts"]["general"] == 1
    assert summary["routing"]["complexity_counts"]["high"] == 1
    assert summary["routing"]["complexity_counts"]["low"] == 1

    assert summary["latency"]["connect_ms"]["count"] == 1
    assert summary["latency"]["connect_ms"]["p50"] == 300.0

    start_to_response = summary["latency"]["request_durations"][
        "proxy_start_to_response_seconds"
    ]
    assert start_to_response["count"] == 2
    assert start_to_response["p50"] == 2.0

    response_to_result = summary["latency"]["request_durations"][
        "proxy_response_to_result_seconds"
    ]
    assert response_to_result["count"] == 1
    assert response_to_result["p50"] == 3.0

    assert summary["retries_and_errors"]["responses_with_attempts_gt_1"] == 1
    assert summary["retries_and_errors"]["proxy_request_errors"]["total"] == 1
    assert summary["retries_and_errors"]["proxy_request_errors"]["by_error_type"] == {
        "ConnectTimeout": 1,
    }
    assert summary["retries_and_errors"]["proxy_request_errors"]["timeouts"] == {
        "true": 1,
    }
    assert summary["retries_and_errors"]["proxy_request_errors"]["by_target"] == {
        "acct-b:gpt-5.2-codex": 1,
    }

    selected_vs_used = summary["consistency"]["selected_vs_used_model"]
    assert selected_vs_used["matched"] == 2
    assert selected_vs_used["mismatched"] == 0
    assert selected_vs_used["top_mismatched_pairs"] == {}

    selected_vs_target = summary["consistency"]["selected_vs_first_attempt_target"]
    assert selected_vs_target["rough_match"] == 1
    assert selected_vs_target["rough_mismatch"] == 1
    assert selected_vs_target["top_rough_mismatched_pairs"] == {
        "selected=gemini/gemini-2.5-flash-lite first_target=acct-b:gpt-5.2-codex": 1,
    }


def test_main_writes_json_report(tmp_path: Path) -> None:
    log_path = tmp_path / "router_decisions.jsonl"
    _write_log(
        log_path,
        [
            {
                "ts": 100,
                "event": "route_decision",
                "request_id": "r1",
                "selected_model": "gemini/gemini-2.5-flash-lite",
                "task": "general",
                "complexity": "low",
            },
        ],
    )

    output_path = tmp_path / "summary.json"
    rc = main(
        [
            "--log-path",
            str(log_path),
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )
    assert rc == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["routing"]["total_decisions"] == 1
    assert payload["routing"]["selected_models"]["gemini/gemini-2.5-flash-lite"] == 1


def test_consistency_prefers_runtime_selected_model_from_proxy_start(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "router_decisions.jsonl"
    _write_log(
        log_path,
        [
            {
                "ts": 100,
                "event": "route_decision",
                "request_id": "r1",
                "selected_model": "openai/codex-1",
                "task": "coding",
                "complexity": "xhigh",
            },
            {
                "ts": 101,
                "event": "proxy_start",
                "request_id": "r1",
                "selected_model": "openai-codex/gpt-5.2-codex",
            },
            {
                "ts": 101,
                "event": "proxy_attempt",
                "request_id": "r1",
                "attempt": 1,
                "target": "acct-a:gpt-5.2-codex",
            },
            {
                "ts": 102,
                "event": "proxy_response",
                "request_id": "r1",
                "model": "openai-codex/gpt-5.2-codex",
                "status": 200,
                "attempts": 1,
            },
        ],
    )

    summary = summarize_log(log_path, top_n=5)
    selected_vs_used = summary["consistency"]["selected_vs_used_model"]
    selected_vs_target = summary["consistency"]["selected_vs_first_attempt_target"]

    assert selected_vs_used["matched"] == 1
    assert selected_vs_used["mismatched"] == 0
    assert selected_vs_used["top_mismatched_pairs"] == {}

    assert selected_vs_target["rough_match"] == 1
    assert selected_vs_target["rough_mismatch"] == 0
    assert selected_vs_target["top_rough_mismatched_pairs"] == {}
