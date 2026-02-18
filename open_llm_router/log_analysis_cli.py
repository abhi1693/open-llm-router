from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ParseState:
    total_objects: int = 0
    non_dict_objects: int = 0
    parse_errors: int = 0


@dataclass(slots=True)
class AggregationState:
    parse: ParseState = field(default_factory=ParseState)
    event_counts: Counter[str] = field(default_factory=Counter)
    min_ts: float | None = None
    max_ts: float | None = None

    selected_by_request_id: dict[str, str] = field(default_factory=dict)
    runtime_selected_by_request_id: dict[str, str] = field(default_factory=dict)
    route_task_by_request_id: dict[str, str] = field(default_factory=dict)
    route_complexity_by_request_id: dict[str, str] = field(default_factory=dict)
    used_model_by_request_id: dict[str, str] = field(default_factory=dict)
    first_attempt_target_by_request_id: dict[str, str] = field(default_factory=dict)

    routing_selected_counts: Counter[str] = field(default_factory=Counter)
    routing_task_counts: Counter[str] = field(default_factory=Counter)
    routing_complexity_counts: Counter[str] = field(default_factory=Counter)
    routing_stream_counts: Counter[str] = field(default_factory=Counter)
    routing_tools_counts: Counter[str] = field(default_factory=Counter)
    routing_selected_by_task: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    routing_selected_by_complexity: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    utility_margin_values: list[float] = field(default_factory=list)

    connect_ms_values: list[float] = field(default_factory=list)
    connect_ms_by_target: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    attempt_number_counts: Counter[str] = field(default_factory=Counter)
    response_status_counts: Counter[str] = field(default_factory=Counter)
    response_attempts_counts: Counter[str] = field(default_factory=Counter)
    response_attempts_gt1: int = 0

    request_error_counts: Counter[str] = field(default_factory=Counter)
    request_error_type_counts: Counter[str] = field(default_factory=Counter)
    request_error_timeout_counts: Counter[str] = field(default_factory=Counter)
    request_error_target_counts: Counter[str] = field(default_factory=Counter)

    finish_reason_counts: Counter[str] = field(default_factory=Counter)

    start_ts_by_request_id: dict[str, float] = field(default_factory=dict)
    response_ts_by_request_id: dict[str, float] = field(default_factory=dict)
    result_ts_by_request_id: dict[str, float] = field(default_factory=dict)


def _format_utc(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, UTC).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    idx = min(len(sorted_values) - 1, rank - 1)
    return sorted_values[idx]


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "avg": None,
        }
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def _coerce_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _tail_model_name(model: str | None) -> str:
    if not model:
        return ""
    model = model.strip()
    if not model:
        return ""
    if "/" in model:
        return model.rsplit("/", 1)[-1].strip()
    return model


def _target_model_name(target_label: str | None) -> str:
    if not target_label:
        return ""
    label = target_label.strip()
    if not label:
        return ""
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label


def _rough_model_match(selected_model: str | None, target_model: str | None) -> bool:
    selected_tail = _tail_model_name(selected_model)
    target_tail = _tail_model_name(target_model)
    if not selected_tail or not target_tail:
        return False
    if selected_tail == target_tail:
        return True
    return selected_tail in target_tail or target_tail in selected_tail


def _parse_objects(path: Path) -> tuple[list[dict[str, Any]], ParseState]:
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    idx = 0
    parse_state = ParseState()
    objects: list[dict[str, Any]] = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break

        try:
            obj, end_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            parse_state.parse_errors += 1
            next_start = text.find("{", idx + 1)
            if next_start == -1:
                break
            idx = next_start
            continue

        parse_state.total_objects += 1
        if isinstance(obj, dict):
            objects.append(obj)
        else:
            parse_state.non_dict_objects += 1
        idx = end_idx

    return objects, parse_state


def _update_latency_maps(
    state: AggregationState, event: dict[str, Any], event_name: str
) -> None:
    request_id = event.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        return

    ts = _safe_float(event.get("ts"))
    if ts is None:
        return

    if event_name == "proxy_start":
        state.start_ts_by_request_id[request_id] = ts
    elif event_name == "proxy_response":
        state.response_ts_by_request_id[request_id] = ts
    elif event_name == "proxy_chat_result":
        state.result_ts_by_request_id[request_id] = ts


def _process_event(state: AggregationState, event: dict[str, Any]) -> None:
    event_name = str(event.get("event") or "unknown")
    state.event_counts[event_name] += 1

    ts = _safe_float(event.get("ts"))
    if ts is not None:
        if state.min_ts is None or ts < state.min_ts:
            state.min_ts = ts
        if state.max_ts is None or ts > state.max_ts:
            state.max_ts = ts

    _update_latency_maps(state, event, event_name)

    request_id = event.get("request_id")
    if event_name == "route_decision":
        selected_model = str(event.get("selected_model") or "")
        task = str(event.get("task") or "unknown")
        complexity = str(event.get("complexity") or "unknown")

        if selected_model:
            state.routing_selected_counts[selected_model] += 1
        state.routing_task_counts[task] += 1
        state.routing_complexity_counts[complexity] += 1
        if selected_model:
            state.routing_selected_by_task[task][selected_model] += 1
            state.routing_selected_by_complexity[complexity][selected_model] += 1

        payload_summary = event.get("payload_summary")
        if isinstance(payload_summary, dict):
            stream_value = "true" if bool(payload_summary.get("stream")) else "false"
            has_tools_value = (
                "true" if bool(payload_summary.get("has_tools")) else "false"
            )
            state.routing_stream_counts[stream_value] += 1
            state.routing_tools_counts[has_tools_value] += 1

        if isinstance(request_id, str) and request_id.strip() and selected_model:
            state.selected_by_request_id[request_id] = selected_model
            state.route_task_by_request_id[request_id] = task
            state.route_complexity_by_request_id[request_id] = complexity

        candidate_scores = event.get("candidate_scores")
        if isinstance(candidate_scores, list):
            utilities = [
                _safe_float(item.get("utility"))
                for item in candidate_scores
                if isinstance(item, dict)
            ]
            normalized = sorted((u for u in utilities if u is not None), reverse=True)
            if len(normalized) >= 2:
                state.utility_margin_values.append(normalized[0] - normalized[1])

        return

    if event_name == "proxy_upstream_connected":
        connect_ms = _safe_float(event.get("connect_ms"))
        if connect_ms is not None:
            state.connect_ms_values.append(connect_ms)
            target = str(event.get("target") or "unknown")
            state.connect_ms_by_target[target].append(connect_ms)
        return

    if event_name == "proxy_attempt":
        attempt = _safe_int(event.get("attempt"))
        if attempt is None:
            state.attempt_number_counts["unknown"] += 1
        else:
            state.attempt_number_counts[str(attempt)] += 1

        if (
            isinstance(request_id, str)
            and request_id.strip()
            and request_id not in state.first_attempt_target_by_request_id
        ):
            target_label = str(event.get("target") or "")
            if target_label:
                state.first_attempt_target_by_request_id[request_id] = target_label
        return

    if event_name == "proxy_start":
        if isinstance(request_id, str) and request_id.strip():
            runtime_selected_model = event.get("selected_model")
            if (
                isinstance(runtime_selected_model, str)
                and runtime_selected_model.strip()
            ):
                state.runtime_selected_by_request_id[request_id] = (
                    runtime_selected_model
                )
        return

    if event_name == "proxy_response":
        status_code = _safe_int(event.get("status"))
        status_key = "unknown" if status_code is None else str(status_code)
        state.response_status_counts[status_key] += 1

        attempts = _safe_int(event.get("attempts"))
        attempts_key = "unknown" if attempts is None else str(attempts)
        state.response_attempts_counts[attempts_key] += 1
        if attempts is not None and attempts > 1:
            state.response_attempts_gt1 += 1

        model_used = event.get("model")
        if (
            isinstance(request_id, str)
            and request_id.strip()
            and isinstance(model_used, str)
        ):
            state.used_model_by_request_id[request_id] = model_used
        return

    if event_name == "proxy_request_error":
        target = str(event.get("target") or "unknown")
        upstream_model = str(event.get("upstream_model") or "unknown")
        status_code = _safe_int(event.get("status_code"))
        status = "none" if status_code is None else str(status_code)
        error_type = str(event.get("error_type") or "unknown")
        is_timeout = bool(event.get("is_timeout"))
        error_message = str(event.get("error") or "")
        if not error_message:
            error_message = "<empty>"
        key = (
            f"target={target} upstream_model={upstream_model} "
            f"status={status} error_type={error_type} error={error_message}"
        )
        state.request_error_counts[key] += 1
        state.request_error_type_counts[error_type] += 1
        state.request_error_timeout_counts["true" if is_timeout else "false"] += 1
        state.request_error_target_counts[target] += 1
        return

    if event_name == "proxy_chat_result":
        finish_reason = str(event.get("finish_reason") or "unknown")
        state.finish_reason_counts[finish_reason] += 1


def _build_duration_stats(
    state: AggregationState,
) -> dict[str, dict[str, float | int | None]]:
    start_to_response: list[float] = []
    response_to_result: list[float] = []
    start_to_result: list[float] = []

    for request_id, start_ts in state.start_ts_by_request_id.items():
        response_ts = state.response_ts_by_request_id.get(request_id)
        if response_ts is not None and response_ts >= start_ts:
            start_to_response.append(response_ts - start_ts)

        result_ts = state.result_ts_by_request_id.get(request_id)
        if result_ts is not None and result_ts >= start_ts:
            start_to_result.append(result_ts - start_ts)
        if (
            response_ts is not None
            and result_ts is not None
            and result_ts >= response_ts
        ):
            response_to_result.append(result_ts - response_ts)

    return {
        "proxy_start_to_response_seconds": _summary_stats(start_to_response),
        "proxy_response_to_result_seconds": _summary_stats(response_to_result),
        "proxy_start_to_result_seconds": _summary_stats(start_to_result),
    }


def _build_consistency_summary(
    state: AggregationState, *, top_n: int
) -> dict[str, Any]:
    matched = 0
    mismatched = 0
    missing = 0

    attempt_match = 0
    attempt_mismatch = 0
    attempt_missing = 0
    selected_used_mismatch_pairs: Counter[str] = Counter()
    selected_target_mismatch_pairs: Counter[str] = Counter()

    for request_id, selected_model in state.selected_by_request_id.items():
        effective_selected_model = (
            state.runtime_selected_by_request_id.get(request_id) or selected_model
        )
        used_model = state.used_model_by_request_id.get(request_id)
        if used_model is None:
            missing += 1
        elif effective_selected_model == used_model:
            matched += 1
        else:
            mismatched += 1
            pair = f"selected={effective_selected_model} used={used_model}"
            selected_used_mismatch_pairs[pair] += 1

        first_attempt_target = state.first_attempt_target_by_request_id.get(request_id)
        if first_attempt_target is None:
            attempt_missing += 1
        else:
            target_model = _target_model_name(first_attempt_target)
            if _rough_model_match(effective_selected_model, target_model):
                attempt_match += 1
            else:
                attempt_mismatch += 1
                pair = (
                    f"selected={effective_selected_model} "
                    f"first_target={first_attempt_target}"
                )
                selected_target_mismatch_pairs[pair] += 1

    return {
        "selected_vs_used_model": {
            "matched": matched,
            "mismatched": mismatched,
            "missing_response_model": missing,
            "top_mismatched_pairs": {
                key: int(count)
                for key, count in selected_used_mismatch_pairs.most_common(top_n)
            },
        },
        "selected_vs_first_attempt_target": {
            "rough_match": attempt_match,
            "rough_mismatch": attempt_mismatch,
            "missing_attempt": attempt_missing,
            "top_rough_mismatched_pairs": {
                key: int(count)
                for key, count in selected_target_mismatch_pairs.most_common(top_n)
            },
        },
    }


def _build_summary(
    path: Path, state: AggregationState, *, top_n: int
) -> dict[str, Any]:
    connect_by_target: dict[str, dict[str, float | int | None]] = {}
    for target, values in sorted(
        state.connect_ms_by_target.items(), key=lambda item: len(item[1]), reverse=True
    ):
        connect_by_target[target] = _summary_stats(values)

    by_task: dict[str, dict[str, int]] = {}
    for task, counter in sorted(state.routing_selected_by_task.items()):
        by_task[task] = {
            model: int(count) for model, count in counter.most_common(top_n)
        }

    by_complexity: dict[str, dict[str, int]] = {}
    for complexity, counter in sorted(state.routing_selected_by_complexity.items()):
        by_complexity[complexity] = {
            model: int(count) for model, count in counter.most_common(top_n)
        }

    return {
        "file": str(path),
        "parse": {
            "total_objects": state.parse.total_objects,
            "parse_errors": state.parse.parse_errors,
            "non_dict_objects": state.parse.non_dict_objects,
        },
        "time_range": {
            "start_epoch": state.min_ts,
            "end_epoch": state.max_ts,
            "start_utc": _format_utc(state.min_ts),
            "end_utc": _format_utc(state.max_ts),
        },
        "events": {
            "counts": _coerce_counter(state.event_counts),
        },
        "routing": {
            "total_decisions": int(state.event_counts.get("route_decision", 0)),
            "selected_models": {
                model: int(count)
                for model, count in state.routing_selected_counts.most_common(top_n)
            },
            "task_counts": _coerce_counter(state.routing_task_counts),
            "complexity_counts": _coerce_counter(state.routing_complexity_counts),
            "stream_counts": _coerce_counter(state.routing_stream_counts),
            "has_tools_counts": _coerce_counter(state.routing_tools_counts),
            "selected_models_by_task": by_task,
            "selected_models_by_complexity": by_complexity,
            "utility_margin": _summary_stats(state.utility_margin_values),
        },
        "latency": {
            "connect_ms": _summary_stats(state.connect_ms_values),
            "connect_ms_by_target": connect_by_target,
            "request_durations": _build_duration_stats(state),
        },
        "retries_and_errors": {
            "attempt_number_counts": _coerce_counter(state.attempt_number_counts),
            "response_status_counts": _coerce_counter(state.response_status_counts),
            "response_attempts_counts": _coerce_counter(state.response_attempts_counts),
            "responses_with_attempts_gt_1": state.response_attempts_gt1,
            "proxy_request_errors": {
                "total": int(state.event_counts.get("proxy_request_error", 0)),
                "by_target_status_error": {
                    key: int(count)
                    for key, count in state.request_error_counts.most_common(top_n)
                },
                "by_error_type": {
                    key: int(count)
                    for key, count in state.request_error_type_counts.most_common(top_n)
                },
                "timeouts": _coerce_counter(state.request_error_timeout_counts),
                "by_target": {
                    key: int(count)
                    for key, count in state.request_error_target_counts.most_common(
                        top_n
                    )
                },
            },
            "finish_reason_counts": _coerce_counter(state.finish_reason_counts),
        },
        "consistency": _build_consistency_summary(state, top_n=top_n),
    }


def summarize_log(path: Path, *, top_n: int = 8) -> dict[str, Any]:
    events, parse_state = _parse_objects(path)
    state = AggregationState(parse=parse_state)
    for event in events:
        _process_event(state, event)
    return _build_summary(path=path, state=state, top_n=max(1, top_n))


def _format_value(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.{digits}f}"
    return str(value)


def _render_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append(f"Log file: {summary['file']}")
    lines.append(
        "Window: "
        f"{summary['time_range']['start_utc'] or 'n/a'} -> {summary['time_range']['end_utc'] or 'n/a'}"
    )

    parse = summary["parse"]
    lines.append(
        "Parsed objects: "
        f"{parse['total_objects']} (errors={parse['parse_errors']}, non_dict={parse['non_dict_objects']})"
    )

    lines.append("")
    lines.append("Event counts:")
    for event, count in sorted(summary["events"]["counts"].items()):
        lines.append(f"  - {event}: {count}")

    routing = summary["routing"]
    lines.append("")
    lines.append(f"Routing decisions: {routing['total_decisions']}")
    lines.append("Selected models:")
    for model, count in routing["selected_models"].items():
        lines.append(f"  - {model}: {count}")
    lines.append(f"Task counts: {routing['task_counts']}")
    lines.append(f"Complexity counts: {routing['complexity_counts']}")
    lines.append(f"Stream counts: {routing['stream_counts']}")
    lines.append(f"Has-tools counts: {routing['has_tools_counts']}")

    utility_margin = routing["utility_margin"]
    lines.append(
        "Utility margin (top1-top2): "
        f"p50={_format_value(utility_margin['p50'], 4)} "
        f"p95={_format_value(utility_margin['p95'], 4)} "
        f"min={_format_value(utility_margin['min'], 4)} "
        f"max={_format_value(utility_margin['max'], 4)}"
    )

    latency = summary["latency"]
    connect = latency["connect_ms"]
    lines.append("")
    lines.append(
        "Connect latency (ms): "
        f"n={connect['count']} "
        f"p50={_format_value(connect['p50'])} "
        f"p95={_format_value(connect['p95'])} "
        f"p99={_format_value(connect['p99'])} "
        f"max={_format_value(connect['max'])}"
    )

    durations = latency["request_durations"]
    start_to_resp = durations["proxy_start_to_response_seconds"]
    resp_to_result = durations["proxy_response_to_result_seconds"]
    lines.append(
        "Proxy start->response (s): "
        f"n={start_to_resp['count']} p50={_format_value(start_to_resp['p50'])} "
        f"p95={_format_value(start_to_resp['p95'])} p99={_format_value(start_to_resp['p99'])} "
        f"max={_format_value(start_to_resp['max'])}"
    )
    lines.append(
        "Proxy response->chat_result (s): "
        f"n={resp_to_result['count']} p50={_format_value(resp_to_result['p50'])} "
        f"p95={_format_value(resp_to_result['p95'])} p99={_format_value(resp_to_result['p99'])} "
        f"max={_format_value(resp_to_result['max'])}"
    )

    retries = summary["retries_and_errors"]
    lines.append("")
    lines.append(f"Attempt number counts: {retries['attempt_number_counts']}")
    lines.append(f"Response status counts: {retries['response_status_counts']}")
    lines.append(f"Response attempts counts: {retries['response_attempts_counts']}")
    lines.append(
        f"Responses with retries (>1 attempt): {retries['responses_with_attempts_gt_1']}"
    )
    lines.append(f"Finish reasons: {retries['finish_reason_counts']}")
    proxy_errors = retries["proxy_request_errors"]
    lines.append(
        f"Proxy request errors: total={proxy_errors['total']} timeouts={proxy_errors['timeouts']}"
    )
    if proxy_errors["by_error_type"]:
        lines.append("Top error types:")
        for error_type, count in proxy_errors["by_error_type"].items():
            lines.append(f"  - {error_type}: {count}")
    if proxy_errors["by_target"]:
        lines.append("Top error targets:")
        for target, count in proxy_errors["by_target"].items():
            lines.append(f"  - {target}: {count}")

    consistency = summary["consistency"]
    lines.append("")
    lines.append("Consistency checks:")
    selected_vs_used = consistency["selected_vs_used_model"]
    lines.append(f"  - selected_vs_used_model: {selected_vs_used}")
    if selected_vs_used["top_mismatched_pairs"]:
        lines.append("    top mismatched selected->used pairs:")
        for pair, count in selected_vs_used["top_mismatched_pairs"].items():
            lines.append(f"      - {pair}: {count}")
    selected_vs_target = consistency["selected_vs_first_attempt_target"]
    lines.append(f"  - selected_vs_first_attempt_target: {selected_vs_target}")
    if selected_vs_target["top_rough_mismatched_pairs"]:
        lines.append("    top mismatched selected->first-target pairs:")
        for pair, count in selected_vs_target["top_rough_mismatched_pairs"].items():
            lines.append(f"      - {pair}: {count}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-log-analyze",
        description="Aggregate router audit logs for performance analysis.",
    )
    parser.add_argument(
        "--log-path",
        default="logs/router_decisions.jsonl",
        help="Path to the audit log file (default: logs/router_decisions.jsonl)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Top-N entries to include for model/target/error breakdowns",
    )
    parser.add_argument(
        "--output",
        help="Optional file path to write the report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_path = Path(args.log_path)
    if not log_path.exists():
        parser.error(f"Log file not found: {log_path}")

    summary = summarize_log(path=log_path, top_n=max(1, int(args.top)))

    if args.format == "json":
        rendered = json.dumps(summary, ensure_ascii=True, indent=2)
    else:
        rendered = _render_text(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
