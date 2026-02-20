from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from open_llm_router.config import load_routing_config_with_metadata
from open_llm_router.router_engine import SmartModelRouter
from open_llm_router.utils.cli_output import write_cli_report


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_no}: each line must be a JSON object")
            rows.append(parsed)
    return rows


def _normalize_case_payload(case: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if "payload" in case:
        payload = case.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Case field 'payload' must be an object when provided.")
    else:
        payload = {
            key: value
            for key, value in case.items()
            if key not in {"endpoint", "expected_model", "expected_task", "id"}
        }
    endpoint = str(case.get("endpoint") or "/v1/chat/completions").strip()
    if not endpoint:
        endpoint = "/v1/chat/completions"
    return payload, endpoint


def _evaluate(
    *,
    router: SmartModelRouter,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        payload, endpoint = _normalize_case_payload(case)
        decision = router.decide(payload=payload, endpoint=endpoint)
        results.append(
            {
                "index": index,
                "id": case.get("id"),
                "expected_model": case.get("expected_model"),
                "expected_task": case.get("expected_task"),
                "selected_model": decision.selected_model,
                "task": decision.task,
                "complexity": decision.complexity,
            }
        )
    return results


def _summarize(
    *,
    baseline: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(baseline) != len(reranked):
        raise ValueError("Baseline and reranked result counts do not match.")
    total = len(baseline)
    changed = 0
    expected_model_cases = 0
    expected_model_baseline_hits = 0
    expected_model_reranked_hits = 0
    expected_task_cases = 0
    expected_task_baseline_hits = 0
    expected_task_reranked_hits = 0
    examples: list[dict[str, Any]] = []

    for base, rerank in zip(baseline, reranked):
        if base["selected_model"] != rerank["selected_model"]:
            changed += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "index": base["index"],
                        "id": base.get("id"),
                        "baseline_model": base["selected_model"],
                        "reranked_model": rerank["selected_model"],
                        "task": rerank["task"],
                        "complexity": rerank["complexity"],
                    }
                )

        expected_model = base.get("expected_model")
        if isinstance(expected_model, str) and expected_model.strip():
            expected_model_cases += 1
            if base["selected_model"] == expected_model:
                expected_model_baseline_hits += 1
            if rerank["selected_model"] == expected_model:
                expected_model_reranked_hits += 1

        expected_task = base.get("expected_task")
        if isinstance(expected_task, str) and expected_task.strip():
            expected_task_cases += 1
            if base["task"] == expected_task:
                expected_task_baseline_hits += 1
            if rerank["task"] == expected_task:
                expected_task_reranked_hits += 1

    return {
        "total_cases": total,
        "selection_changed_cases": changed,
        "selection_changed_rate": round((changed / total) if total else 0.0, 6),
        "expected_model_cases": expected_model_cases,
        "expected_model_baseline_hits": expected_model_baseline_hits,
        "expected_model_reranked_hits": expected_model_reranked_hits,
        "expected_task_cases": expected_task_cases,
        "expected_task_baseline_hits": expected_task_baseline_hits,
        "expected_task_reranked_hits": expected_task_reranked_hits,
        "sample_selection_changes": examples,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-benchmark-reranker",
        description=(
            "Compare routing outcomes with route reranker disabled vs enabled "
            "against a JSONL case file."
        ),
    )
    parser.add_argument(
        "--config",
        default="router.profile.yaml",
        help="Routing config path (profile or effective schema).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "JSONL benchmark cases. "
            "Each row supports keys: payload, endpoint, expected_model, expected_task, id."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path. Writes JSON summary when provided.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config, _ = load_routing_config_with_metadata(args.config)
    if not config.route_reranker.enabled:
        raise ValueError(
            "route_reranker is disabled in config. Enable it before benchmarking."
        )

    dataset_path = Path(args.dataset)
    cases = _load_jsonl(dataset_path)
    if not cases:
        raise ValueError("Dataset is empty.")

    baseline_config = config.model_copy(deep=True)
    baseline_config.route_reranker.enabled = False

    baseline_router = SmartModelRouter(baseline_config)
    reranked_router = SmartModelRouter(config)

    baseline = _evaluate(router=baseline_router, cases=cases)
    reranked = _evaluate(router=reranked_router, cases=cases)
    summary = _summarize(baseline=baseline, reranked=reranked)

    output = {
        "config": args.config,
        "dataset": str(dataset_path),
        "reranker": {
            "enabled": True,
            "backend": config.route_reranker.backend,
            "model": config.route_reranker.local_model_name,
            "similarity_weight": config.route_reranker.similarity_weight,
            "min_similarity": config.route_reranker.min_similarity,
        },
        "summary": summary,
    }
    rendered = json.dumps(output, indent=2)
    write_cli_report(rendered=rendered, output_path=args.output, always_print=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
