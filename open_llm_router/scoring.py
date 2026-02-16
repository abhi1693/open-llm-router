from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from open_llm_router.config import LearnedRoutingConfig, ModelProfile

COMPLEXITY_SCORES = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.8,
    "xhigh": 1.0,
}


@dataclass(slots=True)
class ModelScore:
    model: str
    quality_probability: float
    utility: float
    estimated_cost: float
    estimated_latency_ms: float
    estimated_failure_rate: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "model": self.model,
            "quality_probability": round(self.quality_probability, 6),
            "utility": round(self.utility, 6),
            "estimated_cost": round(self.estimated_cost, 6),
            "estimated_latency_ms": round(self.estimated_latency_ms, 3),
            "estimated_failure_rate": round(self.estimated_failure_rate, 6),
        }


def build_routing_features(
    task: str,
    complexity: str,
    signals: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, float]:
    text_length = float(signals.get("text_length", 0))
    code_score = float(signals.get("code_score", 0))
    think_score = float(signals.get("think_score", 0))
    instruction_score = float(signals.get("instruction_score", 0))
    has_image = 1.0 if signals.get("has_image", False) else 0.0
    reasoning_effort = str(signals.get("reasoning_effort") or "").lower()

    max_tokens = payload.get("max_output_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = 0

    return {
        "text_kchars": text_length / 1000.0,
        "code_score": code_score,
        "think_score": think_score,
        "instruction_score": instruction_score,
        "has_image": has_image,
        "complexity_score": COMPLEXITY_SCORES.get(complexity, 0.5),
        "max_output_k_tokens": float(max_tokens) / 1000.0,
        "reasoning_effort_high": 1.0 if reasoning_effort in {"high", "xhigh", "max"} else 0.0,
        "task_coding": 1.0 if task == "coding" else 0.0,
        "task_thinking": 1.0 if task == "thinking" else 0.0,
        "task_instruction_following": 1.0 if task == "instruction_following" else 0.0,
        "task_general": 1.0 if task == "general" else 0.0,
        "task_image": 1.0 if task == "image" else 0.0,
    }


def score_model(
    model: str,
    profile: ModelProfile,
    features: dict[str, float],
    payload: dict[str, Any],
    signals: dict[str, Any],
    learned_cfg: LearnedRoutingConfig,
) -> ModelScore:
    weighted_feature_sum = 0.0
    for name, value in features.items():
        weighted_feature_sum += learned_cfg.feature_weights.get(name, 0.0) * value

    z = learned_cfg.bias + profile.quality_bias + profile.quality_sensitivity * weighted_feature_sum
    quality_probability = _sigmoid(z)

    input_char_count = float(
        signals.get("text_length_total", signals.get("text_length", 0))
    )
    input_tokens = max(1.0, input_char_count / 4.0)
    max_output_tokens = payload.get("max_output_tokens")
    if max_output_tokens is None:
        max_output_tokens = payload.get("max_tokens")
    if not isinstance(max_output_tokens, (int, float)) or max_output_tokens <= 0:
        max_output_tokens = float(learned_cfg.default_output_tokens)

    estimated_cost = (
        (input_tokens / 1000.0) * profile.cost_input_per_1k
        + (float(max_output_tokens) / 1000.0) * profile.cost_output_per_1k
    )

    utility_penalty = (
        learned_cfg.utility_weights.cost * estimated_cost
        + learned_cfg.utility_weights.latency * (profile.latency_ms / 1000.0)
        + learned_cfg.utility_weights.failure * profile.failure_rate
    )
    utility = quality_probability - utility_penalty

    return ModelScore(
        model=model,
        quality_probability=quality_probability,
        utility=utility,
        estimated_cost=estimated_cost,
        estimated_latency_ms=profile.latency_ms,
        estimated_failure_rate=profile.failure_rate,
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = exp(value)
    return exp_pos / (1.0 + exp_pos)
