from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from open_llm_router.classifier import classify_request
from open_llm_router.config import ModelProfile, RoutingConfig
from open_llm_router.scoring import build_routing_features, score_model


class InvalidModelError(ValueError):
    def __init__(self, requested_model: str, available_models: list[str]):
        self.requested_model = requested_model
        self.available_models = available_models
        super().__init__(
            f"Requested model '{requested_model}' is not configured. "
            "Use 'auto' or a configured model."
        )


@dataclass(slots=True)
class RouteDecision:
    selected_model: str
    source: str
    task: str
    complexity: str
    requested_model: str | None
    fallback_models: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    ranked_models: list[str] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    decision_trace: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "InvalidModelError",
    "RouteDecision",
    "SmartModelRouter",
]


class SmartModelRouter:
    def __init__(self, config: RoutingConfig):
        self.config = config

    @staticmethod
    def _normalize_requested_model(requested_model: Any) -> str | None:
        if requested_model is None:
            return None
        normalized = str(requested_model).strip()
        return normalized or None

    def _require_known_model(self, requested_model: str) -> None:
        available_models = self.config.available_models()
        if requested_model not in available_models:
            raise InvalidModelError(
                requested_model=requested_model,
                available_models=available_models,
            )

    def decide(self, payload: dict[str, Any], endpoint: str) -> RouteDecision:
        requested_model = self._normalize_requested_model(payload.get("model"))
        if self.config.should_auto_route(requested_model):
            task, complexity, signals = classify_request(
                payload=payload,
                endpoint=endpoint,
                complexity_cfg=self.config.complexity,
            )
            routed_models = self.config.route_for(task=task, complexity=complexity)
            default_chain = _dedupe_preserving_order(
                [*routed_models, *self.config.fallback_models]
            )
            decision_trace: dict[str, Any] = {
                "auto": True,
                "routing_mode": "rule_chain",
                "route_task": task,
                "route_complexity": complexity,
                "route_models": list(routed_models),
                "fallback_config": list(self.config.fallback_models),
                "default_chain": list(default_chain),
            }
            if self.config.learned_routing.enabled:
                selected_model, fallbacks, ranked_models, candidate_scores = self._decide_learned(
                    payload=payload,
                    task=task,
                    complexity=complexity,
                    signals=signals,
                    default_chain=default_chain,
                )
                signals["routing_mode"] = "learned_utility"
                decision_trace.update(
                    {
                        "routing_mode": "learned_utility",
                        "learned_candidates": list(ranked_models),
                        "selected_score": candidate_scores[0] if candidate_scores else None,
                    }
                )
            else:
                selected_model = default_chain[0]
                fallbacks = [model for model in default_chain[1:] if model != selected_model]
                ranked_models = [selected_model, *fallbacks]
                candidate_scores = []
                decision_trace.update(
                    {
                        "selected_reason": "head_of_rule_chain",
                    }
                )
            source = "auto"
        else:
            self._require_known_model(requested_model=requested_model)
            task = "explicit"
            complexity = "n/a"
            signals = {}
            selected_model = requested_model
            fallbacks = [model for model in self.config.fallback_models if model != selected_model]
            ranked_models = [selected_model, *fallbacks]
            candidate_scores = []
            source = "request"
            decision_trace = {
                "auto": False,
                "selected_reason": "explicit_model_request",
                "fallback_config": list(self.config.fallback_models),
            }

        return RouteDecision(
            selected_model=selected_model,
            source=source,
            task=task,
            complexity=complexity,
            requested_model=requested_model,
            fallback_models=fallbacks,
            signals=signals,
            ranked_models=ranked_models,
            candidate_scores=candidate_scores,
            decision_trace=decision_trace,
        )

    def _decide_learned(
        self,
        payload: dict[str, Any],
        task: str,
        complexity: str,
        signals: dict[str, Any],
        default_chain: list[str],
    ) -> tuple[str, list[str], list[str], list[dict[str, Any]]]:
        configured_candidates = self.config.learned_routing.task_candidates.get(task, [])
        candidate_models = _dedupe_preserving_order([*configured_candidates, *default_chain])
        if not candidate_models:
            fallback_default = self.config.default_model
            return fallback_default, [], [fallback_default], []

        features = build_routing_features(
            task=task,
            complexity=complexity,
            signals=signals,
            payload=payload,
        )
        scored = []
        for model in candidate_models:
            profile = self.config.model_profiles.get(model, ModelProfile())
            scored.append(
                score_model(
                    model=model,
                    profile=profile,
                    features=features,
                    payload=payload,
                    signals=signals,
                    learned_cfg=self.config.learned_routing,
                )
            )

        scored.sort(key=lambda item: item.utility, reverse=True)
        ranked_models = [item.model for item in scored]
        selected_model = ranked_models[0]
        fallbacks = ranked_models[1:]
        candidate_scores = [item.as_dict() for item in scored]
        return selected_model, fallbacks, ranked_models, candidate_scores


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
