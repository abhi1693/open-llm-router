from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
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


class RoutingConstraintError(ValueError):
    def __init__(self, *, constraint: str, message: str, details: dict[str, Any] | None = None):
        self.constraint = constraint
        self.details = details or {}
        super().__init__(message)


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
    provider_preferences: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "InvalidModelError",
    "RoutingConstraintError",
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
        allowed_model_patterns = _extract_allowed_model_patterns(payload)
        provider_preferences = _extract_provider_preferences(payload)
        if self.config.should_auto_route(requested_model):
            task, complexity, signals = classify_request(
                payload=payload,
                endpoint=endpoint,
                complexity_cfg=self.config.complexity,
            )
            required_capabilities = self._required_capabilities(
                payload=payload,
                endpoint=endpoint,
                signals=signals,
            )
            routed_models = self.config.route_for(task=task, complexity=complexity)
            candidate_chain = _dedupe_preserving_order(
                [*routed_models, *self.config.fallback_models]
            )
            allowed_filtered_chain = _filter_models_by_patterns(
                candidate_models=candidate_chain,
                patterns=allowed_model_patterns,
            )
            if allowed_model_patterns and not allowed_filtered_chain:
                raise RoutingConstraintError(
                    constraint="allowed_models",
                    message="No route candidates match allowed_models constraints.",
                    details={
                        "allowed_models": allowed_model_patterns,
                        "candidate_chain": candidate_chain,
                    },
                )
            default_chain, hard_constraint_trace = self._apply_hard_constraints(
                candidate_models=allowed_filtered_chain,
                required_capabilities=required_capabilities,
                payload=payload,
                signals=signals,
            )
            decision_trace: dict[str, Any] = {
                "auto": True,
                "routing_mode": "rule_chain",
                "route_task": task,
                "route_complexity": complexity,
                "route_models": list(routed_models),
                "fallback_config": list(self.config.fallback_models),
                "default_chain": list(candidate_chain),
                "allowed_models": list(allowed_model_patterns),
                "allowed_model_filtered_chain": list(allowed_filtered_chain),
                "hard_constraint_required_capabilities": sorted(required_capabilities),
                "hard_constraint_filtered_chain": list(default_chain),
                "hard_constraint_rejections": hard_constraint_trace,
                "provider_preferences": dict(provider_preferences),
            }
            if self.config.learned_routing.enabled:
                (
                    selected_model,
                    fallbacks,
                    ranked_models,
                    candidate_scores,
                    learned_trace,
                ) = self._decide_learned(
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
                        "learned_trace": learned_trace,
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
            if allowed_model_patterns and not _model_matches_any_pattern(
                model=requested_model,
                patterns=allowed_model_patterns,
            ):
                raise RoutingConstraintError(
                    constraint="allowed_models",
                    message=(
                        "Requested explicit model does not match allowed_models constraints."
                    ),
                    details={
                        "requested_model": requested_model,
                        "allowed_models": allowed_model_patterns,
                    },
                )
            task = "explicit"
            complexity = "n/a"
            signals = {}
            selected_model = requested_model
            fallbacks = [model for model in self.config.fallback_models if model != selected_model]
            fallbacks = _filter_models_by_patterns(
                candidate_models=fallbacks,
                patterns=allowed_model_patterns,
            )
            ranked_models = [selected_model, *fallbacks]
            candidate_scores = []
            source = "request"
            decision_trace = {
                "auto": False,
                "selected_reason": "explicit_model_request",
                "fallback_config": list(self.config.fallback_models),
                "allowed_models": list(allowed_model_patterns),
                "provider_preferences": dict(provider_preferences),
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
            provider_preferences=provider_preferences,
        )

    def _decide_learned(
        self,
        payload: dict[str, Any],
        task: str,
        complexity: str,
        signals: dict[str, Any],
        default_chain: list[str],
    ) -> tuple[str, list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
        configured_candidates = self.config.learned_routing.task_candidates.get(task, [])
        if default_chain:
            # Keep learned routing inside the complexity-derived rule-chain by default.
            # This prevents low/medium prompts from frequently jumping to xhigh-only models.
            candidate_models = list(default_chain)
            candidate_source = "default_chain"
        else:
            candidate_models = _dedupe_preserving_order(list(configured_candidates))
            candidate_source = "task_candidates"
        if not candidate_models:
            fallback_default = self.config.default_model
            return (
                fallback_default,
                [],
                [fallback_default],
                [],
                {
                    "candidate_source": "default_model_fallback",
                    "configured_candidates": configured_candidates,
                    "default_chain": default_chain,
                },
            )

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
        return (
            selected_model,
            fallbacks,
            ranked_models,
            candidate_scores,
            {
                "candidate_source": candidate_source,
                "configured_candidates": configured_candidates,
                "default_chain": default_chain,
            },
        )

    @staticmethod
    def _required_capabilities(
        payload: dict[str, Any],
        endpoint: str,
        signals: dict[str, Any],
    ) -> set[str]:
        required: set[str] = set()
        if endpoint == "/v1/embeddings":
            required.add("embeddings")
        else:
            required.add("chat")

        if bool(payload.get("stream")):
            required.add("streaming")
        if payload.get("tools") or payload.get("functions"):
            required.add("tool_use")

        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            response_type = str(response_format.get("type") or "").strip().lower()
            if response_type in {"json_object", "json_schema"}:
                required.add("json_mode")

        if signals.get("has_image"):
            required.add("vision")
        return required

    def _apply_hard_constraints(
        self,
        candidate_models: list[str],
        required_capabilities: set[str],
        payload: dict[str, Any],
        signals: dict[str, Any],
    ) -> tuple[list[str], dict[str, list[str]]]:
        if not candidate_models:
            return [], {}

        requested_output_tokens = _extract_requested_output_tokens(payload)
        # Char-count is a cheap approximation and keeps routing on the fast path.
        estimated_input_tokens = max(1, int(float(signals.get("text_length_total", 0)) / 4.0))

        accepted: list[str] = []
        rejected: dict[str, list[str]] = {}

        for model in candidate_models:
            reasons: list[str] = []
            if not self._is_model_available_for_any_enabled_account(model):
                reasons.append("no_enabled_account_supports_model")
            metadata = self.config.models.get(model) or {}
            capabilities = _extract_capabilities(metadata)
            if required_capabilities and capabilities:
                missing = sorted(required_capabilities - capabilities)
                if missing:
                    reasons.append(f"missing_capabilities:{','.join(missing)}")

            limits = metadata.get("limits")
            if isinstance(limits, dict):
                max_output = _as_positive_int(limits.get("max_output_tokens"))
                if (
                    max_output is not None
                    and requested_output_tokens is not None
                    and requested_output_tokens > max_output
                ):
                    reasons.append(
                        f"max_output_tokens_exceeded:{requested_output_tokens}>{max_output}"
                    )

                context_tokens = _as_positive_int(limits.get("context_tokens"))
                if context_tokens is not None:
                    output_budget = requested_output_tokens or 0
                    if estimated_input_tokens + output_budget > context_tokens:
                        reasons.append(
                            (
                                "context_window_exceeded:"
                                f"{estimated_input_tokens + output_budget}>{context_tokens}"
                            )
                        )

            if reasons:
                rejected[model] = reasons
                continue
            accepted.append(model)

        if accepted:
            return accepted, rejected
        # If all candidates were rejected, keep original chain to avoid complete outage.
        return candidate_models, rejected

    def _is_model_available_for_any_enabled_account(self, model: str) -> bool:
        if not self.config.accounts:
            return True
        return any(
            account.enabled and account.supports_model(model)
            for account in self.config.accounts
        )


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _extract_requested_output_tokens(payload: dict[str, Any]) -> int | None:
    for key in ("max_output_tokens", "max_tokens"):
        value = payload.get(key)
        parsed = _as_positive_int(value)
        if parsed is not None:
            return parsed
    return None


def _extract_capabilities(metadata: dict[str, Any]) -> set[str]:
    raw = metadata.get("capabilities")
    if not isinstance(raw, list):
        return set()
    capabilities: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if normalized:
            capabilities.add(normalized)
    return capabilities


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _extract_allowed_model_patterns(payload: dict[str, Any]) -> list[str]:
    patterns: list[str] = []

    direct_patterns = payload.get("allowed_models")
    if isinstance(direct_patterns, list):
        patterns.extend(_coerce_string_list(direct_patterns))

    plugins = payload.get("plugins")
    if isinstance(plugins, list):
        for plugin in plugins:
            if not isinstance(plugin, dict):
                continue
            plugin_id = str(plugin.get("id") or "").strip().lower()
            if plugin_id not in {"auto-router", "auto_router"}:
                continue
            allowed_models = plugin.get("allowed_models")
            if isinstance(allowed_models, list):
                patterns.extend(_coerce_string_list(allowed_models))

    return _dedupe_preserving_order(patterns)


def _extract_provider_preferences(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("provider")
    if not isinstance(raw, dict):
        return {}

    output: dict[str, Any] = {}

    order = raw.get("order")
    if isinstance(order, list):
        normalized_order = _coerce_string_list(order)
        if normalized_order:
            output["order"] = normalized_order

    only = raw.get("only")
    if isinstance(only, list):
        normalized_only = _coerce_string_list(only)
        if normalized_only:
            output["only"] = normalized_only

    ignore = raw.get("ignore")
    if isinstance(ignore, list):
        normalized_ignore = _coerce_string_list(ignore)
        if normalized_ignore:
            output["ignore"] = normalized_ignore

    partition = raw.get("partition")
    if isinstance(partition, str):
        normalized_partition = partition.strip().lower()
        if normalized_partition in {"model", "none"}:
            output["partition"] = normalized_partition

    require_parameters = raw.get("require_parameters")
    if isinstance(require_parameters, bool):
        output["require_parameters"] = require_parameters

    allow_fallbacks = raw.get("allow_fallbacks")
    if isinstance(allow_fallbacks, bool):
        output["allow_fallbacks"] = allow_fallbacks

    raw_sort = raw.get("sort")
    if isinstance(raw_sort, str):
        normalized_sort = raw_sort.strip().lower()
        if normalized_sort in {"price", "latency", "throughput"}:
            output["sort"] = normalized_sort
    elif isinstance(raw_sort, dict):
        sort_by = raw_sort.get("by")
        if isinstance(sort_by, str):
            normalized_sort = sort_by.strip().lower()
            if normalized_sort in {"price", "latency", "throughput"}:
                output["sort"] = normalized_sort
        partition = raw_sort.get("partition")
        if isinstance(partition, str):
            normalized_partition = partition.strip().lower()
            if normalized_partition in {"model", "none"}:
                output["partition"] = normalized_partition

    return output


def _filter_models_by_patterns(candidate_models: list[str], patterns: list[str]) -> list[str]:
    if not patterns:
        return list(candidate_models)
    return [
        model
        for model in candidate_models
        if _model_matches_any_pattern(model=model, patterns=patterns)
    ]


def _model_matches_any_pattern(model: str, patterns: list[str]) -> bool:
    _, _, model_id = model.partition("/")
    for pattern in patterns:
        if fnmatch(model, pattern):
            return True
        if "/" not in pattern and model_id and fnmatch(model_id, pattern):
            return True
    return False


def _coerce_string_list(values: list[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            output.append(normalized)
    return output
