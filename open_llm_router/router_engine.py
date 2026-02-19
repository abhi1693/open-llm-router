from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import lru_cache
import json
from math import log, sqrt
from threading import Lock
from typing import Any, Iterator

from open_llm_router.classifier import classify_request
from open_llm_router.config import ModelProfile, RoutingConfig
from open_llm_router.scoring import ModelScore, build_routing_features, score_model

_CONTEXT_WINDOW_OVERFLOW_TOLERANCE = 0.10
_HIGH_CONTEXT_SUPPLEMENT_TOKENS = 120_000
_MAX_SUPPLEMENTAL_MODELS = 2
_UCB_EXPLORATION_WEIGHT = 0.25


class InvalidModelError(ValueError):
    def __init__(self, requested_model: str, available_models: list[str]):
        self.requested_model = requested_model
        self.available_models = available_models
        super().__init__(
            f"Requested model '{requested_model}' is not configured. "
            "Use 'auto' or a configured model."
        )


class RoutingConstraintError(ValueError):
    def __init__(
        self, *, constraint: str, message: str, details: dict[str, Any] | None = None
    ):
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
        self._bandit_lock = Lock()
        self._bandit_total_decisions = 0
        self._bandit_selection_counts: dict[str, int] = {}

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
                calibration_cfg=self.config.classifier_calibration,
                semantic_cfg=self.config.semantic_classifier,
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
            (
                provider_filtered_chain,
                provider_filter_trace,
            ) = self._apply_provider_preferences_to_models(
                candidate_models=allowed_filtered_chain,
                provider_preferences=provider_preferences,
            )
            if (
                _provider_filters_enabled(provider_preferences)
                and not provider_filtered_chain
            ):
                raise RoutingConstraintError(
                    constraint="provider_preferences",
                    message="No route candidates satisfy provider preference constraints.",
                    details={
                        "provider_preferences": dict(provider_preferences),
                        "candidate_chain": candidate_chain,
                        "allowed_model_filtered_chain": allowed_filtered_chain,
                    },
                )
            (
                default_chain,
                hard_constraint_trace,
                supplemented_models,
                estimated_input_tokens,
                token_estimation_method,
            ) = self._apply_hard_constraints(
                candidate_models=provider_filtered_chain,
                required_capabilities=required_capabilities,
                payload=payload,
                signals=signals,
                allowed_model_patterns=allowed_model_patterns,
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
                "provider_preference_filtered_chain": list(provider_filtered_chain),
                "provider_preference_trace": provider_filter_trace,
                "hard_constraint_required_capabilities": sorted(required_capabilities),
                "hard_constraint_filtered_chain": list(default_chain),
                "hard_constraint_rejections": hard_constraint_trace,
                "hard_constraint_estimated_input_tokens": estimated_input_tokens,
                "hard_constraint_token_estimation": token_estimation_method,
                "provider_preferences": dict(provider_preferences),
            }
            if supplemented_models:
                decision_trace["hard_constraint_supplemented_models"] = list(
                    supplemented_models
                )
            use_factual_rule_chain_guardrail = (
                self._should_pin_factual_general_query_to_rule_chain(
                    task=task,
                    complexity=complexity,
                    signals=signals,
                )
            )
            if self.config.learned_routing.enabled and not use_factual_rule_chain_guardrail:
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
                    provider_preferences=provider_preferences,
                )
                signals["routing_mode"] = "learned_utility"
                decision_trace.update(
                    {
                        "routing_mode": "learned_utility",
                        "learned_candidates": list(ranked_models),
                        "selected_score": (
                            candidate_scores[0] if candidate_scores else None
                        ),
                        "learned_trace": learned_trace,
                    }
                )
            else:
                selected_model = default_chain[0]
                fallbacks = [
                    model for model in default_chain[1:] if model != selected_model
                ]
                ranked_models = [selected_model, *fallbacks]
                candidate_scores = []
                selected_reason = "head_of_rule_chain"
                if use_factual_rule_chain_guardrail:
                    selected_reason = "factual_query_rule_chain_guardrail"
                    signals["routing_mode"] = "rule_chain_guardrail"
                decision_trace.update(
                    {
                        "selected_reason": selected_reason,
                    }
                )
            source = "auto"
        else:
            assert requested_model is not None
            self._require_known_model(requested_model=requested_model)
            (
                provider_filtered_requested,
                _,
            ) = self._apply_provider_preferences_to_models(
                candidate_models=[requested_model],
                provider_preferences=provider_preferences,
            )
            if (
                _provider_filters_enabled(provider_preferences)
                and not provider_filtered_requested
            ):
                raise RoutingConstraintError(
                    constraint="provider_preferences",
                    message=(
                        "Requested explicit model does not satisfy provider preference constraints."
                    ),
                    details={
                        "requested_model": requested_model,
                        "provider_preferences": dict(provider_preferences),
                    },
                )
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
            fallbacks = [
                model
                for model in self.config.fallback_models
                if model != selected_model
            ]
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

    @staticmethod
    def _should_pin_factual_general_query_to_rule_chain(
        *, task: str, complexity: str, signals: dict[str, Any]
    ) -> bool:
        if task != "general" or complexity != "low":
            return False
        if not bool(signals.get("latest_user_factual_query")):
            return False
        if bool(signals.get("latest_user_references_context")):
            return False
        latest_structural_code_score = float(
            signals.get("latest_user_structural_code_score", 0.0)
        )
        if latest_structural_code_score > 0.0:
            return False
        text_length = int(signals.get("latest_user_text_length", signals.get("text_length", 0)))
        if text_length > 220:
            return False
        return True

    def _decide_learned(
        self,
        payload: dict[str, Any],
        task: str,
        complexity: str,
        signals: dict[str, Any],
        default_chain: list[str],
        provider_preferences: dict[str, Any],
    ) -> tuple[str, list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
        configured_candidates = self.config.learned_routing.task_candidates.get(
            task, []
        )
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

        provider_order_index = _provider_order_index_map(
            provider_preferences.get("order")
        )
        provider_order_bonus = {
            item.model: _provider_order_bonus_for_model(
                model=item.model,
                routing_config=self.config,
                provider_order_index=provider_order_index,
            )
            for item in scored
        }
        base_selection_scores = {
            item.model: item.utility + provider_order_bonus.get(item.model, 0.0)
            for item in scored
        }
        scored.sort(
            key=lambda item: (
                base_selection_scores[item.model],
                item.utility,
            ),
            reverse=True,
        )
        bandit_scores: dict[str, float] = {}
        bandit_bonus: dict[str, float] = {}
        bandit_counts: dict[str, int] = {}
        bandit_total_decisions = 0
        if len(scored) >= 2:
            (
                scored,
                bandit_scores,
                bandit_bonus,
                bandit_counts,
                bandit_total_decisions,
            ) = self._rank_with_ucb(
                scored,
                base_selection_scores=base_selection_scores,
            )

        ranked_models = [item.model for item in scored]
        selected_model = ranked_models[0]
        self._record_bandit_selection(selected_model)
        fallbacks = ranked_models[1:]
        candidate_scores: list[dict[str, Any]] = []
        for item in scored:
            row = item.as_dict()
            if item.model in bandit_scores:
                row["bandit_selection_score"] = round(bandit_scores[item.model], 6)
                row["bandit_exploration_bonus"] = round(bandit_bonus[item.model], 6)
            if provider_order_bonus.get(item.model, 0.0) != 0.0:
                row["provider_order_bonus"] = round(
                    provider_order_bonus[item.model], 6
                )
            candidate_scores.append(row)
        return (
            selected_model,
            fallbacks,
            ranked_models,
            candidate_scores,
            {
                "candidate_source": candidate_source,
                "configured_candidates": configured_candidates,
                "default_chain": default_chain,
                "provider_order_applied": bool(provider_order_index),
                "bandit": {
                    "strategy": "ucb1",
                    "enabled": len(scored) >= 2,
                    "exploration_weight": _UCB_EXPLORATION_WEIGHT,
                    "total_decisions_before_selection": bandit_total_decisions,
                    "selection_counts_before_selection": bandit_counts,
                },
            },
        )

    def _rank_with_ucb(
        self,
        scored: list[ModelScore],
        *,
        base_selection_scores: dict[str, float] | None = None,
    ) -> tuple[
        list[ModelScore],
        dict[str, float],
        dict[str, float],
        dict[str, int],
        int,
    ]:
        with self._bandit_lock:
            counts = {
                item.model: self._bandit_selection_counts.get(item.model, 0)
                for item in scored
            }
            total_decisions = self._bandit_total_decisions

        log_term = log(float(total_decisions) + 2.0)
        selection_scores: dict[str, float] = {}
        exploration_bonus: dict[str, float] = {}
        for item in scored:
            pulls = counts[item.model]
            bonus = _UCB_EXPLORATION_WEIGHT * sqrt(log_term / float(pulls + 1))
            exploration_bonus[item.model] = bonus
            base_score = (
                float(base_selection_scores.get(item.model, item.utility))
                if base_selection_scores is not None
                else item.utility
            )
            selection_scores[item.model] = base_score + bonus

        ranked = sorted(
            scored,
            key=lambda item: (selection_scores[item.model], item.utility),
            reverse=True,
        )
        return ranked, selection_scores, exploration_bonus, counts, total_decisions

    def _record_bandit_selection(self, model: str) -> None:
        with self._bandit_lock:
            self._bandit_total_decisions += 1
            self._bandit_selection_counts[model] = (
                self._bandit_selection_counts.get(model, 0) + 1
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
        allowed_model_patterns: list[str],
    ) -> tuple[list[str], dict[str, list[str]], list[str], int, str]:
        if not candidate_models:
            return [], {}, [], 1, "char_heuristic"

        requested_output_tokens = _extract_requested_output_tokens(payload)
        estimated_input_tokens, token_estimation_method = _estimate_payload_tokens(
            payload=payload,
            model_hint=self._resolve_tokenizer_model_hint(candidate_models),
            fallback_char_count=float(signals.get("text_length_total", 0)),
        )

        accepted: list[str] = []
        rejected: dict[str, list[str]] = {}

        for model in candidate_models:
            reasons = self._constraint_reasons_for_model(
                model=model,
                required_capabilities=required_capabilities,
                requested_output_tokens=requested_output_tokens,
                estimated_input_tokens=estimated_input_tokens,
            )
            if reasons:
                rejected[model] = reasons
                continue
            accepted.append(model)

        if accepted:
            supplemented = self._supplement_large_context_candidates(
                accepted_models=accepted,
                candidate_models=candidate_models,
                required_capabilities=required_capabilities,
                requested_output_tokens=requested_output_tokens,
                estimated_input_tokens=estimated_input_tokens,
                allowed_model_patterns=allowed_model_patterns,
            )
            return (
                accepted,
                rejected,
                supplemented,
                estimated_input_tokens,
                token_estimation_method,
            )
        # If all candidates were rejected, keep original chain to avoid complete outage.
        return (
            candidate_models,
            rejected,
            [],
            estimated_input_tokens,
            token_estimation_method,
        )

    def _resolve_tokenizer_model_hint(self, candidate_models: list[str]) -> str | None:
        for model in candidate_models:
            metadata = self.config.models.get(model)
            if isinstance(metadata, dict):
                model_id = metadata.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id.strip()
            if model.strip():
                return model.strip()
        return None

    def _apply_provider_preferences_to_models(
        self,
        *,
        candidate_models: list[str],
        provider_preferences: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        normalized_only = _normalized_provider_values(provider_preferences.get("only"))
        normalized_ignore = _normalized_provider_values(
            provider_preferences.get("ignore")
        )
        provider_order_index = _provider_order_index_map(
            provider_preferences.get("order")
        )

        filtered: list[str] = []
        rejections: dict[str, str] = {}
        for model in candidate_models:
            if not _model_satisfies_provider_filters(
                model=model,
                routing_config=self.config,
                only=normalized_only,
                ignore=normalized_ignore,
            ):
                if normalized_only and not _model_matches_provider_values(
                    model=model,
                    routing_config=self.config,
                    values=normalized_only,
                ):
                    rejections[model] = "provider_only_mismatch"
                elif normalized_ignore and _model_matches_provider_values(
                    model=model,
                    routing_config=self.config,
                    values=normalized_ignore,
                ):
                    rejections[model] = "provider_ignore_match"
                else:
                    rejections[model] = "provider_filter_mismatch"
                continue
            filtered.append(model)

        if provider_order_index and len(filtered) > 1:
            filtered = _sort_models_by_provider_order(
                models=filtered,
                routing_config=self.config,
                provider_order_index=provider_order_index,
            )

        trace = {
            "only": sorted(normalized_only),
            "ignore": sorted(normalized_ignore),
            "order_applied": bool(provider_order_index),
            "rejections": rejections,
        }
        return filtered, trace

    def _constraint_reasons_for_model(
        self,
        *,
        model: str,
        required_capabilities: set[str],
        requested_output_tokens: int | None,
        estimated_input_tokens: int,
    ) -> list[str]:
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
                estimated_total_tokens = estimated_input_tokens + output_budget
                if estimated_total_tokens > context_tokens:
                    overflow_ratio = (
                        estimated_total_tokens - context_tokens
                    ) / max(1, context_tokens)
                    if overflow_ratio > _CONTEXT_WINDOW_OVERFLOW_TOLERANCE:
                        reasons.append(
                            (
                                "context_window_exceeded:"
                                f"{estimated_total_tokens}>{context_tokens}"
                            )
                        )
        return reasons

    def _supplement_large_context_candidates(
        self,
        *,
        accepted_models: list[str],
        candidate_models: list[str],
        required_capabilities: set[str],
        requested_output_tokens: int | None,
        estimated_input_tokens: int,
        allowed_model_patterns: list[str],
    ) -> list[str]:
        if len(accepted_models) >= 2:
            return []
        if estimated_input_tokens < _HIGH_CONTEXT_SUPPLEMENT_TOKENS:
            return []

        supplemental_pool = self.config.available_models()
        if allowed_model_patterns:
            supplemental_pool = _filter_models_by_patterns(
                candidate_models=supplemental_pool,
                patterns=allowed_model_patterns,
            )

        accepted_set = set(accepted_models)
        candidate_set = set(candidate_models)
        supplemented: list[str] = []

        for model in supplemental_pool:
            if model in accepted_set or model in candidate_set:
                continue
            reasons = self._constraint_reasons_for_model(
                model=model,
                required_capabilities=required_capabilities,
                requested_output_tokens=requested_output_tokens,
                estimated_input_tokens=estimated_input_tokens,
            )
            if reasons:
                continue
            accepted_models.append(model)
            accepted_set.add(model)
            supplemented.append(model)
            if len(accepted_models) >= 2 or len(supplemented) >= _MAX_SUPPLEMENTAL_MODELS:
                break

        return supplemented

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


def _estimate_payload_tokens(
    *,
    payload: dict[str, Any],
    model_hint: str | None,
    fallback_char_count: float = 0.0,
) -> tuple[int, str]:
    encoder = _resolve_token_encoder(model_hint)
    texts = list(_iter_payload_text(payload))

    if encoder is not None:
        token_count = 0
        for text in texts:
            token_count += len(encoder.encode(text, disallowed_special=()))
        messages = payload.get("messages")
        if isinstance(messages, list):
            token_count += len(messages) * 3
        return max(1, token_count), "tiktoken"

    if texts:
        fallback_char_count = float(sum(len(text) for text in texts))
    return max(1, int(fallback_char_count / 4.0)), "char_heuristic"


def _iter_payload_text(payload: dict[str, Any]) -> Iterator[str]:
    for key in ("messages", "input", "prompt"):
        value = payload.get(key)
        yield from _iter_text_fragments(value)

    for key in ("tools", "functions", "response_format"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            yield json.dumps(value, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError):
            continue


def _iter_text_fragments(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value:
            yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_text_fragments(item)
        return
    if not isinstance(value, dict):
        return

    text = value.get("text")
    if isinstance(text, str) and text:
        yield text
    input_text = value.get("input_text")
    if isinstance(input_text, str) and input_text:
        yield input_text
    content = value.get("content")
    if content is not None:
        yield from _iter_text_fragments(content)
    prompt = value.get("prompt")
    if prompt is not None:
        yield from _iter_text_fragments(prompt)


@lru_cache(maxsize=32)
def _resolve_token_encoder(model_hint: str | None) -> Any | None:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None

    hints: list[str] = []
    if isinstance(model_hint, str):
        normalized = model_hint.strip()
        if normalized:
            hints.append(normalized)
            if "/" in normalized:
                _, _, suffix = normalized.partition("/")
                suffix = suffix.strip()
                if suffix:
                    hints.append(suffix)

    for hint in hints:
        try:
            return tiktoken.encoding_for_model(hint)
        except KeyError:
            continue

    try:
        return tiktoken.get_encoding("cl100k_base")
    except KeyError:
        return None


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


def _provider_filters_enabled(provider_preferences: dict[str, Any]) -> bool:
    return bool(provider_preferences.get("only")) or bool(
        provider_preferences.get("ignore")
    )


def _normalized_provider_values(raw_values: Any) -> set[str]:
    if not isinstance(raw_values, list):
        return set()
    values: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if normalized:
            values.add(normalized)
    return values


def _model_provider_identity(
    *,
    model: str,
    routing_config: RoutingConfig,
) -> tuple[str | None, str]:
    provider, sep, model_id = model.partition("/")
    if sep and provider.strip() and model_id.strip():
        return provider.strip().lower(), model_id.strip().lower()
    metadata = routing_config.models.get(model)
    if isinstance(metadata, dict):
        raw_provider = metadata.get("provider")
        if isinstance(raw_provider, str) and raw_provider.strip():
            return raw_provider.strip().lower(), model.strip().lower()
    return None, model.strip().lower()


def _model_matches_provider_values(
    *,
    model: str,
    routing_config: RoutingConfig,
    values: set[str],
) -> bool:
    if not values:
        return False
    provider, model_id = _model_provider_identity(
        model=model,
        routing_config=routing_config,
    )
    normalized_model = model.strip().lower()
    if provider and provider in values:
        return True
    if model_id in values:
        return True
    return normalized_model in values


def _model_satisfies_provider_filters(
    *,
    model: str,
    routing_config: RoutingConfig,
    only: set[str],
    ignore: set[str],
) -> bool:
    if only and not _model_matches_provider_values(
        model=model,
        routing_config=routing_config,
        values=only,
    ):
        return False
    if ignore and _model_matches_provider_values(
        model=model,
        routing_config=routing_config,
        values=ignore,
    ):
        return False
    return True


def _provider_order_index_map(raw_order: Any) -> dict[str, int]:
    if not isinstance(raw_order, list):
        return {}
    index_map: dict[str, int] = {}
    for index, item in enumerate(raw_order):
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if not normalized or normalized in index_map:
            continue
        index_map[normalized] = index
    return index_map


def _provider_order_rank_for_model(
    *,
    model: str,
    routing_config: RoutingConfig,
    provider_order_index: dict[str, int],
) -> int:
    if not provider_order_index:
        return 10_000
    provider, model_id = _model_provider_identity(
        model=model,
        routing_config=routing_config,
    )
    normalized_model = model.strip().lower()
    ranks = [
        provider_order_index.get(normalized_model, 10_000),
        provider_order_index.get(model_id, 10_000),
    ]
    if provider:
        ranks.append(provider_order_index.get(provider, 10_000))
    return min(ranks)


def _provider_order_bonus_for_model(
    *,
    model: str,
    routing_config: RoutingConfig,
    provider_order_index: dict[str, int],
) -> float:
    if not provider_order_index:
        return 0.0
    rank = _provider_order_rank_for_model(
        model=model,
        routing_config=routing_config,
        provider_order_index=provider_order_index,
    )
    if rank >= 10_000:
        return 0.0
    span = float(max(1, len(provider_order_index)))
    return 0.05 * ((span - float(rank)) / span)


def _sort_models_by_provider_order(
    *,
    models: list[str],
    routing_config: RoutingConfig,
    provider_order_index: dict[str, int],
) -> list[str]:
    indexed_models = list(enumerate(models))
    indexed_models.sort(
        key=lambda item: (
            _provider_order_rank_for_model(
                model=item[1],
                routing_config=routing_config,
                provider_order_index=provider_order_index,
            ),
            item[0],
        )
    )
    return [model for _, model in indexed_models]


def _filter_models_by_patterns(
    candidate_models: list[str], patterns: list[str]
) -> list[str]:
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
