from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from open_llm_router.catalog import (
    CatalogLookupError,
    CatalogValidationError,
    ModelCatalogEntry,
    RouterCatalog,
    load_internal_catalog,
    validate_routing_document_against_catalog,
)
from open_llm_router.profile_config import (
    GuardrailThresholds,
    RouterProfileConfig,
)

DEFAULT_RETRY_STATUSES = [429, 500, 502, 503, 504]
DEFAULT_COMPLEXITY = {
    "low_max_chars": 1200,
    "medium_max_chars": 6000,
    "high_max_chars": 16000,
}
TASKS = ["general", "coding", "thinking", "instruction_following", "image"]
ROUTE_TIERS = ["low", "medium", "high", "xhigh", "default"]

BASE_EFFECTIVE_CONFIG: dict[str, Any] = {
    "default_model": "",
    "task_routes": {},
    "fallback_models": [],
    "models": {},
    "model_profiles": {},
    "accounts": [],
    "retry_statuses": list(DEFAULT_RETRY_STATUSES),
    "complexity": dict(DEFAULT_COMPLEXITY),
    "learned_routing": {
        "enabled": False,
        "bias": -4.0,
        "default_output_tokens": 512,
        "feature_weights": {},
        "task_candidates": {},
        "utility_weights": {
            "cost": 12.0,
            "latency": 0.2,
            "failure": 3.0,
        },
    },
}


@dataclass(frozen=True)
class CompileResult:
    effective_config: dict[str, Any]
    explain: dict[str, Any]


@lru_cache
def _load_profiles_file() -> dict[str, Any]:
    profiles_path = Path(__file__).resolve().parent / "catalog" / "profiles.yaml"
    with profiles_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Expected profiles catalog to be an object.")
    profiles = raw.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ValueError("Expected 'profiles' mapping in profile catalog.")
    return profiles


def list_builtin_profiles() -> list[tuple[str, str]]:
    profiles = _load_profiles_file()
    rows: list[tuple[str, str]] = []
    for name, body in sorted(profiles.items()):
        description = ""
        if isinstance(body, dict):
            description = str(body.get("description") or "")
        rows.append((name, description))
    return rows


def get_builtin_profile_template(name: str) -> dict[str, Any]:
    profiles = _load_profiles_file()
    template = profiles.get(name)
    if template is None:
        raise ValueError(
            f"Unknown profile '{name}'. Available profiles: {', '.join(sorted(profiles))}"
        )
    if not isinstance(template, dict):
        raise ValueError(f"Invalid template for profile '{name}'.")
    return deepcopy(template)


def is_profile_document(raw: dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    if "profile" in raw:
        return True
    if "accounts" in raw and "raw_overrides" in raw:
        return True
    return False


def load_profile_document(path: str | Path) -> RouterProfileConfig:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML object in {profile_path}")
    return RouterProfileConfig.model_validate(raw)


def compile_profile_file(
    path: str | Path,
    *,
    catalog: RouterCatalog | None = None,
) -> CompileResult:
    profile = load_profile_document(path)
    return compile_profile_config(profile, catalog=catalog)


def compile_profile_document(
    raw: dict[str, Any],
    *,
    catalog: RouterCatalog | None = None,
) -> CompileResult:
    profile = RouterProfileConfig.model_validate(raw)
    return compile_profile_config(profile, catalog=catalog)


def compile_profile_config(
    profile: RouterProfileConfig,
    *,
    catalog: RouterCatalog | None = None,
) -> CompileResult:
    catalog = catalog or load_internal_catalog()

    effective = deepcopy(BASE_EFFECTIVE_CONFIG)
    explain: dict[str, Any] = {
        "schema_version": profile.schema_version,
        "profile_layers": [],
        "guardrail_pruned": [],
        "accounts": [],
    }

    global_profile_name = profile.profile.default
    global_template = get_builtin_profile_template(global_profile_name)
    _apply_layer(
        effective,
        global_template,
        layer_name=f"profile.default:{global_profile_name}",
        explain=explain,
    )

    for task, task_profile_name in sorted(profile.profile.per_task.items()):
        task_template = get_builtin_profile_template(task_profile_name)
        _apply_per_task_profile_layer(
            effective,
            task=task,
            template=task_template,
            layer_name=f"profile.per_task.{task}:{task_profile_name}",
            explain=explain,
        )

    compiled_accounts, account_models = _compile_accounts(profile, catalog, explain)
    if compiled_accounts:
        effective["accounts"] = compiled_accounts

    _materialize_model_catalog_data(effective, catalog, seed_models=account_models)
    _apply_guardrails(effective, profile, catalog, explain)
    _align_routing_to_enabled_accounts(
        effective,
        catalog=catalog,
        available_models=account_models,
        explain=explain,
    )

    if profile.raw_overrides:
        _merge_dict(effective, profile.raw_overrides)
        explain["profile_layers"].append("raw_overrides")

    validate_routing_document_against_catalog(effective, catalog=catalog)

    from open_llm_router.config import RoutingConfig

    validated = RoutingConfig.model_validate(effective)
    return CompileResult(
        effective_config=validated.model_dump(mode="python"),
        explain=explain,
    )


def _compile_accounts(
    profile: RouterProfileConfig,
    catalog: RouterCatalog,
    explain: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    output_accounts: list[dict[str, Any]] = []
    referenced_models: set[str] = set()
    errors: list[str] = []

    for idx, account in enumerate(profile.accounts):
        try:
            provider_id = catalog.resolve_provider_id(account.provider)
        except CatalogLookupError as exc:
            errors.append(exc.format_for_path(f"accounts[{idx}].provider"))
            continue

        provider_catalog = catalog.get_provider(provider_id)

        account_models_raw = account.models or [
            model_id
            for model_id in catalog.model_ids
            if model_id.startswith(f"{provider_id}/")
        ]
        account_models: list[str] = []
        for model_idx, model_ref in enumerate(account_models_raw):
            try:
                canonical_model = catalog.resolve_model_id(
                    model_ref,
                    provider_hint=provider_id,
                )
            except CatalogLookupError as exc:
                errors.append(
                    exc.format_for_path(f"accounts[{idx}].models[{model_idx}]")
                )
                continue

            if not canonical_model.startswith(f"{provider_id}/"):
                errors.append(
                    (
                        f"Model at accounts[{idx}].models[{model_idx}] belongs to provider "
                        f"'{canonical_model.split('/', 1)[0]}', expected '{provider_id}'."
                    )
                )
                continue

            account_models.append(canonical_model)
            referenced_models.add(canonical_model)

        account_payload = account.model_dump(mode="python", exclude_none=True)
        account_payload["provider"] = provider_id
        account_payload["base_url"] = provider_catalog.base_url
        account_payload["models"] = _dedupe(account_models)

        output_accounts.append(account_payload)
        explain["accounts"].append(
            {
                "name": account_payload.get("name"),
                "provider": provider_id,
                "base_url_source": "catalog.providers",
                "models": list(account_payload.get("models", [])),
            }
        )

    if errors:
        raise CatalogValidationError(errors)

    return output_accounts, referenced_models


def _materialize_model_catalog_data(
    effective: dict[str, Any],
    catalog: RouterCatalog,
    *,
    seed_models: set[str],
) -> None:
    model_refs = _collect_model_references(effective)
    model_refs.update(seed_models)

    models_map = effective.setdefault("models", {})
    if not isinstance(models_map, dict):
        models_map = {}
        effective["models"] = models_map

    model_profiles = effective.setdefault("model_profiles", {})
    if not isinstance(model_profiles, dict):
        model_profiles = {}
        effective["model_profiles"] = model_profiles

    for model_ref in sorted(model_refs):
        canonical = catalog.resolve_model_id(model_ref)
        entry = catalog.get_model(canonical)

        metadata = models_map.setdefault(canonical, {})
        if not isinstance(metadata, dict):
            metadata = {}
            models_map[canonical] = metadata

        metadata.setdefault("id", entry.id)
        metadata.setdefault("provider", entry.provider)
        if isinstance(entry.created, int):
            metadata.setdefault("created", entry.created)
        metadata.setdefault("aliases", list(entry.aliases))
        metadata.setdefault("costs", entry.costs.model_dump(mode="python"))
        metadata.setdefault("limits", entry.limits.model_dump(mode="python"))
        metadata.setdefault("capabilities", list(entry.capabilities))
        metadata.setdefault("priors", entry.priors.model_dump(mode="python"))
        if entry.family:
            metadata.setdefault("family", entry.family)
        if entry.type:
            metadata.setdefault("type", entry.type)
        if entry.tier:
            metadata.setdefault("tier", entry.tier)
        if entry.task_affinity:
            metadata.setdefault("task_affinity", dict(entry.task_affinity))

        profile = model_profiles.setdefault(canonical, {})
        if not isinstance(profile, dict):
            profile = {}
            model_profiles[canonical] = profile
        profile.setdefault("quality_bias", entry.priors.quality_bias)
        profile.setdefault("quality_sensitivity", entry.priors.quality_sensitivity)
        profile.setdefault("cost_input_per_1k", entry.costs.input_per_1k)
        profile.setdefault("cost_output_per_1k", entry.costs.output_per_1k)
        profile.setdefault("latency_ms", entry.priors.latency_ms)
        profile.setdefault("failure_rate", entry.priors.failure_rate)


def _align_routing_to_enabled_accounts(
    effective: dict[str, Any],
    *,
    catalog: RouterCatalog,
    available_models: set[str],
    explain: dict[str, Any],
) -> None:
    if not available_models:
        return

    available = sorted(_dedupe(list(available_models)))
    available_set = set(available)
    alignment_notes: list[dict[str, Any]] = []

    models_map = effective.get("models")
    if isinstance(models_map, dict):
        for model in list(models_map.keys()):
            canonical = (
                model if model in available_set else catalog.resolve_model_id(model)
            )
            if canonical not in available_set:
                models_map.pop(model, None)
                alignment_notes.append({"context": "models", "removed": model})

    model_profiles = effective.get("model_profiles")
    if isinstance(model_profiles, dict):
        for model in list(model_profiles.keys()):
            canonical = (
                model if model in available_set else catalog.resolve_model_id(model)
            )
            if canonical not in available_set:
                model_profiles.pop(model, None)
                alignment_notes.append({"context": "model_profiles", "removed": model})

    task_routes = effective.get("task_routes")
    if not isinstance(task_routes, dict):
        task_routes = {}
        effective["task_routes"] = task_routes

    for task in TASKS:
        route = task_routes.get(task)
        if not isinstance(route, dict):
            route = {}
            task_routes[task] = route

        for tier in ROUTE_TIERS:
            configured = _coerce_model_list(route.get(tier))
            filtered = _filter_models_to_available(
                configured,
                available_set=available_set,
                catalog=catalog,
            )
            if filtered:
                route[tier] = filtered
                continue

            suggested = _suggest_models_for_task_tier(
                available_models=available,
                task=task,
                tier=tier,
                catalog=catalog,
                limit=2,
            )
            if suggested:
                route[tier] = suggested
                if configured:
                    alignment_notes.append(
                        {
                            "context": f"task_routes.{task}.{tier}",
                            "from": configured,
                            "to": suggested,
                        }
                    )
                else:
                    alignment_notes.append(
                        {
                            "context": f"task_routes.{task}.{tier}",
                            "to": suggested,
                        }
                    )

    default_model_raw = str(effective.get("default_model") or "").strip()
    default_model: str | None = None
    if default_model_raw:
        default_model = catalog.resolve_model_id(default_model_raw)
        if default_model not in available_set:
            default_model = None

    if default_model is None:
        candidates = _suggest_models_for_task_tier(
            available_models=available,
            task="general",
            tier="medium",
            catalog=catalog,
            limit=1,
        )
        if candidates:
            old_default = default_model_raw or None
            effective["default_model"] = candidates[0]
            default_model = candidates[0]
            alignment_notes.append(
                {
                    "context": "default_model",
                    "from": old_default,
                    "to": candidates[0],
                }
            )

    fallback_models = _filter_models_to_available(
        _coerce_model_list(effective.get("fallback_models")),
        available_set=available_set,
        catalog=catalog,
    )
    if not fallback_models:
        fallback_models = _suggest_models_for_task_tier(
            available_models=available,
            task="general",
            tier="high",
            catalog=catalog,
            limit=max(1, min(4, len(available))),
        )
    if default_model:
        fallback_models = [model for model in fallback_models if model != default_model]
    effective["fallback_models"] = _dedupe(fallback_models)

    learned = effective.get("learned_routing")
    if isinstance(learned, dict):
        task_candidates = learned.get("task_candidates")
        if not isinstance(task_candidates, dict):
            task_candidates = {}
            learned["task_candidates"] = task_candidates
        for task in TASKS:
            filtered = _filter_models_to_available(
                _coerce_model_list(task_candidates.get(task)),
                available_set=available_set,
                catalog=catalog,
            )
            if not filtered:
                filtered = _suggest_models_for_task_tier(
                    available_models=available,
                    task=task,
                    tier="medium",
                    catalog=catalog,
                    limit=min(4, len(available)),
                )
            if filtered:
                task_candidates[task] = filtered

    if alignment_notes:
        explain["account_alignment"] = alignment_notes


def _filter_models_to_available(
    models: list[str],
    *,
    available_set: set[str],
    catalog: RouterCatalog,
) -> list[str]:
    filtered: list[str] = []
    for model in _dedupe(models):
        canonical = catalog.resolve_model_id(model)
        if canonical in available_set:
            filtered.append(canonical)
    return filtered


def _suggest_models_for_task_tier(
    *,
    available_models: list[str],
    task: str,
    tier: str,
    catalog: RouterCatalog,
    limit: int,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for model in available_models:
        entry = catalog.get_model(model)
        score = _metadata_score_for_task_tier(
            model=model,
            entry=entry,
            task=task,
            tier=tier,
        )
        scored.append((score, model))

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )
    return [model for _, model in scored[: max(1, limit)]]


def _metadata_score_for_task_tier(
    *,
    model: str,
    entry: ModelCatalogEntry,
    task: str,
    tier: str,
) -> float:
    capabilities = {
        item.strip().lower() for item in entry.capabilities if isinstance(item, str)
    }
    affinity = _task_affinity_score(
        model=model,
        task=task,
        capabilities=capabilities,
        explicit_affinity=entry.task_affinity,
        model_type=entry.type,
    )

    model_tier = _tier_rank(entry.tier, entry.priors.quality_bias)
    target_tier = {
        "low": 0.0,
        "medium": 1.0,
        "default": 1.0,
        "high": 2.0,
        "xhigh": 3.0,
    }.get(tier, 1.0)
    tier_bonus = 0.35 - 0.12 * abs(model_tier - target_tier)

    quality_component = float(entry.priors.quality_bias) * (
        0.75 if tier in {"high", "xhigh"} else 0.25
    )
    cost_total = float(entry.costs.input_per_1k) + float(entry.costs.output_per_1k)
    if tier == "low":
        cost_penalty = cost_total * 120.0
        latency_penalty = float(entry.priors.latency_ms) / 1800.0
    elif tier == "medium":
        cost_penalty = cost_total * 55.0
        latency_penalty = float(entry.priors.latency_ms) / 2600.0
    elif tier == "high":
        cost_penalty = cost_total * 18.0
        latency_penalty = float(entry.priors.latency_ms) / 5200.0
    else:
        cost_penalty = cost_total * 8.0
        latency_penalty = float(entry.priors.latency_ms) / 7000.0

    failure_penalty = float(entry.priors.failure_rate) * 4.0
    return (
        (affinity * 2.0)
        + tier_bonus
        + quality_component
        - cost_penalty
        - latency_penalty
        - failure_penalty
    )


def _task_affinity_score(
    *,
    model: str,
    task: str,
    capabilities: set[str],
    explicit_affinity: dict[str, float],
    model_type: str | None,
) -> float:
    explicit = explicit_affinity.get(task)
    if isinstance(explicit, (int, float)):
        return float(explicit)

    model_lower = model.lower()
    score = 0.4
    if task == "coding":
        score = 0.45
        if "reasoning" in capabilities:
            score += 0.25
        if "tool_use" in capabilities:
            score += 0.15
        if "json_mode" in capabilities:
            score += 0.05
        if any(token in model_lower for token in ("codex", "coder", "-code", "/code")):
            score += 0.55
    elif task == "thinking":
        score = 0.45
        if "reasoning" in capabilities:
            score += 0.5
        if "tool_use" in capabilities:
            score += 0.05
    elif task == "instruction_following":
        score = 0.5
        if "json_mode" in capabilities:
            score += 0.25
        if "tool_use" in capabilities:
            score += 0.05
    elif task == "image":
        if "image_generation" in capabilities:
            score = 1.1
        elif "vision" in capabilities or "image" in capabilities:
            score = 0.75
        else:
            score = -0.8
    elif task == "general":
        score = 0.55
        if "reasoning" in capabilities:
            score += 0.1
        if "tool_use" in capabilities:
            score += 0.05
        if "streaming" in capabilities:
            score += 0.05

    normalized_type = (model_type or "").strip().lower()
    if task == "coding" and normalized_type == "coding":
        score += 0.35
    if task == "image" and normalized_type in {"image", "vision", "multimodal"}:
        score += 0.2
    if task == "image" and normalized_type in {"coding", "text"}:
        score -= 0.35

    return score


def _tier_rank(raw_tier: str | None, quality_bias: float) -> float:
    tier = (raw_tier or "").strip().lower()
    tier_map = {
        "economy": 0.0,
        "cheap": 0.0,
        "balanced": 1.0,
        "standard": 1.0,
        "quality": 2.0,
        "premium": 2.0,
        "ultra": 3.0,
    }
    if tier in tier_map:
        return tier_map[tier]
    if quality_bias >= 0.76:
        return 3.0
    if quality_bias >= 0.62:
        return 2.0
    if quality_bias >= 0.45:
        return 1.0
    return 0.0


def _apply_guardrails(
    effective: dict[str, Any],
    profile: RouterProfileConfig,
    catalog: RouterCatalog,
    explain: dict[str, Any],
) -> None:
    task_routes = effective.get("task_routes")
    if isinstance(task_routes, dict):
        for task, route in task_routes.items():
            if not isinstance(task, str) or not isinstance(route, dict):
                continue
            thresholds = _thresholds_for_task(profile, task)
            for tier in ["low", "medium", "high", "xhigh", "default"]:
                models = _coerce_model_list(route.get(tier))
                if not models:
                    continue
                filtered = _prune_models(
                    models,
                    context=f"task_routes.{task}.{tier}",
                    thresholds=thresholds,
                    catalog=catalog,
                    explain=explain,
                )
                route[tier] = filtered

    fallback_models = _coerce_model_list(effective.get("fallback_models"))
    effective["fallback_models"] = _prune_models(
        fallback_models,
        context="fallback_models",
        thresholds=_thresholds_for_task(profile, None),
        catalog=catalog,
        explain=explain,
    )

    learned = effective.get("learned_routing")
    if isinstance(learned, dict):
        task_candidates = learned.get("task_candidates")
        if isinstance(task_candidates, dict):
            for task, values in task_candidates.items():
                if not isinstance(task, str):
                    continue
                filtered = _prune_models(
                    _coerce_model_list(values),
                    context=f"learned_routing.task_candidates.{task}",
                    thresholds=_thresholds_for_task(profile, task),
                    catalog=catalog,
                    explain=explain,
                )
                task_candidates[task] = filtered

    default_model_raw = str(effective.get("default_model") or "").strip()
    if not default_model_raw:
        return

    default_model = catalog.resolve_model_id(default_model_raw)
    default_entry = catalog.get_model(default_model)
    global_thresholds = _thresholds_for_task(profile, None)
    default_reasons = _guardrail_reasons(default_entry, global_thresholds)
    if not default_reasons:
        effective["default_model"] = default_model
        return

    candidate_pool = [
        *_coerce_model_list(effective.get("fallback_models")),
        *sorted(_collect_model_references(effective)),
    ]
    replacement = None
    for candidate in _dedupe(candidate_pool):
        candidate_model = catalog.resolve_model_id(candidate)
        candidate_entry = catalog.get_model(candidate_model)
        reasons = _guardrail_reasons(candidate_entry, global_thresholds)
        if not reasons:
            replacement = candidate_model
            break

    if replacement is None:
        raise CatalogValidationError(
            [
                (
                    "Guardrails removed the default model and no eligible replacement remains. "
                    f"default_model={default_model}"
                )
            ]
        )

    explain.setdefault("guardrail_pruned", []).append(
        {
            "context": "default_model",
            "model": default_model,
            "reasons": default_reasons,
        }
    )
    explain["default_model_replaced"] = {
        "from": default_model,
        "to": replacement,
    }
    effective["default_model"] = replacement


def _coerce_model_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    output.append(normalized)
        return output
    return []


def _prune_models(
    models: list[str],
    *,
    context: str,
    thresholds: GuardrailThresholds,
    catalog: RouterCatalog,
    explain: dict[str, Any],
) -> list[str]:
    filtered: list[str] = []
    for model_ref in _dedupe(models):
        canonical = catalog.resolve_model_id(model_ref)
        entry = catalog.get_model(canonical)
        reasons = _guardrail_reasons(entry, thresholds)
        if reasons:
            explain.setdefault("guardrail_pruned", []).append(
                {
                    "context": context,
                    "model": canonical,
                    "reasons": reasons,
                }
            )
            continue
        filtered.append(canonical)
    return filtered


def _guardrail_reasons(
    entry: ModelCatalogEntry,
    thresholds: GuardrailThresholds,
) -> list[str]:
    reasons: list[str] = []
    if (
        thresholds.max_cost_input_per_1k is not None
        and entry.costs.input_per_1k > thresholds.max_cost_input_per_1k
    ):
        reasons.append("violated_max_cost_input_per_1k")
    if (
        thresholds.max_cost_output_per_1k is not None
        and entry.costs.output_per_1k > thresholds.max_cost_output_per_1k
    ):
        reasons.append("violated_max_cost_output_per_1k")
    if (
        thresholds.max_latency_ms is not None
        and entry.priors.latency_ms > thresholds.max_latency_ms
    ):
        reasons.append("violated_max_latency_ms")
    if (
        thresholds.max_failure_rate is not None
        and entry.priors.failure_rate > thresholds.max_failure_rate
    ):
        reasons.append("violated_max_failure_rate")
    return reasons


def _thresholds_for_task(
    profile: RouterProfileConfig,
    task: str | None,
) -> GuardrailThresholds:
    base = GuardrailThresholds(
        max_cost_input_per_1k=profile.guardrails.max_cost_input_per_1k,
        max_cost_output_per_1k=profile.guardrails.max_cost_output_per_1k,
        max_latency_ms=profile.guardrails.max_latency_ms,
        max_failure_rate=profile.guardrails.max_failure_rate,
    )
    if not task:
        return base

    task_overrides = profile.guardrails.per_task.get(task)
    if task_overrides is None:
        return base

    for key, value in task_overrides.model_dump(mode="python").items():
        if value is not None:
            setattr(base, key, value)
    return base


def _collect_model_references(raw: dict[str, Any]) -> set[str]:
    refs: set[str] = set()

    default_model = raw.get("default_model")
    if isinstance(default_model, str) and default_model.strip():
        refs.add(default_model)

    for value in _coerce_model_list(raw.get("fallback_models")):
        refs.add(value)

    models = raw.get("models")
    if isinstance(models, dict):
        for key in models:
            if isinstance(key, str) and key.strip():
                refs.add(key)
    elif isinstance(models, list):
        for value in _coerce_model_list(models):
            refs.add(value)

    model_profiles = raw.get("model_profiles")
    if isinstance(model_profiles, dict):
        for key in model_profiles:
            if isinstance(key, str) and key.strip():
                refs.add(key)

    task_routes = raw.get("task_routes")
    if isinstance(task_routes, dict):
        for route in task_routes.values():
            if not isinstance(route, dict):
                continue
            for tier in ["low", "medium", "high", "xhigh", "default"]:
                for value in _coerce_model_list(route.get(tier)):
                    refs.add(value)

    learned = raw.get("learned_routing")
    if isinstance(learned, dict):
        candidates = learned.get("task_candidates")
        if isinstance(candidates, dict):
            for values in candidates.values():
                for value in _coerce_model_list(values):
                    refs.add(value)

    accounts = raw.get("accounts")
    if isinstance(accounts, list):
        for account in accounts:
            if not isinstance(account, dict):
                continue
            for value in _coerce_model_list(account.get("models")):
                refs.add(value)

    return refs


def _apply_layer(
    effective: dict[str, Any],
    template: dict[str, Any],
    *,
    layer_name: str,
    explain: dict[str, Any],
) -> None:
    sanitized = {key: value for key, value in template.items() if key != "description"}
    _merge_dict(effective, sanitized)
    explain["profile_layers"].append(layer_name)


def _apply_per_task_profile_layer(
    effective: dict[str, Any],
    *,
    task: str,
    template: dict[str, Any],
    layer_name: str,
    explain: dict[str, Any],
) -> None:
    task_routes = template.get("task_routes")
    if isinstance(task_routes, dict):
        override_route = task_routes.get(task)
        if override_route is None and task != "general":
            override_route = task_routes.get("general")
        if isinstance(override_route, dict):
            effective.setdefault("task_routes", {})
            effective["task_routes"][task] = deepcopy(override_route)

    learned = template.get("learned_routing")
    if isinstance(learned, dict):
        task_candidates = learned.get("task_candidates")
        if isinstance(task_candidates, dict):
            override_candidates = task_candidates.get(task)
            if override_candidates is None and task != "general":
                override_candidates = task_candidates.get("general")
            if isinstance(override_candidates, list):
                effective.setdefault("learned_routing", {})
                effective["learned_routing"].setdefault("task_candidates", {})
                effective["learned_routing"]["task_candidates"][task] = deepcopy(
                    override_candidates
                )

    explain["profile_layers"].append(layer_name)


def _merge_dict(base: dict[str, Any], layer: dict[str, Any]) -> None:
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
            continue
        base[key] = deepcopy(value)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
