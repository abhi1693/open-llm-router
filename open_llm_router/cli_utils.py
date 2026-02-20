from __future__ import annotations

from typing import Any

from open_llm_router.sequence_utils import dedupe_preserving_order


def parse_comma_separated_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def qualify_model(provider: str, model: str) -> str:
    normalized_provider = provider.strip()
    normalized_model = model.strip()
    if not normalized_provider or "/" in normalized_model:
        return normalized_model
    return f"{normalized_provider}/{normalized_model}"


def qualify_models(
    provider: str,
    models: list[str],
    *,
    dedupe: bool = False,
) -> list[str]:
    qualified = [qualify_model(provider, model) for model in models]
    if dedupe:
        return dedupe_preserving_order(qualified)
    return qualified


def drop_none_fields(data: dict[str, Any]) -> None:
    for key in [item_key for item_key, value in data.items() if value is None]:
        data.pop(key, None)


def without_none_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def build_router_summary(
    *,
    default_model: str | None,
    models_count: int,
    account_names: list[str],
    tasks: list[str],
    learned_enabled: bool,
) -> dict[str, Any]:
    return {
        "default_model": default_model,
        "models_count": models_count,
        "accounts": account_names,
        "tasks": tasks,
        "learned_enabled": learned_enabled,
    }
