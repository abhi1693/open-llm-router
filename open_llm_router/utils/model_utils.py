from __future__ import annotations

from typing import Any


def default_model_id(model_key: str) -> str:
    provider, sep, model_id = model_key.partition("/")
    if sep and provider.strip() and model_id.strip():
        return model_id.strip()
    return model_key


def split_model_ref(model: str) -> tuple[str | None, str]:
    normalized = model.strip()
    if not normalized:
        return None, ""
    if "/" not in normalized:
        return None, normalized
    provider, model_id = normalized.split("/", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    if not provider or not model_id:
        return None, normalized
    return provider, model_id


def normalize_model_metadata(
    model_key: str,
    raw_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(raw_metadata or {})
    raw_id = metadata.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        metadata["id"] = raw_id.strip()
    else:
        metadata["id"] = default_model_id(model_key)
    return metadata


def coerce_models_map(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, list):
        coerced: dict[str, dict[str, Any]] = {}
        for item in value:
            if not isinstance(item, str):
                continue
            model_key = item.strip()
            if model_key:
                coerced.setdefault(model_key, normalize_model_metadata(model_key, {}))
        return coerced
    if isinstance(value, dict):
        normalized_models: dict[str, dict[str, Any]] = {}
        for raw_model_key, raw_metadata in value.items():
            if not isinstance(raw_model_key, str):
                continue
            model_key = raw_model_key.strip()
            if not model_key:
                continue
            if raw_metadata is None:
                normalized_models[model_key] = normalize_model_metadata(model_key, {})
                continue
            if not isinstance(raw_metadata, dict):
                msg = f"Model metadata for '{model_key}' must be an object."
                raise TypeError(msg)
            normalized_models[model_key] = normalize_model_metadata(
                model_key,
                raw_metadata,
            )
        return normalized_models
    msg = "Expected 'models' to be either a list or a mapping."
    raise ValueError(msg)
