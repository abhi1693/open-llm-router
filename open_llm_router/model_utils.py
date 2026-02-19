from __future__ import annotations


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
