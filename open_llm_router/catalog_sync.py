from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from open_llm_router.utils.yaml_utils import load_yaml_dict, write_yaml_dict

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CATALOG_MODELS_PATH = (
    Path(__file__).resolve().parent / "catalog" / "models.yaml"
)

# Internal provider ids do not always match OpenRouter's model id prefixes.
PROVIDER_ID_ALIASES: dict[str, tuple[str, ...]] = {
    "openai-codex": ("openai",),
    "gemini": ("google", "gemini"),
}


@dataclass(frozen=True)
class CatalogSyncStats:
    total_local_models: int
    updated: int
    unchanged: int
    missing_remote: int
    missing_pricing: int


def fetch_openrouter_models(
    *,
    source_url: str = OPENROUTER_MODELS_URL,
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    response = httpx.get(
        source_url,
        timeout=timeout_seconds,
        headers={"Accept": "application/json"},
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Failed to fetch OpenRouter models ({response.status_code}): {response.text}"
        )

    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError(
            "Invalid OpenRouter models response: expected top-level object."
        )

    data = body.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Invalid OpenRouter models response: missing 'data' list.")

    normalized: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def load_catalog_models_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    raw = load_yaml_dict(path, error_message=f"Expected YAML object in '{path}'.")

    models = raw.get("models")
    if not isinstance(models, list):
        raise ValueError(f"Expected 'models' list in '{path}'.")

    return raw


def write_catalog_models_document(path: Path, payload: dict[str, Any]) -> None:
    write_yaml_dict(path, payload)


def sync_catalog_models_pricing(
    *,
    catalog_document: dict[str, Any],
    openrouter_models: list[dict[str, Any]],
) -> CatalogSyncStats:
    models = catalog_document.get("models")
    if not isinstance(models, list):
        raise ValueError("Catalog document missing 'models' list.")

    remote_by_canonical: dict[str, dict[str, Any]] = {}
    remote_by_suffix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in openrouter_models:
        canonical = item.get("id")
        if not isinstance(canonical, str):
            continue
        canonical = canonical.strip()
        if not canonical or "/" not in canonical:
            continue
        remote_by_canonical[canonical] = item
        _, suffix = canonical.split("/", 1)
        remote_by_suffix[suffix].append(item)

    updated = 0
    unchanged = 0
    missing_remote = 0
    missing_pricing = 0

    for entry in models:
        if not isinstance(entry, dict):
            continue
        local_provider_raw = entry.get("provider")
        local_id_raw = entry.get("id")
        if not isinstance(local_provider_raw, str) or not isinstance(local_id_raw, str):
            continue

        local_provider = local_provider_raw.strip()
        local_id = local_id_raw.strip()
        if not local_provider or not local_id:
            continue

        remote = _match_remote_model(
            local_provider=local_provider,
            local_id=local_id,
            aliases=entry.get("aliases"),
            remote_by_canonical=remote_by_canonical,
            remote_by_suffix=remote_by_suffix,
        )
        if remote is None:
            missing_remote += 1
            continue

        entry_updated = False
        remote_created = _extract_created_timestamp(remote)
        if remote_created is not None:
            current_created = _coerce_int(entry.get("created"))
            if current_created != remote_created:
                entry["created"] = remote_created
                entry_updated = True

        prompt_per_1k, completion_per_1k = _extract_costs_per_1k(remote)
        if prompt_per_1k is None or completion_per_1k is None:
            missing_pricing += 1
            if entry_updated:
                updated += 1
            continue

        costs = entry.get("costs")
        if not isinstance(costs, dict):
            costs = {}
            entry["costs"] = costs

        old_prompt = _coerce_number(costs.get("input_per_1k"))
        old_completion = _coerce_number(costs.get("output_per_1k"))
        if old_prompt == prompt_per_1k and old_completion == completion_per_1k:
            if entry_updated:
                updated += 1
                continue
            unchanged += 1
            continue

        costs["input_per_1k"] = prompt_per_1k
        costs["output_per_1k"] = completion_per_1k
        updated += 1

    return CatalogSyncStats(
        total_local_models=len(models),
        updated=updated,
        unchanged=unchanged,
        missing_remote=missing_remote,
        missing_pricing=missing_pricing,
    )


def _extract_costs_per_1k(
    remote_model: dict[str, Any],
) -> tuple[float | None, float | None]:
    pricing = remote_model.get("pricing")
    if not isinstance(pricing, dict):
        return None, None

    prompt_per_token = _coerce_number(pricing.get("prompt"))
    completion_per_token = _coerce_number(pricing.get("completion"))
    if prompt_per_token is None or completion_per_token is None:
        return None, None
    if prompt_per_token < 0 or completion_per_token < 0:
        return None, None

    # Keep stable decimal output for YAML and comparisons.
    return round(prompt_per_token * 1000.0, 12), round(
        completion_per_token * 1000.0, 12
    )


def _extract_created_timestamp(remote_model: dict[str, Any]) -> int | None:
    created = _coerce_int(remote_model.get("created"))
    if created is None:
        return None
    if created < 0:
        return None
    return created


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _match_remote_model(
    *,
    local_provider: str,
    local_id: str,
    aliases: Any,
    remote_by_canonical: dict[str, dict[str, Any]],
    remote_by_suffix: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    provider_candidates = _provider_candidates(local_provider)

    # 1) Exact provider/id mapping.
    for provider in provider_candidates:
        matched = remote_by_canonical.get(f"{provider}/{local_id}")
        if matched is not None:
            return matched

    # 2) Canonical alias match.
    for alias in _normalize_aliases(aliases):
        if "/" in alias:
            matched = remote_by_canonical.get(alias)
            if matched is not None:
                return matched
        for provider in provider_candidates:
            matched = remote_by_canonical.get(f"{provider}/{alias}")
            if matched is not None:
                return matched

    # 3) Unique suffix match with provider preference.
    matched_by_suffix = _pick_by_suffix(local_id, provider_candidates, remote_by_suffix)
    if matched_by_suffix is not None:
        return matched_by_suffix

    for alias in _normalize_aliases(aliases):
        matched_by_suffix = _pick_by_suffix(
            alias, provider_candidates, remote_by_suffix
        )
        if matched_by_suffix is not None:
            return matched_by_suffix

    return None


def _provider_candidates(local_provider: str) -> list[str]:
    candidates = [local_provider]
    candidates.extend(PROVIDER_ID_ALIASES.get(local_provider, ()))

    seen: set[str] = set()
    output: list[str] = []
    for item in candidates:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _normalize_aliases(aliases: Any) -> list[str]:
    if not isinstance(aliases, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _pick_by_suffix(
    suffix: str,
    provider_candidates: list[str],
    remote_by_suffix: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    matches = remote_by_suffix.get(suffix)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    for provider in provider_candidates:
        scoped = [
            item
            for item in matches
            if isinstance(item.get("id"), str)
            and str(item["id"]).startswith(f"{provider}/")
        ]
        if len(scoped) == 1:
            return scoped[0]
    return None
