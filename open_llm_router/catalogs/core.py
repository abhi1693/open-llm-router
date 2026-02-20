from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from open_llm_router.catalogs.paths import CatalogDataPaths
from open_llm_router.utils.yaml_utils import load_yaml_dict

if TYPE_CHECKING:
    from pathlib import Path


class CatalogValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Catalog validation failed:\n" + "\n".join(
            f"- {item}" for item in errors
        )
        super().__init__(message)


class CatalogLookupError(ValueError):
    def __init__(
        self,
        *,
        kind: str,
        raw_value: str,
        suggestions: list[str] | None = None,
        detail: str | None = None,
    ):
        self.kind = kind
        self.raw_value = raw_value
        self.suggestions = suggestions or []
        self.detail = detail
        suffix = ""
        if self.suggestions:
            suffix = f" Suggestions: {', '.join(self.suggestions)}"
        if detail:
            suffix = f" {detail}{suffix}"
        super().__init__(f"Unknown {kind} '{raw_value}'.{suffix}")

    def format_for_path(self, path: str) -> str:
        message = f"Unknown {self.kind} at {path}: '{self.raw_value}'."
        if self.detail:
            message += f" {self.detail}"
        if self.suggestions:
            message += f" Suggested canonical ids: {', '.join(self.suggestions)}"
        return message


class CatalogCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


class CatalogLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_tokens: int = 0
    max_output_tokens: int = 0
    min_input_chars: int = 0


class CatalogPriors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_bias: float = 0.0
    quality_sensitivity: float = 1.0
    latency_ms: float = 0.0
    failure_rate: float = 0.0


class ProviderCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    base_url: str
    auth_modes: list[str] = Field(default_factory=list)
    endpoint_defaults: dict[str, str] = Field(default_factory=dict)
    reliability_priors: dict[str, float] = Field(default_factory=dict)


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    provider: str
    created: int | None = None
    family: str | None = None
    type: str | None = None
    tier: str | None = None
    task_affinity: dict[str, float] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    costs: CatalogCost = Field(default_factory=CatalogCost)
    limits: CatalogLimits = Field(default_factory=CatalogLimits)
    capabilities: list[str] = Field(default_factory=list)
    priors: CatalogPriors = Field(default_factory=CatalogPriors)

    @property
    def canonical_id(self) -> str:
        return f"{self.provider}/{self.id}"


@dataclass(frozen=True)
class RouterCatalog:
    version: int
    providers: dict[str, ProviderCatalogEntry]
    models: dict[str, ModelCatalogEntry]
    alias_index: dict[str, set[str]]

    @property
    def provider_ids(self) -> list[str]:
        return sorted(self.providers)

    @property
    def model_ids(self) -> list[str]:
        return sorted(self.models)

    def resolve_provider_id(self, provider_id: str) -> str:
        normalized = provider_id.strip()
        if normalized in self.providers:
            return normalized

        lower_map = {key.lower(): key for key in self.providers}
        lowered = normalized.lower()
        if lowered in lower_map:
            return lower_map[lowered]

        suggestions = _suggest(normalized, self.provider_ids)
        if not suggestions:
            suggestions = self.provider_ids[:3]
        raise CatalogLookupError(
            kind="provider",
            raw_value=provider_id,
            suggestions=suggestions,
        )

    def get_provider(self, provider_id: str) -> ProviderCatalogEntry:
        canonical = self.resolve_provider_id(provider_id)
        return self.providers[canonical]

    def resolve_model_id(self, model_ref: str, provider_hint: str | None = None) -> str:
        normalized = model_ref.strip()
        if not normalized:
            raise CatalogLookupError(kind="model", raw_value=model_ref)

        normalized_provider_hint = None
        if provider_hint:
            normalized_provider_hint = self.resolve_provider_id(provider_hint)
            # Some providers expose model ids that include "/" (for example "z-ai/glm5").
            # When a provider hint exists, prefer exact scoped lookup before splitting by "/".
            hinted_key = f"{normalized_provider_hint}/{normalized}"
            if hinted_key in self.models:
                return hinted_key

        provider, model_id = _split_model_ref(normalized)
        if provider is not None:
            canonical_provider = self.resolve_provider_id(provider)
            key = f"{canonical_provider}/{model_id}"
            if key in self.models:
                return key

            alias_hits = self.alias_index.get(normalized, set())
            alias_hits |= self.alias_index.get(
                f"{canonical_provider}/{model_id}",
                set(),
            )
            alias_hits = {
                item for item in alias_hits if item.startswith(f"{canonical_provider}/")
            }
            if len(alias_hits) == 1:
                return next(iter(alias_hits))
            if len(alias_hits) > 1:
                raise CatalogLookupError(
                    kind="model",
                    raw_value=model_ref,
                    suggestions=sorted(alias_hits),
                    detail="Model alias is ambiguous.",
                )

            suggestions = _suggest(key, self.model_ids)
            if not suggestions:
                suggestions = self.model_ids[:3]
            raise CatalogLookupError(
                kind="model",
                raw_value=model_ref,
                suggestions=suggestions,
            )

        if normalized_provider_hint:
            scoped_key = f"{normalized_provider_hint}/{normalized}"
            if scoped_key in self.models:
                return scoped_key

        alias_hits = set(self.alias_index.get(normalized, set()))
        if normalized_provider_hint:
            alias_hits = {
                item
                for item in alias_hits
                if item.startswith(f"{normalized_provider_hint}/")
            }

        if len(alias_hits) == 1:
            return next(iter(alias_hits))
        if len(alias_hits) > 1:
            raise CatalogLookupError(
                kind="model",
                raw_value=model_ref,
                suggestions=sorted(alias_hits),
                detail="Model alias is ambiguous.",
            )

        by_suffix = [item for item in self.model_ids if item.endswith(f"/{normalized}")]
        if normalized_provider_hint:
            by_suffix = [
                item
                for item in by_suffix
                if item.startswith(f"{normalized_provider_hint}/")
            ]
        if len(by_suffix) == 1:
            return by_suffix[0]
        if len(by_suffix) > 1:
            raise CatalogLookupError(
                kind="model",
                raw_value=model_ref,
                suggestions=by_suffix,
                detail="Model id is ambiguous across providers.",
            )

        suggestions = _suggest(normalized, [*self.model_ids, *self.alias_index.keys()])
        if not suggestions:
            suggestions = self.model_ids[:3]
        raise CatalogLookupError(
            kind="model",
            raw_value=model_ref,
            suggestions=suggestions,
        )

    def get_model(
        self,
        model_ref: str,
        provider_hint: str | None = None,
    ) -> ModelCatalogEntry:
        canonical = self.resolve_model_id(model_ref, provider_hint=provider_hint)
        return self.models[canonical]


def _suggest(value: str, options: list[str], *, limit: int = 3) -> list[str]:
    return get_close_matches(value, options, n=limit, cutoff=0.45)


def _split_model_ref(model_ref: str) -> tuple[str | None, str]:
    provider, sep, model_id = model_ref.partition("/")
    if sep and provider.strip() and model_id.strip():
        return provider.strip(), model_id.strip()
    return None, model_ref.strip()


@dataclass(frozen=True)
class CatalogDocumentStore:
    providers_path: Path
    models_path: Path

    @classmethod
    def from_package_data(cls) -> CatalogDocumentStore:
        return cls(
            providers_path=CatalogDataPaths.providers_yaml(),
            models_path=CatalogDataPaths.models_yaml(),
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        return load_yaml_dict(path, error_message=f"Expected YAML object in {path}")

    def load_documents(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._load_yaml(self.providers_path), self._load_yaml(self.models_path)


def _build_alias_index(models: dict[str, ModelCatalogEntry]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for canonical_id, entry in models.items():
        aliases = {canonical_id, entry.id, *entry.aliases}
        for alias in aliases:
            normalized = str(alias).strip()
            if not normalized:
                continue
            index.setdefault(normalized, set()).add(canonical_id)
    return index


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_model_items_with_presets(
    models_raw: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_models = models_raw.get("models", [])
    if not isinstance(raw_models, list):
        msg = "Expected 'models' list in models catalog."
        raise TypeError(msg)

    raw_presets = models_raw.get("metadata_presets", {})
    if raw_presets is None:
        raw_presets = {}
    if not isinstance(raw_presets, dict):
        msg = "Expected 'metadata_presets' mapping in models catalog."
        raise TypeError(msg)

    expanded: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_models):
        if not isinstance(item, dict):
            msg = f"Expected object at models[{idx}] in models catalog."
            raise TypeError(msg)
        current = deepcopy(item)
        preset_name = current.pop("metadata_preset", None)
        if preset_name is None:
            expanded.append(current)
            continue
        if not isinstance(preset_name, str) or not preset_name.strip():
            msg = f"Expected non-empty string metadata_preset at models[{idx}]."
            raise ValueError(
                msg,
            )
        preset_key = preset_name.strip()
        preset_payload = raw_presets.get(preset_key)
        if not isinstance(preset_payload, dict):
            msg = f"Unknown metadata_preset '{preset_key}' referenced at models[{idx}]."
            raise TypeError(
                msg,
            )
        expanded.append(_merge_dict(preset_payload, current))
    return expanded


@lru_cache
def load_internal_catalog() -> RouterCatalog:
    providers_raw, models_raw = (
        CatalogDocumentStore.from_package_data().load_documents()
    )

    provider_entries: dict[str, ProviderCatalogEntry] = {}
    for item in providers_raw.get("providers", []):
        entry = ProviderCatalogEntry.model_validate(item)
        provider_entries[entry.id] = entry

    model_entries: dict[str, ModelCatalogEntry] = {}
    for item in _expand_model_items_with_presets(models_raw):
        model_entry = ModelCatalogEntry.model_validate(item)
        if model_entry.provider not in provider_entries:
            msg = f"Catalog model '{model_entry.canonical_id}' references unknown provider '{model_entry.provider}'."
            raise ValueError(
                msg,
            )
        model_entries[model_entry.canonical_id] = model_entry

    alias_index = _build_alias_index(model_entries)
    return RouterCatalog(
        version=int(models_raw.get("version") or providers_raw.get("version") or 1),
        providers=provider_entries,
        models=model_entries,
        alias_index=alias_index,
    )


def validate_routing_document_against_catalog(
    raw: dict[str, Any],
    *,
    catalog: RouterCatalog | None = None,
) -> None:
    catalog = catalog or load_internal_catalog()
    errors: list[str] = []

    accounts = raw.get("accounts")
    account_provider_by_index: dict[int, str] = {}
    if isinstance(accounts, list):
        for idx, account in enumerate(accounts):
            if not isinstance(account, dict):
                continue
            provider_value = account.get("provider")
            if isinstance(provider_value, str):
                try:
                    account_provider_by_index[idx] = catalog.resolve_provider_id(
                        provider_value,
                    )
                except CatalogLookupError as exc:
                    errors.append(exc.format_for_path(f"accounts[{idx}].provider"))

    def _validate_model(
        path: str,
        value: str,
        provider_hint: str | None = None,
    ) -> None:
        try:
            catalog.resolve_model_id(value, provider_hint=provider_hint)
        except CatalogLookupError as exc:
            errors.append(exc.format_for_path(path))

    for path, provider_hint, model_ref in _iter_model_references(
        raw,
        account_provider_by_index,
    ):
        _validate_model(path, model_ref, provider_hint=provider_hint)

    if errors:
        raise CatalogValidationError(errors)


def _iter_model_references(
    raw: dict[str, Any],
    account_provider_by_index: dict[int, str],
) -> list[tuple[str, str | None, str]]:
    refs: list[tuple[str, str | None, str]] = []

    default_model = raw.get("default_model")
    if isinstance(default_model, str) and default_model.strip():
        refs.append(("default_model", None, default_model))

    fallback_models = raw.get("fallback_models")
    if isinstance(fallback_models, list):
        for idx, model in enumerate(fallback_models):
            if isinstance(model, str) and model.strip():
                refs.append((f"fallback_models[{idx}]", None, model))

    model_profiles = raw.get("model_profiles")
    if isinstance(model_profiles, dict):
        refs.extend(
            (f"model_profiles.{model_key}", None, model_key)
            for model_key in model_profiles
            if isinstance(model_key, str) and model_key.strip()
        )

    models = raw.get("models")
    if isinstance(models, list):
        for idx, model in enumerate(models):
            if isinstance(model, str) and model.strip():
                refs.append((f"models[{idx}]", None, model))
    elif isinstance(models, dict):
        refs.extend(
            (f"models.{model_key}", None, model_key)
            for model_key in models
            if isinstance(model_key, str) and model_key.strip()
        )

    task_routes = raw.get("task_routes")
    if isinstance(task_routes, dict):
        for task, route in task_routes.items():
            if not isinstance(task, str) or not isinstance(route, dict):
                continue
            for tier, value in route.items():
                if isinstance(value, str) and value.strip():
                    refs.append((f"task_routes.{task}.{tier}", None, value))
                elif isinstance(value, list):
                    for idx, model in enumerate(value):
                        if isinstance(model, str) and model.strip():
                            refs.append(
                                (f"task_routes.{task}.{tier}[{idx}]", None, model),
                            )

    learned = raw.get("learned_routing")
    if isinstance(learned, dict):
        candidates = learned.get("task_candidates")
        if isinstance(candidates, dict):
            for task, values in candidates.items():
                if not isinstance(task, str) or not isinstance(values, list):
                    continue
                for idx, model in enumerate(values):
                    if isinstance(model, str) and model.strip():
                        refs.append(
                            (
                                f"learned_routing.task_candidates.{task}[{idx}]",
                                None,
                                model,
                            ),
                        )

    accounts = raw.get("accounts")
    if isinstance(accounts, list):
        for idx, account in enumerate(accounts):
            if not isinstance(account, dict):
                continue
            account_models = account.get("models")
            if not isinstance(account_models, list):
                continue
            provider_hint = account_provider_by_index.get(idx)
            for model_idx, model_ref in enumerate(account_models):
                if isinstance(model_ref, str) and model_ref.strip():
                    refs.append(
                        (
                            f"accounts[{idx}].models[{model_idx}]",
                            provider_hint,
                            model_ref,
                        ),
                    )

    return refs
