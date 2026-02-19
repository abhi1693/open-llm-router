from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ComplexityConfig(BaseModel):
    low_max_chars: int = 1200
    medium_max_chars: int = 6000
    high_max_chars: int = 16000


class ClassifierCalibrationConfig(BaseModel):
    enabled: bool = False
    min_samples: int = 30
    target_secondary_success_rate: float = 0.8
    secondary_low_confidence_min_confidence: float = 0.18
    secondary_mixed_signal_min_confidence: float = 0.35
    adjustment_step: float = 0.03
    deadband: float = 0.05
    min_threshold: float = 0.05
    max_threshold: float = 0.9


class SemanticClassifierConfig(BaseModel):
    enabled: bool = False
    backend: Literal["prototype", "local_embedding"] = "prototype"
    local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = True
    local_max_length: int = 256
    min_confidence: float = 0.2


class RouteRerankerConfig(BaseModel):
    enabled: bool = False
    backend: Literal["local_embedding"] = "local_embedding"
    local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = True
    local_max_length: int = 256
    similarity_weight: float = 0.35
    min_similarity: float = 0.0
    model_hints: dict[str, str] = Field(default_factory=dict)


class UtilityWeights(BaseModel):
    cost: float = 12.0
    latency: float = 0.2
    failure: float = 3.0


class LearnedRoutingConfig(BaseModel):
    enabled: bool = False
    bias: float = -4.0
    default_output_tokens: int = 512
    feature_weights: dict[str, float] = Field(default_factory=dict)
    task_candidates: dict[str, list[str]] = Field(default_factory=dict)
    utility_weights: UtilityWeights = Field(default_factory=UtilityWeights)


class TaskRoute(BaseModel):
    low: str | list[str] | None = None
    medium: str | list[str] | None = None
    high: str | list[str] | None = None
    xhigh: str | list[str] | None = None
    default: str | list[str] | None = None

    @staticmethod
    def _coerce_models(value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if item:
                cleaned.append(item)
        return cleaned

    @staticmethod
    def _pick_first_non_empty(*candidates: str | list[str] | None) -> list[str]:
        for candidate in candidates:
            models = TaskRoute._coerce_models(candidate)
            if models:
                return models
        return []

    def pick(self, complexity: str) -> list[str]:
        if complexity == "low":
            return self._pick_first_non_empty(
                self.low, self.medium, self.high, self.xhigh, self.default
            )
        if complexity == "medium":
            return self._pick_first_non_empty(
                self.medium, self.high, self.xhigh, self.default
            )
        if complexity == "high":
            return self._pick_first_non_empty(self.high, self.xhigh, self.default)
        if complexity == "xhigh":
            return self._pick_first_non_empty(self.xhigh, self.high, self.default)
        return self._pick_first_non_empty(self.default, self.high, self.xhigh)


class BackendAccount(BaseModel):
    name: str
    provider: str = "openai"
    base_url: str
    auth_mode: Literal["api_key", "oauth", "passthrough"] = "api_key"
    api_key: str | None = None
    api_key_env: str | None = None
    oauth_access_token: str | None = None
    oauth_access_token_env: str | None = None
    oauth_refresh_token: str | None = None
    oauth_refresh_token_env: str | None = None
    oauth_expires_at: int | None = None
    oauth_expires_at_env: str | None = None
    oauth_token_url: str | None = None
    oauth_client_id: str | None = None
    oauth_client_id_env: str | None = None
    oauth_client_secret: str | None = None
    oauth_client_secret_env: str | None = None
    oauth_account_id: str | None = None
    oauth_account_id_env: str | None = None
    organization: str | None = None
    project: str | None = None
    models: list[str] = Field(default_factory=list)
    enabled: bool = True

    @staticmethod
    def _split_model_ref(model: str) -> tuple[str | None, str]:
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

    def supports_model(self, model: str) -> bool:
        def _matches_model_id(expected_model_id: str) -> bool:
            expected = expected_model_id.strip()
            if not expected:
                return False
            for configured in self.models:
                configured_provider, configured_model_id = self._split_model_ref(
                    configured
                )
                if configured_model_id != expected:
                    continue
                if (
                    configured_provider
                    and configured_provider.lower() != self.provider.strip().lower()
                ):
                    continue
                return True
            return False

        provider, model_id = self._split_model_ref(model)
        if not self.models:
            return True
        if provider:
            if self.provider.strip().lower() != provider.lower():
                return False
            return model in self.models or _matches_model_id(model_id)
        return model in self.models or _matches_model_id(model)

    def upstream_model(self, requested_model: str) -> str:
        provider, model_id = self._split_model_ref(requested_model)
        if provider and self.provider.strip().lower() == provider.lower():
            return model_id
        return requested_model.strip()

    @staticmethod
    def _resolve_env_or_value(env_name: str | None, value: str | None) -> str | None:
        if env_name:
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                return env_value
        return value

    def resolved_api_key(self) -> str | None:
        if self.auth_mode != "api_key":
            return None
        return self._resolve_env_or_value(self.api_key_env, self.api_key)

    def resolved_oauth_access_token(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(
            self.oauth_access_token_env, self.oauth_access_token
        )

    def resolved_oauth_refresh_token(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(
            self.oauth_refresh_token_env, self.oauth_refresh_token
        )

    def resolved_oauth_client_id(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(
            self.oauth_client_id_env, self.oauth_client_id
        )

    def resolved_oauth_client_secret(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(
            self.oauth_client_secret_env, self.oauth_client_secret
        )

    def resolved_oauth_account_id(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(
            self.oauth_account_id_env, self.oauth_account_id
        )

    def resolved_oauth_expires_at(self) -> int | None:
        if self.auth_mode != "oauth":
            return None

        raw: str | int | None = self.oauth_expires_at
        if self.oauth_expires_at_env:
            env_value = os.getenv(self.oauth_expires_at_env, "").strip()
            if env_value:
                raw = env_value

        if raw is None:
            return None

        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def effective_oauth_token_url(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        if self.oauth_token_url:
            return self.oauth_token_url
        provider = self.provider.strip().lower()
        if provider in {"openai-codex", "openai", "chatgpt"}:
            return "https://auth.openai.com/oauth/token"
        return None

    def allows_passthrough_auth(self) -> bool:
        if self.auth_mode == "passthrough":
            return True
        if self.auth_mode == "api_key":
            return not bool(self.resolved_api_key())
        return False


class ModelProfile(BaseModel):
    quality_bias: float = 0.0
    quality_sensitivity: float = 1.0
    cost_input_per_1k: float = 0.0
    cost_output_per_1k: float = 0.0
    latency_ms: float = 0.0
    failure_rate: float = 0.0


class RoutingConfig(BaseModel):
    default_model: str
    task_routes: dict[str, TaskRoute]
    fallback_models: list[str] = Field(default_factory=list)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    model_profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    accounts: list[BackendAccount] = Field(default_factory=list)
    retry_statuses: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
    classifier_calibration: ClassifierCalibrationConfig = Field(
        default_factory=ClassifierCalibrationConfig
    )
    semantic_classifier: SemanticClassifierConfig = Field(
        default_factory=SemanticClassifierConfig
    )
    route_reranker: RouteRerankerConfig = Field(default_factory=RouteRerankerConfig)
    learned_routing: LearnedRoutingConfig = Field(default_factory=LearnedRoutingConfig)

    @staticmethod
    def _default_model_id(model_key: str) -> str:
        provider, sep, model_id = model_key.partition("/")
        if sep and provider.strip() and model_id.strip():
            return model_id.strip()
        return model_key

    @classmethod
    def _normalize_model_metadata(
        cls, model_key: str, raw_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        metadata = dict(raw_metadata or {})
        raw_id = metadata.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            metadata["id"] = raw_id.strip()
        else:
            metadata["id"] = cls._default_model_id(model_key)
        return metadata

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models(cls, value: Any) -> dict[str, dict[str, Any]]:
        if value is None:
            return {}
        if isinstance(value, list):
            coerced: dict[str, dict[str, Any]] = {}
            for item in value:
                if not isinstance(item, str):
                    continue
                model_key = item.strip()
                if model_key:
                    coerced.setdefault(
                        model_key, cls._normalize_model_metadata(model_key, {})
                    )
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
                    normalized_models[model_key] = cls._normalize_model_metadata(
                        model_key, {}
                    )
                    continue
                if not isinstance(raw_metadata, dict):
                    raise ValueError(
                        f"Model metadata for '{model_key}' must be an object."
                    )
                normalized_models[model_key] = cls._normalize_model_metadata(
                    model_key, raw_metadata
                )
            return normalized_models
        raise ValueError("Expected 'models' to be either a list or a mapping.")

    def should_auto_route(self, requested_model: str | None) -> bool:
        if not requested_model or not requested_model.strip():
            return True
        normalized = requested_model.strip().lower()
        return normalized in {"auto", "openrouter/auto"}

    def route_for(self, task: str, complexity: str) -> list[str]:
        route = self.task_routes.get(task) or self.task_routes.get("general")
        if route:
            candidates = route.pick(complexity)
            if candidates:
                return candidates
        return [self.default_model]

    def available_models(self) -> list[str]:
        discovered: set[str] = set(self.models.keys())
        discovered.add(self.default_model)
        discovered.update(self.fallback_models)
        for account in self.accounts:
            discovered.update(account.models)
        for route in self.task_routes.values():
            discovered.update(TaskRoute._coerce_models(route.low))
            discovered.update(TaskRoute._coerce_models(route.medium))
            discovered.update(TaskRoute._coerce_models(route.high))
            discovered.update(TaskRoute._coerce_models(route.xhigh))
            discovered.update(TaskRoute._coerce_models(route.default))
        return sorted(discovered)


def load_routing_config(config_path: str) -> RoutingConfig:
    routing_config, _ = load_routing_config_with_metadata(config_path)
    return routing_config


def load_routing_config_with_metadata(
    config_path: str,
) -> tuple[RoutingConfig, dict[str, Any] | None]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Routing config not found at '{config_path}'. "
            "Create it or set ROUTING_CONFIG_PATH."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML object in '{config_path}'.")

    from open_llm_router.catalog import (
        load_internal_catalog,
        validate_routing_document_against_catalog,
    )
    from open_llm_router.profile_compiler import (
        compile_profile_document,
        is_profile_document,
    )

    catalog = load_internal_catalog()
    explain_metadata: dict[str, Any] | None = None

    if is_profile_document(raw):
        compiled = compile_profile_document(raw, catalog=catalog)
        raw = compiled.effective_config
        explain_metadata = compiled.explain
    else:
        validate_routing_document_against_catalog(raw, catalog=catalog)

    return RoutingConfig.model_validate(raw), explain_metadata
