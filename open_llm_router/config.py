from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ComplexityConfig(BaseModel):
    low_max_chars: int = 1200
    medium_max_chars: int = 6000
    high_max_chars: int = 16000


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
    def _pick_first_non_empty(
        *candidates: str | list[str] | None
    ) -> list[str]:
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
        provider, model_id = self._split_model_ref(model)
        if provider:
            if self.provider.strip().lower() != provider.lower():
                return False
            return not self.models or model_id in self.models or model in self.models
        return not self.models or model in self.models

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
        return self._resolve_env_or_value(self.oauth_access_token_env, self.oauth_access_token)

    def resolved_oauth_refresh_token(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(self.oauth_refresh_token_env, self.oauth_refresh_token)

    def resolved_oauth_client_id(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(self.oauth_client_id_env, self.oauth_client_id)

    def resolved_oauth_client_secret(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(self.oauth_client_secret_env, self.oauth_client_secret)

    def resolved_oauth_account_id(self) -> str | None:
        if self.auth_mode != "oauth":
            return None
        return self._resolve_env_or_value(self.oauth_account_id_env, self.oauth_account_id)

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
    models: list[str] = Field(default_factory=list)
    model_profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    accounts: list[BackendAccount] = Field(default_factory=list)
    retry_statuses: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
    learned_routing: LearnedRoutingConfig = Field(default_factory=LearnedRoutingConfig)

    def should_auto_route(self, requested_model: str | None) -> bool:
        if not requested_model or not requested_model.strip():
            return True
        return requested_model.strip().lower() == "auto"

    def route_for(self, task: str, complexity: str) -> list[str]:
        route = self.task_routes.get(task) or self.task_routes.get("general")
        if route:
            candidates = route.pick(complexity)
            if candidates:
                return candidates
        return [self.default_model]

    def available_models(self) -> list[str]:
        discovered: set[str] = set(self.models)
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
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Routing config not found at '{config_path}'. "
            "Create it or set ROUTING_CONFIG_PATH."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return RoutingConfig.model_validate(raw)
