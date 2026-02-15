from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProfileSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str = "auto"
    per_task: dict[str, str] = Field(default_factory=dict)


class GuardrailThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cost_input_per_1k: float | None = None
    max_cost_output_per_1k: float | None = None
    max_latency_ms: float | None = None
    max_failure_rate: float | None = None


class GuardrailsConfig(GuardrailThresholds):
    per_task: dict[str, GuardrailThresholds] = Field(default_factory=dict)


class ProfileAccountConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
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


class RouterProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile: ProfileSelection = Field(default_factory=ProfileSelection)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    accounts: list[ProfileAccountConfig] = Field(default_factory=list)
    raw_overrides: dict[str, Any] = Field(default_factory=dict)
