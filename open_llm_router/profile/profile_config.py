from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from open_llm_router.account_fields import AccountCommonFields


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


class ProfileAccountConfig(AccountCommonFields):
    model_config = ConfigDict(extra="forbid")


class RouterProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profile: ProfileSelection = Field(default_factory=ProfileSelection)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    accounts: list[ProfileAccountConfig] = Field(default_factory=list)
    raw_overrides: dict[str, Any] = Field(default_factory=dict)
