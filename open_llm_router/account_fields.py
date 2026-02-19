from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AccountCommonFields(BaseModel):
    name: str
    provider: str
    base_url: str | None = None
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
