from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    backend_base_url: str = "http://localhost:11434"
    backend_api_key: str | None = None
    backend_timeout_seconds: float = 120.0
    backend_connect_timeout_seconds: float = 5.0
    backend_read_timeout_seconds: float = 30.0
    backend_write_timeout_seconds: float = 30.0
    backend_pool_timeout_seconds: float = 5.0
    routing_config_path: str = "router.profile.yaml"
    ingress_auth_required: bool = False
    ingress_api_keys: str = ""
    oauth_enabled: bool = False
    oauth_issuer: str | None = None
    oauth_audience: str | None = None
    oauth_jwks_url: str | None = None
    oauth_algorithms: str = "RS256"
    oauth_required_scopes: str = ""
    oauth_jwt_secret: str | None = None
    oauth_clock_skew_seconds: int = 30
    router_audit_log_enabled: bool = True
    router_audit_log_path: str = "logs/router_decisions.jsonl"
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: float = 30.0
    circuit_breaker_half_open_max_requests: int = 1
    idempotency_enabled: bool = True
    idempotency_ttl_seconds: int = 120
    idempotency_wait_timeout_seconds: float = 30.0
    redis_url: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
    )

    @property
    def oauth_is_configured(self) -> bool:
        return self.oauth_enabled and bool(self.oauth_jwt_secret or self.oauth_jwks_url)

    @property
    def ingress_api_keys_list(self) -> list[str]:
        return _split_csv(self.ingress_api_keys)

    @property
    def oauth_algorithms_list(self) -> list[str]:
        values = _split_csv(self.oauth_algorithms)
        return values or ["RS256"]

    @property
    def oauth_required_scopes_list(self) -> list[str]:
        return _split_csv(self.oauth_required_scopes)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
