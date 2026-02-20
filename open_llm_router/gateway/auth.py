from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Request, status
from fastapi.responses import JSONResponse
from jwt import InvalidTokenError, PyJWKClient
from jwt.types import Options

from open_llm_router.config.settings import Settings


class AuthConfigurationError(RuntimeError):
    """Raised when OAuth authentication is enabled but misconfigured."""


@dataclass(slots=True)
class AuthResult:
    method: str
    principal: str
    scopes: set[str]
    claims: dict[str, Any] | None = None


class OAuthVerifier:
    def __init__(self, settings: Settings):
        self.algorithms = settings.oauth_algorithms_list
        self.audience = settings.oauth_audience
        self.issuer = settings.oauth_issuer
        self.required_scopes = set(settings.oauth_required_scopes_list)
        self.clock_skew = settings.oauth_clock_skew_seconds
        self.jwt_secret = settings.oauth_jwt_secret
        self.jwks_client: PyJWKClient | None = None

        if self.jwt_secret:
            return

        if settings.oauth_jwks_url:
            jwks_url = settings.oauth_jwks_url
        elif settings.oauth_issuer:
            jwks_url = settings.oauth_issuer.rstrip("/") + "/.well-known/jwks.json"
        else:
            raise AuthConfigurationError(
                "OAuth is enabled but no JWT verification source is configured. "
                "Set OAUTH_JWKS_URL, or set both OAUTH_ISSUER and OAUTH_JWKS_URL, "
                "or use OAUTH_JWT_SECRET for shared-secret tokens.",
            )

        self.jwks_client = PyJWKClient(jwks_url)

    def verify(self, token: str) -> AuthResult:
        if self.jwt_secret:
            signing_key = self.jwt_secret
        else:
            if not self.jwks_client:
                raise AuthConfigurationError("OAuth verifier is missing a JWKS client.")
            signing_key = self.jwks_client.get_signing_key_from_jwt(token).key

        options: Options = {"verify_aud": self.audience is not None}
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=self.algorithms,
            issuer=self.issuer,
            audience=self.audience,
            options=options,
            leeway=self.clock_skew,
        )

        scopes = _extract_scopes(claims)
        if self.required_scopes and not self.required_scopes.issubset(scopes):
            missing = sorted(self.required_scopes - scopes)
            raise InvalidTokenError(f"Missing required scopes: {', '.join(missing)}")

        principal = str(
            claims.get("sub")
            or claims.get("email")
            or claims.get("client_id")
            or "oauth-user",
        )
        return AuthResult(
            method="oauth",
            principal=principal,
            scopes=scopes,
            claims=claims,
        )


class Authenticator:
    def __init__(self, settings: Settings):
        self.required = settings.ingress_auth_required
        self.api_keys = set(settings.ingress_api_keys_list)
        self.oauth_verifier: OAuthVerifier | None = None

        if settings.oauth_enabled:
            self.oauth_verifier = OAuthVerifier(settings)

        if self.required and not self.api_keys and not self.oauth_verifier:
            raise AuthConfigurationError(
                "Ingress auth is required, but no API keys or OAuth verifier are configured.",
            )

    async def authenticate_request(self, request: Request) -> JSONResponse | None:
        if not self.required:
            return None

        auth_header = request.headers.get("authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return _unauthorized("Missing Bearer token.")

        bearer_token = token.strip()

        if bearer_token in self.api_keys:
            request.state.auth = AuthResult(
                method="api_key",
                principal="api-key-client",
                scopes=set(),
                claims=None,
            )
            return None

        if self.oauth_verifier:
            try:
                result = self.oauth_verifier.verify(bearer_token)
                request.state.auth = result
                return None
            except InvalidTokenError:
                return _unauthorized("Invalid OAuth token.")
            except Exception:
                return _unauthorized("Invalid OAuth token.")

        return _unauthorized("Invalid API key or OAuth token.")


def _extract_scopes(claims: dict[str, Any]) -> set[str]:
    raw_scope = claims.get("scope", claims.get("scp"))
    if isinstance(raw_scope, str):
        return {scope for scope in raw_scope.split() if scope}
    if isinstance(raw_scope, list):
        return {str(scope).strip() for scope in raw_scope if str(scope).strip()}
    return set()


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": "Bearer"},
        content={
            "error": {
                "message": message,
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            },
        },
    )
