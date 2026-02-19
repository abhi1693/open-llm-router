from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import yaml

from open_llm_router.model_utils import (
    coerce_models_map as _coerce_models_map,
    normalize_model_metadata as _normalize_model_metadata,
)
from open_llm_router.persistence import YamlFileStore
from open_llm_router.sequence_utils import dedupe_preserving_order as _dedupe

DEFAULT_RETRY_STATUSES = [429, 500, 502, 503, 504]
DEFAULT_COMPLEXITY = {
    "low_max_chars": 1200,
    "medium_max_chars": 6000,
    "high_max_chars": 16000,
}
DEFAULT_CLASSIFIER_CALIBRATION = {
    "enabled": False,
    "min_samples": 30,
    "target_secondary_success_rate": 0.8,
    "secondary_low_confidence_min_confidence": 0.18,
    "secondary_mixed_signal_min_confidence": 0.35,
    "adjustment_step": 0.03,
    "deadband": 0.05,
    "min_threshold": 0.05,
    "max_threshold": 0.9,
}
DEFAULT_ROUTE_RERANKER = {
    "enabled": False,
    "backend": "local_embedding",
    "local_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "local_files_only": True,
    "local_max_length": 256,
    "similarity_weight": 0.35,
    "min_similarity": 0.0,
    "model_hints": {},
}
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CHATGPT_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CHATGPT_TOKEN_URL = "https://auth.openai.com/oauth/token"
CHATGPT_REDIRECT_URI = "http://localhost:1455/auth/callback"
CHATGPT_SCOPE = "openid profile email offline_access"
CHATGPT_ACCOUNT_CLAIM_PATH = "https://api.openai.com/auth"
__all__ = ["main", "httpx", "secrets", "webbrowser"]


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _qualify_model(provider: str, model: str) -> str:
    normalized_provider = provider.strip()
    normalized_model = model.strip()
    if not normalized_provider or "/" in normalized_model:
        return normalized_model
    return f"{normalized_provider}/{normalized_model}"


def _qualify_models(provider: str, models: list[str]) -> list[str]:
    return [_qualify_model(provider, model) for model in models]


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_key_value(entries: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE format, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Feature key is empty in: {raw}")
        parsed[key] = float(value.strip())
    return parsed


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class AccountConfigMutator:
    _NORMALIZED_OPTIONAL_FIELDS: dict[str, str] = {
        "api_key": "api_key",
        "api_key_env": "api_key_env",
        "oauth_access_token": "oauth_access_token",
        "oauth_access_token_env": "oauth_access_token_env",
        "oauth_refresh_token": "oauth_refresh_token",
        "oauth_refresh_token_env": "oauth_refresh_token_env",
        "oauth_expires_at_env": "oauth_expires_at_env",
        "oauth_token_url": "oauth_token_url",
        "oauth_client_id": "oauth_client_id",
        "oauth_client_id_env": "oauth_client_id_env",
        "oauth_client_secret": "oauth_client_secret",
        "oauth_client_secret_env": "oauth_client_secret_env",
        "oauth_account_id": "oauth_account_id",
        "oauth_account_id_env": "oauth_account_id_env",
        "organization": "organization",
        "project": "project",
    }

    @classmethod
    def apply_from_args(cls, *, account: dict[str, Any], args: argparse.Namespace) -> None:
        for arg_name, config_key in cls._NORMALIZED_OPTIONAL_FIELDS.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                account[config_key] = _normalize_optional(value)

        oauth_expires_at = getattr(args, "oauth_expires_at", None)
        if oauth_expires_at is not None:
            account["oauth_expires_at"] = oauth_expires_at


class AccountEntryManager:
    @staticmethod
    def find_or_create(*, data: dict[str, Any], name: str) -> dict[str, Any]:
        accounts = data["accounts"]
        account = _find_account(accounts, name)
        if account is None:
            account = {"name": name}
            accounts.append(account)
        return account

    @staticmethod
    def configure_core(
        *,
        account: dict[str, Any],
        provider: str,
        base_url: str,
        enabled: bool,
        auth_mode: str | None = None,
        default_auth_mode: str | None = None,
    ) -> None:
        account["provider"] = provider
        account["base_url"] = base_url
        account["enabled"] = enabled
        if auth_mode is not None:
            account["auth_mode"] = auth_mode
        elif default_auth_mode is not None and "auth_mode" not in account:
            account["auth_mode"] = default_auth_mode

    @staticmethod
    def merge_qualified_models(
        *,
        data: dict[str, Any],
        account: dict[str, Any],
        provider: str,
        raw_models: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        account_models = _qualify_models(provider, raw_models)
        raw_to_qualified = {
            raw_model: qualified_model
            for raw_model, qualified_model in zip(raw_models, account_models, strict=False)
        }
        if not account_models:
            return account_models, raw_to_qualified

        existing = _qualify_models(provider, account.get("models", []))
        account["models"] = _dedupe([*existing, *account_models])
        _add_models(data, account_models)
        return account_models, raw_to_qualified


def _drop_none_fields(data: dict[str, Any]) -> None:
    for key in [item_key for item_key, value in data.items() if value is None]:
        data.pop(key, None)


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _generate_pkce() -> tuple[str, str]:
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    raw = value.strip()
    if not raw:
        return None, None

    try:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            params = parse_qs(parsed.query)
            code = _first_query_param(params, "code")
            state = _first_query_param(params, "state")
            return code, state
    except Exception:
        pass

    if "#" in raw:
        code, state = raw.split("#", 1)
        return code.strip() or None, state.strip() or None

    if "code=" in raw:
        params = parse_qs(raw)
        code = _first_query_param(params, "code")
        state = _first_query_param(params, "state")
        return code, state

    return raw, None


def _first_query_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    first = values[0].strip()
    return first or None


def _extract_chatgpt_account_id(access_token: str) -> str | None:
    parts = access_token.split(".")
    if len(parts) != 3:
        return None

    payload_b64 = parts[1]
    padding = "=" * ((4 - len(payload_b64) % 4) % 4)
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        )
    except Exception:
        return None

    claim = payload.get(CHATGPT_ACCOUNT_CLAIM_PATH)
    if isinstance(claim, dict):
        account_id = claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id.strip():
            return account_id.strip()
    return None


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        state = _first_query_param(params, "state")
        code = _first_query_param(params, "code")
        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return
        if state != getattr(server, "expected_state", None):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            return

        setattr(server, "auth_code", code)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<!doctype html><html><body><p>Authentication successful. Return to your terminal.</p></body></html>"
        )


def _start_callback_server(
    host: str, port: int, expected_state: str
) -> tuple[ThreadingHTTPServer | None, threading.Thread | None]:
    try:
        server = ThreadingHTTPServer((host, port), _OAuthCallbackHandler)
    except OSError:
        return None, None

    setattr(server, "expected_state", expected_state)
    setattr(server, "auth_code", None)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _wait_for_callback_code(
    server: ThreadingHTTPServer, timeout_seconds: int
) -> str | None:
    deadline = time.time() + max(1, timeout_seconds)
    while time.time() < deadline:
        code = getattr(server, "auth_code", None)
        if isinstance(code, str) and code.strip():
            return code.strip()
        time.sleep(0.1)
    return None


def _run_chatgpt_oauth_login_flow(args: argparse.Namespace) -> dict[str, Any]:
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    auth_params = {
        "response_type": "code",
        "client_id": args.client_id,
        "redirect_uri": args.redirect_uri,
        "scope": args.scope,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": args.originator,
    }
    auth_url = f"{args.authorize_url}?{urlencode(auth_params)}"

    server: ThreadingHTTPServer | None = None
    if not args.no_local_callback:
        parsed_redirect = urlparse(args.redirect_uri)
        host = parsed_redirect.hostname or "127.0.0.1"
        port = parsed_redirect.port or 1455
        if host in {"localhost", "127.0.0.1"}:
            server, _thread = _start_callback_server(
                host="127.0.0.1", port=port, expected_state=state
            )

    print(f"\nOpen this URL in your browser and complete sign-in:\n{auth_url}\n")
    force_paste_mode = bool(getattr(args, "paste_url", False))
    browser_opened = False
    if force_paste_mode:
        print("Paste mode enabled; skipping browser auto-open and callback wait.")
    elif not args.no_browser:
        try:
            browser_opened = bool(webbrowser.open(auth_url))
        except Exception:
            browser_opened = False
    else:
        print("Browser auto-open disabled; using manual paste flow.")

    code: str | None = None
    if args.manual_code:
        code, input_state = _parse_authorization_input(args.manual_code)
        if input_state and input_state != state:
            raise ValueError("State mismatch in provided manual code/URL.")
    elif server is not None and browser_opened:
        print(f"Waiting for OAuth callback on {args.redirect_uri} ...")
        code = _wait_for_callback_code(server, args.timeout_seconds)
        if not code:
            print("Callback not received in time.")
    elif server is not None:
        print(
            "Browser auto-open is unavailable. Complete login manually and paste the full redirect URL or code."
        )

    if not code:
        manual = input("Paste authorization code (or full redirect URL): ").strip()
        code, input_state = _parse_authorization_input(manual)
        if input_state and input_state != state:
            raise ValueError("State mismatch in pasted code/URL.")

    if server is not None:
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass

    if not code:
        raise ValueError("Missing authorization code.")

    response = httpx.post(
        args.token_url,
        data={
            "grant_type": "authorization_code",
            "client_id": args.client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": args.redirect_uri,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OAuth token exchange failed ({response.status_code}): {response.text}"
        )

    body = response.json()
    access_token = str(body.get("access_token") or "").strip()
    refresh_token = str(body.get("refresh_token") or "").strip()
    expires_in = body.get("expires_in")

    if (
        not access_token
        or not refresh_token
        or not isinstance(expires_in, (int, float))
    ):
        raise RuntimeError(
            "OAuth response missing required fields: access_token/refresh_token/expires_in."
        )

    expires_at = int(time.time()) + int(expires_in)
    account_id = _extract_chatgpt_account_id(access_token)
    return {
        "oauth_access_token": access_token,
        "oauth_refresh_token": refresh_token,
        "oauth_expires_at": expires_at,
        "oauth_account_id": account_id,
        "oauth_client_id": args.client_id,
        "oauth_token_url": args.token_url,
    }


def _load_config(path: Path) -> dict[str, Any]:
    data = YamlFileStore(path).load(default={})
    if not isinstance(data, dict):
        raise ValueError(f"Expected root object in {path}, found: {type(data).__name__}")
    _ensure_schema(data)
    return data


def _ensure_schema(data: dict[str, Any]) -> None:
    data.setdefault("default_model", "")
    data["models"] = _coerce_models_map(data.get("models"))
    data.setdefault("model_profiles", {})
    data.setdefault("accounts", [])
    data.setdefault("task_routes", {})
    data.setdefault("fallback_models", [])
    data.setdefault("retry_statuses", list(DEFAULT_RETRY_STATUSES))
    data.setdefault("complexity", dict(DEFAULT_COMPLEXITY))
    data.setdefault("classifier_calibration", {})
    classifier_calibration = data["classifier_calibration"]
    for key, value in DEFAULT_CLASSIFIER_CALIBRATION.items():
        classifier_calibration.setdefault(key, value)
    data.setdefault("route_reranker", {})
    route_reranker = data["route_reranker"]
    if not isinstance(route_reranker, dict):
        route_reranker = {}
        data["route_reranker"] = route_reranker
    for reranker_key, reranker_value in DEFAULT_ROUTE_RERANKER.items():
        if reranker_key == "model_hints" and isinstance(reranker_value, dict):
            route_reranker.setdefault(reranker_key, dict(reranker_value))
        else:
            route_reranker.setdefault(reranker_key, reranker_value)
    data.setdefault("learned_routing", {})
    learned = data["learned_routing"]
    learned.setdefault("enabled", False)
    learned.setdefault("bias", -4.0)
    learned.setdefault("default_output_tokens", 512)
    learned.setdefault("feature_weights", {})
    learned.setdefault("task_candidates", {})
    learned.setdefault("utility_weights", {})
    learned["utility_weights"].setdefault("cost", 12.0)
    learned["utility_weights"].setdefault("latency", 0.2)
    learned["utility_weights"].setdefault("failure", 3.0)


def _ensure_default_model(data: dict[str, Any]) -> None:
    current = str(data.get("default_model") or "").strip()
    if current:
        return
    models = _coerce_models_map(data.get("models"))
    data["models"] = models
    if models:
        data["default_model"] = next(iter(models.keys()))


def _save_config(path: Path, data: dict[str, Any]) -> None:
    YamlFileStore(path).write(data, sort_keys=False)


def _find_account(accounts: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for account in accounts:
        if account.get("name") == name:
            return account
    return None


def _add_models(data: dict[str, Any], models: list[str]) -> None:
    models_map = _coerce_models_map(data.get("models"))
    for model in models:
        normalized = model.strip()
        if normalized and normalized not in models_map:
            models_map[normalized] = _normalize_model_metadata(normalized, {})
    data["models"] = models_map


def cmd_add_account(args: argparse.Namespace, data: dict[str, Any]) -> str:
    account = AccountEntryManager.find_or_create(data=data, name=args.name)
    AccountEntryManager.configure_core(
        account=account,
        provider=args.provider,
        base_url=args.base_url,
        enabled=not args.disabled,
        auth_mode=args.auth_mode,
        default_auth_mode="api_key",
    )

    AccountConfigMutator.apply_from_args(account=account, args=args)

    account_models, _ = AccountEntryManager.merge_qualified_models(
        data=data,
        account=account,
        provider=args.provider,
        raw_models=_parse_csv(args.models),
    )

    if args.set_default and account_models:
        data["default_model"] = account_models[0]

    _drop_none_fields(account)
    _ensure_default_model(data)
    return f"Account '{args.name}' saved."


def cmd_login_chatgpt(args: argparse.Namespace, data: dict[str, Any]) -> str:
    if args.provider != "openai-codex":
        raise ValueError("login-chatgpt only supports --provider openai-codex.")

    account = AccountEntryManager.find_or_create(data=data, name=args.account)
    AccountEntryManager.configure_core(
        account=account,
        provider=args.provider,
        base_url=args.base_url,
        enabled=True,
        auth_mode="oauth",
    )

    oauth_data = _run_chatgpt_oauth_login_flow(args)
    account.update(oauth_data)

    raw_account_models = _parse_csv(args.models)
    account_models, raw_to_qualified = AccountEntryManager.merge_qualified_models(
        data=data,
        account=account,
        provider=args.provider,
        raw_models=raw_account_models,
    )

    if args.set_default and account_models:
        data["default_model"] = account_models[0]
    elif account_models:
        current_default = str(data.get("default_model") or "").strip()
        if current_default:
            if current_default in raw_to_qualified:
                data["default_model"] = raw_to_qualified[current_default]
            else:
                qualified_default = _qualify_model(args.provider, current_default)
                if qualified_default in account["models"]:
                    data["default_model"] = qualified_default
    _drop_none_fields(account)
    _ensure_default_model(data)

    account_id = account.get("oauth_account_id")
    if account_id:
        return f"ChatGPT OAuth login saved for account '{args.account}' (account_id: {account_id})."
    return f"ChatGPT OAuth login saved for account '{args.account}'."


def cmd_add_model(args: argparse.Namespace, data: dict[str, Any]) -> str:
    _add_models(data, [args.model])
    if args.account:
        account = _find_account(data["accounts"], args.account)
        if account is None:
            raise ValueError(
                f"Account '{args.account}' not found. Add it first with add-account."
            )
        account["models"] = _dedupe([*account.get("models", []), args.model])

    if args.set_default:
        data["default_model"] = args.model
    _ensure_default_model(data)
    return f"Model '{args.model}' added."


def cmd_set_route(args: argparse.Namespace, data: dict[str, Any]) -> str:
    route = data["task_routes"].setdefault(args.task, {})
    models = _parse_csv(args.model)
    if not models:
        raise ValueError("At least one model is required for --model.")
    route[args.tier] = models
    _add_models(data, models)
    _ensure_default_model(data)
    if len(models) == 1:
        return f"Route {args.task}.{args.tier} -> {models[0]}"
    return f"Route {args.task}.{args.tier} -> {', '.join(models)}"


def cmd_set_fallbacks(args: argparse.Namespace, data: dict[str, Any]) -> str:
    models = _parse_csv(args.models)
    if args.append:
        data["fallback_models"] = _dedupe([*data.get("fallback_models", []), *models])
    else:
        data["fallback_models"] = _dedupe(models)
    _add_models(data, models)
    return "Fallback models updated."


def cmd_set_profile(args: argparse.Namespace, data: dict[str, Any]) -> str:
    profile = data["model_profiles"].setdefault(args.model, {})
    maybe_values = {
        "quality_bias": args.quality_bias,
        "quality_sensitivity": args.quality_sensitivity,
        "cost_input_per_1k": args.cost_input_per_1k,
        "cost_output_per_1k": args.cost_output_per_1k,
        "latency_ms": args.latency_ms,
        "failure_rate": args.failure_rate,
    }
    for key, value in maybe_values.items():
        if value is not None:
            profile[key] = value
    _add_models(data, [args.model])
    _ensure_default_model(data)
    return f"Profile updated for model '{args.model}'."


def cmd_set_candidates(args: argparse.Namespace, data: dict[str, Any]) -> str:
    models = _parse_csv(args.models)
    learned = data["learned_routing"]
    learned["task_candidates"][args.task] = models
    _add_models(data, models)
    if args.enable:
        learned["enabled"] = True
    return f"Learned candidates set for task '{args.task}'."


def cmd_set_learned(args: argparse.Namespace, data: dict[str, Any]) -> str:
    learned = data["learned_routing"]
    if args.enabled is not None:
        learned["enabled"] = _parse_bool(args.enabled)
    if args.bias is not None:
        learned["bias"] = args.bias
    if args.default_output_tokens is not None:
        learned["default_output_tokens"] = args.default_output_tokens
    if args.utility_cost is not None:
        learned["utility_weights"]["cost"] = args.utility_cost
    if args.utility_latency is not None:
        learned["utility_weights"]["latency"] = args.utility_latency
    if args.utility_failure is not None:
        learned["utility_weights"]["failure"] = args.utility_failure
    if args.clear_features:
        learned["feature_weights"] = {}
    if args.set_feature:
        learned["feature_weights"].update(_parse_key_value(args.set_feature))
    return "Learned routing settings updated."


def cmd_show(_: argparse.Namespace, data: dict[str, Any]) -> str:
    models_map = _coerce_models_map(data.get("models"))
    summary = {
        "default_model": data.get("default_model"),
        "models_count": len(models_map),
        "accounts": [account.get("name") for account in data.get("accounts", [])],
        "tasks": sorted(data.get("task_routes", {}).keys()),
        "learned_enabled": bool(data.get("learned_routing", {}).get("enabled", False)),
    }
    return yaml.safe_dump(summary, sort_keys=False).rstrip()


class RouterConfigCliParserBuilder:
    _MODEL_HELP_SINGLE = (
        "Model id (e.g. gpt-5.2) or provider-qualified model (provider/modelId)."
    )
    _MODEL_LIST_HELP = (
        "Comma-separated model list. Model ids may be plain or provider/modelId."
    )

    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            prog="router",
            description="Manage open-llm-router router.yaml entries from CLI.",
        )
        self._parser.add_argument(
            "--path", default="router.yaml", help="Path to router YAML config."
        )
        self._parser.add_argument(
            "--dry-run", action="store_true", help="Print changes without writing file."
        )
        self._subparsers = self._parser.add_subparsers(dest="command", required=True)

    def build(self) -> argparse.ArgumentParser:
        self._build_add_account()
        self._build_login_chatgpt()
        self._build_add_model()
        self._build_set_route()
        self._build_set_fallbacks()
        self._build_set_profile()
        self._build_set_candidates()
        self._build_set_learned()
        self._build_show()
        return self._parser

    def _add_models_argument(
        self,
        parser: argparse.ArgumentParser,
        *,
        required: bool,
        default: str | None,
        help_text: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"required": required}
        if default is not None:
            kwargs["default"] = default
        if help_text:
            kwargs["help"] = help_text
        parser.add_argument("--models", **kwargs)

    @staticmethod
    def _add_set_default_argument(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--set-default", action="store_true")

    def _build_add_account(self) -> None:
        parser = self._subparsers.add_parser(
            "add-account", help="Add or update a backend account/provider."
        )
        parser.add_argument("--name", required=True)
        parser.add_argument("--provider", required=True)
        parser.add_argument("--base-url", required=True)
        parser.add_argument(
            "--auth-mode",
            choices=["api_key", "oauth", "passthrough"],
            help="Backend auth mode for this account.",
        )
        parser.add_argument("--api-key")
        parser.add_argument("--api-key-env")
        parser.add_argument("--oauth-access-token")
        parser.add_argument("--oauth-access-token-env")
        parser.add_argument("--oauth-refresh-token")
        parser.add_argument("--oauth-refresh-token-env")
        parser.add_argument("--oauth-expires-at", type=int)
        parser.add_argument("--oauth-expires-at-env")
        parser.add_argument("--oauth-token-url")
        parser.add_argument("--oauth-client-id")
        parser.add_argument("--oauth-client-id-env")
        parser.add_argument("--oauth-client-secret")
        parser.add_argument("--oauth-client-secret-env")
        parser.add_argument("--oauth-account-id")
        parser.add_argument("--oauth-account-id-env")
        parser.add_argument("--organization")
        parser.add_argument("--project")
        self._add_models_argument(parser, required=False, default="", help_text=None)
        self._add_set_default_argument(parser)
        parser.add_argument("--disabled", action="store_true")
        parser.set_defaults(handler=cmd_add_account, mutates=True)

    def _build_login_chatgpt(self) -> None:
        parser = self._subparsers.add_parser(
            "login-chatgpt",
            help="Run ChatGPT/Codex OAuth login and save tokens into an account.",
        )
        parser.add_argument("--account", required=True)
        parser.add_argument("--provider", required=True, choices=["openai-codex"])
        parser.add_argument("--base-url", default="https://chatgpt.com/backend-api")
        self._add_models_argument(parser, required=False, default="", help_text=None)
        self._add_set_default_argument(parser)
        parser.add_argument("--client-id", default=CHATGPT_CLIENT_ID)
        parser.add_argument("--authorize-url", default=CHATGPT_AUTHORIZE_URL)
        parser.add_argument("--token-url", default=CHATGPT_TOKEN_URL)
        parser.add_argument("--redirect-uri", default=CHATGPT_REDIRECT_URI)
        parser.add_argument("--scope", default=CHATGPT_SCOPE)
        parser.add_argument("--originator", default="pi")
        parser.add_argument("--manual-code")
        parser.add_argument(
            "--paste-url",
            action="store_true",
            help=(
                "Force manual paste flow (do not open browser; "
                "paste redirect URL/code in terminal)."
            ),
        )
        parser.add_argument("--timeout-seconds", type=int, default=180)
        parser.add_argument("--no-browser", action="store_true")
        parser.add_argument("--no-local-callback", action="store_true")
        parser.set_defaults(handler=cmd_login_chatgpt, mutates=True)

    def _build_add_model(self) -> None:
        parser = self._subparsers.add_parser(
            "add-model", help="Add model globally and optionally to one account."
        )
        parser.add_argument(
            "--model",
            required=True,
            help=self._MODEL_HELP_SINGLE,
        )
        parser.add_argument("--account")
        self._add_set_default_argument(parser)
        parser.set_defaults(handler=cmd_add_model, mutates=True)

    def _build_set_route(self) -> None:
        parser = self._subparsers.add_parser(
            "set-route", help="Set task routing tier model."
        )
        parser.add_argument("--task", required=True)
        parser.add_argument(
            "--tier",
            required=True,
            choices=["low", "medium", "high", "xhigh", "default"],
        )
        parser.add_argument(
            "--model",
            required=True,
            help=(
                "Comma-separated model list (single value is supported). "
                "Each value may be modelId or provider/modelId."
            ),
        )
        parser.set_defaults(handler=cmd_set_route, mutates=True)

    def _build_set_fallbacks(self) -> None:
        parser = self._subparsers.add_parser(
            "set-fallbacks", help="Set or append fallback models."
        )
        self._add_models_argument(
            parser,
            required=True,
            default=None,
            help_text=self._MODEL_LIST_HELP,
        )
        parser.add_argument("--append", action="store_true")
        parser.set_defaults(handler=cmd_set_fallbacks, mutates=True)

    def _build_set_profile(self) -> None:
        parser = self._subparsers.add_parser(
            "set-profile", help="Set model profile fields."
        )
        parser.add_argument(
            "--model",
            required=True,
            help=self._MODEL_HELP_SINGLE,
        )
        parser.add_argument("--quality-bias", type=float)
        parser.add_argument("--quality-sensitivity", type=float)
        parser.add_argument("--cost-input-per-1k", type=float)
        parser.add_argument("--cost-output-per-1k", type=float)
        parser.add_argument("--latency-ms", type=float)
        parser.add_argument("--failure-rate", type=float)
        parser.set_defaults(handler=cmd_set_profile, mutates=True)

    def _build_set_candidates(self) -> None:
        parser = self._subparsers.add_parser(
            "set-candidates", help="Set learned-routing candidate models for a task."
        )
        parser.add_argument("--task", required=True)
        self._add_models_argument(
            parser,
            required=True,
            default=None,
            help_text=self._MODEL_LIST_HELP,
        )
        parser.add_argument("--enable", action="store_true")
        parser.set_defaults(handler=cmd_set_candidates, mutates=True)

    def _build_set_learned(self) -> None:
        parser = self._subparsers.add_parser(
            "set-learned", help="Set learned-routing options and weights."
        )
        parser.add_argument("--enabled", help="true/false")
        parser.add_argument("--bias", type=float)
        parser.add_argument("--default-output-tokens", type=int)
        parser.add_argument("--utility-cost", type=float)
        parser.add_argument("--utility-latency", type=float)
        parser.add_argument("--utility-failure", type=float)
        parser.add_argument(
            "--set-feature",
            action="append",
            help="Feature weight KEY=VALUE (repeatable).",
        )
        parser.add_argument("--clear-features", action="store_true")
        parser.set_defaults(handler=cmd_set_learned, mutates=True)

    def _build_show(self) -> None:
        parser = self._subparsers.add_parser(
            "show", help="Show a short config summary."
        )
        parser.set_defaults(handler=cmd_show, mutates=False)


def _build_parser() -> argparse.ArgumentParser:
    return RouterConfigCliParserBuilder().build()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.path)
    data = _load_config(config_path)

    result = args.handler(args, data)
    if getattr(args, "mutates", False):
        _ensure_default_model(data)
        if args.dry_run:
            print(yaml.safe_dump(data, sort_keys=False).rstrip())
        else:
            _save_config(config_path, data)

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
