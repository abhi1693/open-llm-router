from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, cast

import yaml

from open_llm_router.catalog import (
    load_internal_catalog,
    validate_routing_document_against_catalog,
)
from open_llm_router.catalog_sync import (
    DEFAULT_CATALOG_MODELS_PATH,
    OPENROUTER_MODELS_URL,
    fetch_openrouter_models,
    load_catalog_models_document,
    sync_catalog_models_pricing,
    write_catalog_models_document,
)
from open_llm_router.config import (
    RoutingConfig,
    load_routing_config_with_metadata,
)
from open_llm_router.profile_compiler import (
    compile_profile_document,
    compile_profile_file,
    get_builtin_profile_template,
    is_profile_document,
    list_builtin_profiles,
)
from open_llm_router.profile_config import RouterProfileConfig
from open_llm_router.scoring import build_routing_features, score_model
from open_llm_router.sequence_utils import dedupe_preserving_order as _dedupe

LOGIN_CHATGPT_DEFAULT_PROVIDER = "openai-codex"
LOGIN_CHATGPT_DEFAULT_MODELS = "gpt-5.2,gpt-5.2-codex"
OPENAI_APIKEY_DEFAULT_MODELS = "openai/gpt-5.2"
OPENAI_APIKEY_DEFAULT_KEY_ENV = "OPENAI_API_KEY"
GEMINI_APIKEY_DEFAULT_MODELS = "gemini/gemini-2.5-flash,gemini/gemini-2.5-flash-lite"
GEMINI_APIKEY_DEFAULT_KEY_ENV = "GEMINI_API_KEY"
NVIDIA_APIKEY_DEFAULT_MODELS = "nvidia/z-ai/glm5,nvidia/moonshotai/kimi-k2.5"
NVIDIA_APIKEY_DEFAULT_KEY_ENV = "NVIDIA_API_KEY"
GITHUB_APIKEY_DEFAULT_MODELS = "github/openai/gpt-4.1,github/openai/gpt-4.1-mini,github/meta/Llama-3.3-70B-Instruct"
GITHUB_APIKEY_DEFAULT_KEY_ENV = "GITHUB_TOKEN"
DEFAULT_RUNTIME_OVERRIDES_PATH = "logs/router.runtime.overrides.yaml"
DEFAULT_PROFILE_CONFIG_PATH = "router.profile.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in '{path}'.")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _add_profile_path_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--path", default=DEFAULT_PROFILE_CONFIG_PATH)


def cmd_init(args: argparse.Namespace) -> int:
    template = {
        "schema_version": 1,
        "profile": {
            "default": args.profile,
            "per_task": {},
        },
        "guardrails": {},
        "accounts": [],
        "raw_overrides": {},
    }

    output_path = Path(args.path)
    if output_path.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite existing file: {output_path}. Use --force to overwrite."
        )

    _write_yaml(output_path, template)
    print(f"Wrote profile config: {output_path}")
    return 0


def cmd_compile_config(args: argparse.Namespace) -> int:
    profile_path = Path(args.path)
    result = compile_profile_file(profile_path)

    if args.stdout:
        print(yaml.safe_dump(result.effective_config, sort_keys=False).rstrip())
    else:
        output_path = Path(args.output)
        _write_yaml(output_path, result.effective_config)
        print(f"Wrote effective config: {output_path}")

    if args.explain:
        print(yaml.safe_dump(result.explain, sort_keys=False).rstrip())

    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    config_path = Path(args.path)
    raw = _read_yaml(config_path)

    if is_profile_document(raw):
        compile_profile_file(config_path)
        print(f"Profile config is valid: {config_path}")
        return 0

    catalog = load_internal_catalog()
    validate_routing_document_against_catalog(raw, catalog=catalog)
    RoutingConfig.model_validate(raw)
    print(f"Routing config is valid: {config_path}")
    return 0


def cmd_profile_list(_: argparse.Namespace) -> int:
    rows = list_builtin_profiles()
    for name, description in rows:
        suffix = f" - {description}" if description else ""
        print(f"{name}{suffix}")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    template = get_builtin_profile_template(args.name)
    payload = {"name": args.name, **template}
    print(yaml.safe_dump(payload, sort_keys=False).rstrip())
    return 0


def cmd_explain_route(args: argparse.Namespace) -> int:
    config_path = Path(args.path)
    raw = _read_yaml(config_path)

    explain_meta: dict[str, Any] | None = None
    if is_profile_document(raw):
        compiled = compile_profile_file(config_path)
        config = RoutingConfig.model_validate(compiled.effective_config)
        explain_meta = compiled.explain
    else:
        config, explain_meta = load_routing_config_with_metadata(str(config_path))

    task = args.task
    complexity = args.complexity
    route_models = config.route_for(task=task, complexity=complexity)
    default_chain = _dedupe([*route_models, *config.fallback_models])

    result: dict[str, Any] = {
        "task": task,
        "complexity": complexity,
        "selected_profile_source": (explain_meta or {}).get("profile_layers"),
        "guardrail_pruned": _extract_guardrail_entries(explain_meta, task),
        "default_chain": default_chain,
    }

    if not default_chain:
        raise ValueError("No route candidates available for requested task/complexity.")

    if config.learned_routing.enabled:
        signals = _synthetic_signals(task=task, input_chars=args.input_chars)
        payload = {"max_output_tokens": args.max_output_tokens}
        feature_vector = build_routing_features(
            task=task,
            complexity=complexity,
            signals=signals,
            payload=payload,
        )

        candidates = _dedupe(
            [*config.learned_routing.task_candidates.get(task, []), *default_chain]
        )
        scored: list[dict[str, Any]] = []
        for model in candidates:
            profile = config.model_profiles.get(model)
            if profile is None:
                continue
            scored.append(
                score_model(
                    model=model,
                    profile=profile,
                    features=feature_vector,
                    payload=payload,
                    signals=signals,
                    learned_cfg=config.learned_routing,
                ).as_dict()
            )

        scored.sort(key=lambda item: float(item.get("utility") or 0.0), reverse=True)
        if not scored:
            selected_model = default_chain[0]
            fallback_chain = default_chain[1:]
            learned_summary: list[dict[str, Any]] = []
        else:
            selected_model = str(scored[0]["model"])
            fallback_chain = [
                str(item["model"])
                for item in scored[1:]
                if str(item["model"]) != selected_model
            ]
            learned_summary = scored if args.debug else scored[:3]

        result["learned_score_summary"] = learned_summary
    else:
        selected_model = default_chain[0]
        fallback_chain = default_chain[1:]
        result["learned_score_summary"] = []

    result["final_selection"] = selected_model
    result["fallback_chain"] = fallback_chain

    print(yaml.safe_dump(result, sort_keys=False).rstrip())
    return 0


def cmd_calibration_report(args: argparse.Namespace) -> int:
    config, _ = load_routing_config_with_metadata(args.path)
    cfg = config.classifier_calibration

    overrides_path = Path(args.overrides_path)
    overrides_found = overrides_path.exists()
    calibration_runtime: dict[str, Any] = {}
    if overrides_found:
        raw_overrides = _read_yaml(overrides_path)
        runtime_section = raw_overrides.get("classifier_calibration")
        if isinstance(runtime_section, dict):
            calibration_runtime = runtime_section

    config_low = float(cfg.secondary_low_confidence_min_confidence)
    config_mixed = float(cfg.secondary_mixed_signal_min_confidence)
    active_low = _coerce_float(
        calibration_runtime.get("secondary_low_confidence_min_confidence"),
        config_low,
    )
    active_mixed = _coerce_float(
        calibration_runtime.get("secondary_mixed_signal_min_confidence"),
        config_mixed,
    )
    active_mixed = max(active_mixed, active_low)

    target = float(cfg.target_secondary_success_rate)
    observed_success_rate = _coerce_optional_float(
        calibration_runtime.get("secondary_success_rate")
    )
    secondary_samples = _coerce_optional_int(calibration_runtime.get("secondary_total"))
    if secondary_samples is None:
        secondary_samples = _coerce_optional_int(
            calibration_runtime.get("secondary_samples")
        )

    drift = (
        None
        if observed_success_rate is None
        else float(observed_success_rate) - float(target)
    )

    history = calibration_runtime.get("history")
    normalized_history: list[dict[str, Any]] = []
    if isinstance(history, list):
        for entry in history:
            if isinstance(entry, dict):
                normalized_history.append(entry)
    history_limit = max(0, int(args.history_limit))
    if history_limit > 0:
        normalized_history = normalized_history[-history_limit:]
    else:
        normalized_history = []

    payload = {
        "config_path": str(args.path),
        "overrides_path": str(overrides_path),
        "overrides_found": overrides_found,
        "enabled": bool(cfg.enabled),
        "target_secondary_success_rate": target,
        "observed_secondary_success_rate": observed_success_rate,
        "success_rate_drift": drift,
        "secondary_samples": secondary_samples,
        "thresholds": {
            "config": {
                "secondary_low_confidence_min_confidence": config_low,
                "secondary_mixed_signal_min_confidence": config_mixed,
            },
            "active": {
                "secondary_low_confidence_min_confidence": active_low,
                "secondary_mixed_signal_min_confidence": active_mixed,
            },
        },
        "history": normalized_history,
    }
    print(yaml.safe_dump(payload, sort_keys=False).rstrip())
    return 0


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _synthetic_signals(task: str, input_chars: int) -> dict[str, Any]:
    return {
        "text_length": max(1, int(input_chars)),
        "code_score": 1.0 if task == "coding" else 0.0,
        "think_score": 1.0 if task == "thinking" else 0.0,
        "instruction_score": 1.0 if task == "instruction_following" else 0.0,
        "has_image": task == "image",
    }


def _extract_guardrail_entries(
    explain_meta: dict[str, Any] | None,
    task: str,
) -> list[dict[str, Any]]:
    if not explain_meta:
        return []
    entries = explain_meta.get("guardrail_pruned")
    if not isinstance(entries, list):
        return []
    output: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        context = str(entry.get("context") or "")
        if (
            task in context
            or context.startswith("fallback_models")
            or context == "default_model"
        ):
            output.append(entry)
    return output


def _without_none_values(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _normalize_provider_alias(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "gemeni":
        return "gemini"
    if normalized in {"nim", "nvidia-nim", "nvidia-ai"}:
        return "nvidia"
    if normalized in {"github-models", "githubmodels", "gh-models", "ghmodels", "gh"}:
        return "github"
    return normalized


def _parse_models_csv(models_csv: str | None) -> list[str]:
    if not models_csv:
        return []
    return [item.strip() for item in models_csv.split(",") if item.strip()]


def _qualify_models(provider: str, models: list[str]) -> list[str]:
    output: list[str] = []
    normalized_provider = provider.strip()
    for model in models:
        if "/" in model:
            output.append(model)
        else:
            output.append(f"{normalized_provider}/{model}")
    return _dedupe(output)


def _run_chatgpt_oauth_login_flow(args: argparse.Namespace) -> dict[str, Any]:
    from open_llm_router import config_cli

    return config_cli._run_chatgpt_oauth_login_flow(args)


def _load_or_init_profile_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        profile = RouterProfileConfig()
        return profile.model_dump(mode="python")

    raw = _read_yaml(path)
    if raw and not is_profile_document(raw):
        raise ValueError(
            f"Expected profile document at '{path}', but found raw routing schema."
        )
    profile = RouterProfileConfig.model_validate(raw)
    return profile.model_dump(mode="python")


def _upsert_profile_account(
    profile_payload: dict[str, Any], account_payload: dict[str, Any]
) -> None:
    accounts = profile_payload.setdefault("accounts", [])
    if not isinstance(accounts, list):
        raise ValueError("Invalid profile format: 'accounts' must be a list.")

    target_name = str(account_payload.get("name") or "").strip()
    if not target_name:
        raise ValueError("Account payload missing required 'name'.")

    clean_payload = _without_none_values(account_payload)

    for idx, entry in enumerate(accounts):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name") or "").strip() == target_name:
            merged = dict(entry)
            merged.update(clean_payload)
            merged_models = merged.get("models", [])
            if isinstance(merged_models, list):
                merged["models"] = _dedupe(
                    [str(m).strip() for m in merged_models if str(m).strip()]
                )
            accounts[idx] = _without_none_values(merged)
            return

    accounts.append(clean_payload)


def _save_profile_payload(
    path: Path, payload: dict[str, Any], *, dry_run: bool
) -> None:
    # Compile once as validation gate before writing invalid profile docs.
    compile_profile_document(payload)
    if dry_run:
        print(yaml.safe_dump(payload, sort_keys=False).rstrip())
        return
    _write_yaml(path, payload)


def cmd_config_show(args: argparse.Namespace) -> int:
    config, _ = load_routing_config_with_metadata(args.path)
    summary = {
        "default_model": config.default_model,
        "models_count": len(config.models),
        "accounts": [account.name for account in config.accounts],
        "tasks": sorted(config.task_routes.keys()),
        "learned_enabled": bool(config.learned_routing.enabled),
    }
    print(yaml.safe_dump(summary, sort_keys=False).rstrip())
    return 0


def cmd_provider_login(args: argparse.Namespace) -> int:
    profile_path = Path(args.path)
    profile_payload = _load_or_init_profile_payload(profile_path)

    provider = _normalize_provider_alias(args.provider)
    kind = args.kind

    if provider == "openai" and kind is None:
        raise ValueError("For provider 'openai', use --kind chatgpt or --kind apikey.")

    if provider in {"gemini", "nvidia", "github"}:
        kind = kind or "apikey"
        if kind != "apikey":
            raise ValueError(
                f"Provider '{provider}' currently supports only --kind apikey."
            )

    if kind == "chatgpt":
        if provider != "openai":
            raise ValueError("--kind chatgpt is only valid for provider 'openai'.")
        models = _qualify_models(
            LOGIN_CHATGPT_DEFAULT_PROVIDER,
            _parse_models_csv(args.models or LOGIN_CHATGPT_DEFAULT_MODELS),
        )
        oauth_args = argparse.Namespace(
            client_id=args.client_id,
            authorize_url=args.authorize_url,
            token_url=args.token_url,
            redirect_uri=args.redirect_uri,
            scope=args.scope,
            originator=args.originator,
            manual_code=args.manual_code,
            paste_url=False,
            timeout_seconds=args.timeout_seconds,
            no_browser=args.no_browser,
            no_local_callback=args.no_local_callback,
        )
        oauth_payload = _run_chatgpt_oauth_login_flow(oauth_args)
        account_payload = {
            "name": args.name,
            "provider": LOGIN_CHATGPT_DEFAULT_PROVIDER,
            "auth_mode": "oauth",
            "models": models,
            "enabled": True,
            **oauth_payload,
        }
        _upsert_profile_account(profile_payload, account_payload)
        if args.set_default and models:
            profile_payload.setdefault("raw_overrides", {})["default_model"] = models[0]
        _save_profile_payload(profile_path, profile_payload, dry_run=args.dry_run)
        print(f"Provider login saved: {args.name} ({provider}/{kind})")
        return 0

    if kind == "apikey":
        if args.api_key and args.api_key_env:
            raise ValueError("Use only one of --apikey or --api-key-env.")

        if provider == "openai":
            account = args.name
            models_csv = args.models or OPENAI_APIKEY_DEFAULT_MODELS
            api_key_env = args.api_key_env or OPENAI_APIKEY_DEFAULT_KEY_ENV
            mapped_provider = "openai"
        elif provider == "gemini":
            account = args.name
            models_csv = args.models or GEMINI_APIKEY_DEFAULT_MODELS
            api_key_env = args.api_key_env or GEMINI_APIKEY_DEFAULT_KEY_ENV
            mapped_provider = "gemini"
        elif provider == "nvidia":
            account = args.name
            models_csv = args.models or NVIDIA_APIKEY_DEFAULT_MODELS
            api_key_env = args.api_key_env or NVIDIA_APIKEY_DEFAULT_KEY_ENV
            mapped_provider = "nvidia"
        elif provider == "github":
            account = args.name
            models_csv = args.models or GITHUB_APIKEY_DEFAULT_MODELS
            api_key_env = args.api_key_env or GITHUB_APIKEY_DEFAULT_KEY_ENV
            mapped_provider = "github"
        else:
            raise ValueError(
                "Unsupported provider "
                f"'{provider}'. Supported: openai, gemini, nvidia, github."
            )

        models = _qualify_models(mapped_provider, _parse_models_csv(models_csv))
        api_key = (args.api_key or "").strip()
        account_payload = {
            "name": account,
            "provider": mapped_provider,
            "auth_mode": "api_key",
            "models": models,
            "enabled": True,
        }
        if api_key:
            account_payload["api_key"] = api_key
        else:
            account_payload["api_key_env"] = api_key_env
        _upsert_profile_account(profile_payload, account_payload)
        if args.set_default and models:
            profile_payload.setdefault("raw_overrides", {})["default_model"] = models[0]
        _save_profile_payload(profile_path, profile_payload, dry_run=args.dry_run)
        print(f"Provider login saved: {args.name} ({provider}/{kind})")
        return 0

    raise ValueError(f"Unsupported --kind '{kind}'. Supported kinds: chatgpt, apikey.")


def cmd_catalog_sync(args: argparse.Namespace) -> int:
    catalog_path = Path(args.catalog_path)
    catalog_document = load_catalog_models_document(catalog_path)
    remote_models = fetch_openrouter_models(
        source_url=args.source_url,
        timeout_seconds=float(args.timeout_seconds),
    )
    stats = sync_catalog_models_pricing(
        catalog_document=catalog_document,
        openrouter_models=remote_models,
    )

    if args.dry_run:
        print("Dry run: catalog not written.")
    else:
        write_catalog_models_document(catalog_path, catalog_document)
        print(f"Wrote catalog: {catalog_path}")

    print(
        yaml.safe_dump(
            {
                "total_local_models": stats.total_local_models,
                "updated": stats.updated,
                "unchanged": stats.unchanged,
                "missing_remote": stats.missing_remote,
                "missing_pricing": stats.missing_pricing,
                "source_url": args.source_url,
                "catalog_path": str(catalog_path),
                "dry_run": bool(args.dry_run),
            },
            sort_keys=False,
        ).rstrip()
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router",
        description="Profile-driven configuration and diagnostics for open-llm-router.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cmd = subparsers.add_parser("init", help="Create a profile config file.")
    init_cmd.add_argument("--profile", default="auto")
    _add_profile_path_argument(init_cmd)
    init_cmd.add_argument("--force", action="store_true")
    init_cmd.set_defaults(handler=cmd_init)

    compile_cmd = subparsers.add_parser(
        "compile-config",
        help="Compile router.profile.yaml into full routing schema.",
    )
    _add_profile_path_argument(compile_cmd)
    compile_cmd.add_argument("--output", default="router.effective.yaml")
    compile_cmd.add_argument("--stdout", action="store_true")
    compile_cmd.add_argument("--explain", action="store_true")
    compile_cmd.set_defaults(handler=cmd_compile_config)

    validate_cmd = subparsers.add_parser(
        "validate-config",
        help="Validate profile or raw routing config with strict catalog checks.",
    )
    _add_profile_path_argument(validate_cmd)
    validate_cmd.set_defaults(handler=cmd_validate_config)

    explain_cmd = subparsers.add_parser(
        "explain-route",
        help="Explain route selection for a task/complexity combination.",
    )
    _add_profile_path_argument(explain_cmd)
    explain_cmd.add_argument("--task", required=True)
    explain_cmd.add_argument(
        "--complexity",
        default="medium",
        choices=["low", "medium", "high", "xhigh"],
    )
    explain_cmd.add_argument("--input-chars", type=int, default=1200)
    explain_cmd.add_argument("--max-output-tokens", type=int, default=512)
    explain_cmd.add_argument("--debug", action="store_true")
    explain_cmd.set_defaults(handler=cmd_explain_route)

    calibration_cmd = subparsers.add_parser(
        "calibration-report",
        help="Show classifier calibration drift and recent adjustment history.",
    )
    _add_profile_path_argument(calibration_cmd)
    calibration_cmd.add_argument(
        "--overrides-path",
        default=DEFAULT_RUNTIME_OVERRIDES_PATH,
        help="Path to runtime overrides file written by policy updater.",
    )
    calibration_cmd.add_argument(
        "--history-limit",
        type=int,
        default=10,
        help="Number of recent history entries to print.",
    )
    calibration_cmd.set_defaults(handler=cmd_calibration_report)

    profile_cmd = subparsers.add_parser(
        "profile", help="Inspect built-in profile templates."
    )
    profile_subparsers = profile_cmd.add_subparsers(
        dest="profile_command", required=True
    )

    profile_list_cmd = profile_subparsers.add_parser(
        "list", help="List built-in profiles."
    )
    profile_list_cmd.set_defaults(handler=cmd_profile_list)

    profile_show_cmd = profile_subparsers.add_parser(
        "show", help="Show one profile template."
    )
    profile_show_cmd.add_argument("name")
    profile_show_cmd.set_defaults(handler=cmd_profile_show)

    show_cmd = subparsers.add_parser(
        "show", help="Show concise summary for router config."
    )
    _add_profile_path_argument(show_cmd)
    show_cmd.set_defaults(handler=cmd_config_show)

    provider_cmd = subparsers.add_parser(
        "provider",
        help="Provider-level setup/login workflow.",
    )
    provider_subparsers = provider_cmd.add_subparsers(
        dest="provider_command", required=True
    )

    provider_login_cmd = provider_subparsers.add_parser(
        "login",
        help=(
            "Login/setup a provider account. Examples: "
            "'router provider login openai --kind chatgpt', "
            "'router provider login openai --kind apikey', "
            "'router provider login gemini' (defaults to apikey), "
            "'router provider login nvidia' (defaults to apikey), "
            "'router provider login github' (defaults to apikey)."
        ),
    )
    provider_login_cmd.add_argument(
        "provider",
        help=(
            "Provider id: openai, gemini, nvidia, or github "
            "(gemeni/gemini, nim/nvidia, and github-models/github aliases supported)."
        ),
    )
    provider_login_cmd.add_argument(
        "--kind",
        choices=["chatgpt", "apikey"],
        help="Auth flow kind. For gemini/nvidia, default is apikey.",
    )
    provider_login_cmd.add_argument(
        "--type",
        dest="kind",
        choices=["chatgpt", "apikey"],
        help=argparse.SUPPRESS,
    )
    _add_profile_path_argument(provider_login_cmd)
    provider_login_cmd.add_argument("--dry-run", action="store_true")
    provider_login_cmd.add_argument(
        "--name", required=True, help="Account name (required)."
    )
    provider_login_cmd.add_argument("--models")
    provider_login_cmd.add_argument("--apikey", dest="api_key")
    provider_login_cmd.add_argument("--api-key-env", dest="api_key_env")
    provider_login_cmd.add_argument("--apikey-env", dest="api_key_env")
    provider_login_cmd.add_argument(
        "--set-default", dest="set_default", action="store_true"
    )
    provider_login_cmd.add_argument(
        "--no-set-default", dest="set_default", action="store_false"
    )
    provider_login_cmd.set_defaults(set_default=True)
    provider_login_cmd.add_argument(
        "--client-id", default="app_EMoamEEZ73f0CkXaXp7hrann"
    )
    provider_login_cmd.add_argument(
        "--authorize-url", default="https://auth.openai.com/oauth/authorize"
    )
    provider_login_cmd.add_argument(
        "--token-url", default="https://auth.openai.com/oauth/token"
    )
    provider_login_cmd.add_argument(
        "--redirect-uri", default="http://localhost:1455/auth/callback"
    )
    provider_login_cmd.add_argument(
        "--scope", default="openid profile email offline_access"
    )
    provider_login_cmd.add_argument("--originator", default="pi")
    provider_login_cmd.add_argument("--manual-code")
    provider_login_cmd.add_argument("--timeout-seconds", type=int, default=180)
    provider_login_cmd.add_argument("--no-browser", action="store_true")
    provider_login_cmd.add_argument("--no-local-callback", action="store_true")
    provider_login_cmd.set_defaults(handler=cmd_provider_login)

    catalog_cmd = subparsers.add_parser(
        "catalog",
        help="Catalog maintenance commands.",
    )
    catalog_subparsers = catalog_cmd.add_subparsers(
        dest="catalog_command", required=True
    )

    catalog_sync_cmd = catalog_subparsers.add_parser(
        "sync",
        help="Sync local model pricing from OpenRouter models API.",
    )
    catalog_sync_cmd.add_argument(
        "--source-url",
        default=OPENROUTER_MODELS_URL,
        help="OpenRouter models API URL.",
    )
    catalog_sync_cmd.add_argument(
        "--catalog-path",
        default=str(DEFAULT_CATALOG_MODELS_PATH),
        help="Path to local catalog models.yaml file.",
    )
    catalog_sync_cmd.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for the source API.",
    )
    catalog_sync_cmd.add_argument("--dry-run", action="store_true")
    catalog_sync_cmd.set_defaults(handler=cmd_catalog_sync)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)

    try:
        return handler(args)
    except Exception as exc:  # pragma: no cover - covered via CLI tests
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
