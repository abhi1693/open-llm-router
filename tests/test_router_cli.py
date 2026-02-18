from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import open_llm_router.router_cli as router_cli
from open_llm_router.router_cli import main


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _save(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_router_cli_init_compile_validate(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    effective_path = tmp_path / "router.effective.yaml"

    assert main(["init", "--profile", "auto", "--path", str(profile_path)]) == 0

    profile_payload = _load(profile_path)
    profile_payload["accounts"] = [
        {
            "name": "codex-main",
            "provider": "openai-codex",
            "auth_mode": "passthrough",
        }
    ]
    _save(profile_path, profile_payload)

    assert (
        main(
            [
                "compile-config",
                "--path",
                str(profile_path),
                "--output",
                str(effective_path),
            ]
        )
        == 0
    )
    assert main(["validate-config", "--path", str(profile_path)]) == 0

    effective_payload = _load(effective_path)
    assert effective_payload["accounts"][0]["provider"] == "openai-codex"
    assert (
        effective_payload["accounts"][0]["base_url"]
        == "https://chatgpt.com/backend-api"
    )


def test_router_cli_profile_commands_and_explain(tmp_path: Any, capsys: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert main(["init", "--profile", "balanced", "--path", str(profile_path)]) == 0

    assert main(["profile", "list"]) == 0
    list_out = capsys.readouterr().out
    assert "auto" in list_out
    assert "balanced" in list_out

    assert main(["profile", "show", "auto"]) == 0
    show_out = capsys.readouterr().out
    assert "default_model" in show_out

    assert (
        main(
            [
                "explain-route",
                "--path",
                str(profile_path),
                "--task",
                "coding",
                "--complexity",
                "high",
                "--debug",
            ]
        )
        == 0
    )
    explain_out = capsys.readouterr().out
    assert "final_selection" in explain_out
    assert "fallback_chain" in explain_out


def test_router_cli_show_reads_profile_config(tmp_path: Any, capsys: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert main(["init", "--profile", "auto", "--path", str(profile_path)]) == 0
    assert main(["show", "--path", str(profile_path)]) == 0
    output = capsys.readouterr().out
    assert "default_model" in output
    assert "learned_enabled" in output


def test_router_provider_login_openai_chatgpt_writes_profile(
    monkeypatch: Any, tmp_path: Any
) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    fake_oauth = {
        "oauth_access_token": "access-1",
        "oauth_refresh_token": "refresh-1",
        "oauth_expires_at": 2000000000,
        "oauth_account_id": "acct-1",
        "oauth_client_id": "client-1",
        "oauth_token_url": "https://auth.openai.com/oauth/token",
    }
    monkeypatch.setattr(
        router_cli, "_run_chatgpt_oauth_login_flow", lambda _args: fake_oauth
    )

    assert (
        main(
            [
                "provider",
                "login",
                "openai",
                "--kind",
                "chatgpt",
                "--name",
                "openai-codex-work",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )

    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["name"] == "openai-codex-work"
    assert account["provider"] == "openai-codex"
    assert account["auth_mode"] == "oauth"
    assert account["oauth_access_token"] == "access-1"
    assert payload["raw_overrides"]["default_model"] == "openai-codex/gpt-5.2"


def test_router_provider_login_openai_apikey_writes_profile(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "openai",
                "--kind",
                "apikey",
                "--name",
                "openai-work",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["provider"] == "openai"
    assert account["auth_mode"] == "api_key"
    assert account["api_key_env"] == "OPENAI_API_KEY"


def test_router_provider_login_gemini_defaults_to_apikey(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "gemini",
                "--name",
                "gemini-work",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["provider"] == "gemini"
    assert account["auth_mode"] == "api_key"
    assert account["api_key_env"] == "GEMINI_API_KEY"


def test_router_provider_login_nvidia_defaults_to_apikey(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "nvidia",
                "--name",
                "nvidia-work",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["provider"] == "nvidia"
    assert account["auth_mode"] == "api_key"
    assert account["api_key_env"] == "NVIDIA_API_KEY"
    assert "nvidia/z-ai/glm5" in account["models"]
    assert "nvidia/moonshotai/kimi-k2.5" in account["models"]


def test_router_provider_login_apikey_flag_sets_inline_key(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "openai",
                "--kind",
                "apikey",
                "--name",
                "openai-inline-key",
                "--apikey",
                "sk-inline-secret",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["auth_mode"] == "api_key"
    assert account["api_key"] == "sk-inline-secret"
    assert "api_key_env" not in account


def test_router_provider_login_accepts_apikey_env_alias(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "gemini",
                "--name",
                "gemini-env-alias",
                "--apikey-env",
                "MY_GEMINI_KEY",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["api_key_env"] == "MY_GEMINI_KEY"


def test_router_provider_login_accepts_gemeni_alias(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "gemeni",
                "--name",
                "gemini-work-2",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["provider"] == "gemini"


def test_router_provider_login_accepts_nim_alias(tmp_path: Any) -> None:
    profile_path = tmp_path / "router.profile.yaml"
    assert (
        main(
            [
                "provider",
                "login",
                "nim",
                "--name",
                "nvidia-work-2",
                "--path",
                str(profile_path),
            ]
        )
        == 0
    )
    payload = _load(profile_path)
    account = payload["accounts"][0]
    assert account["provider"] == "nvidia"


def test_provider_login_rejects_raw_schema_path(tmp_path: Any) -> None:
    raw_path = tmp_path / "router.yaml"
    _save(
        raw_path,
        {
            "default_model": "openai/gpt-5.2",
            "task_routes": {"general": {"default": ["openai/gpt-5.2"]}},
        },
    )
    try:
        main(
            [
                "provider",
                "login",
                "gemini",
                "--name",
                "gemini-work",
                "--path",
                str(raw_path),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError(
            "Expected failure when provider login is pointed at raw schema file"
        )


def test_removed_old_top_level_login_commands() -> None:
    try:
        main(["login-chatgpt"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError(
            "Expected parser failure for removed command login-chatgpt"
        )

    try:
        main(["add-account", "--models", "openai/gpt-5.2"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("Expected parser failure for removed command add-account")


def test_provider_login_requires_name() -> None:
    try:
        main(["provider", "login", "gemini"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("Expected parser failure when --name is omitted")


def test_provider_login_rejects_apikey_and_api_key_env_together() -> None:
    try:
        main(
            [
                "provider",
                "login",
                "gemini",
                "--name",
                "gemini-bad",
                "--apikey",
                "secret-1",
                "--api-key-env",
                "GEMINI_API_KEY",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError(
            "Expected failure when both --apikey and --api-key-env are set"
        )


def test_catalog_sync_dry_run_does_not_write_catalog(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    catalog_path = tmp_path / "models.yaml"
    initial_payload = {
        "version": 1,
        "models": [
            {
                "id": "gpt-5.2",
                "provider": "openai-codex",
                "aliases": [],
                "costs": {"input_per_1k": 1.23, "output_per_1k": 4.56},
            }
        ],
    }
    _save(catalog_path, initial_payload)

    monkeypatch.setattr(
        router_cli,
        "fetch_openrouter_models",
        lambda **_kwargs: [
            {
                "id": "openai/gpt-5.2",
                "pricing": {"prompt": "0.0000019", "completion": "0.0000067"},
            }
        ],
    )

    assert (
        main(
            [
                "catalog",
                "sync",
                "--catalog-path",
                str(catalog_path),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Dry run: catalog not written." in output
    assert "updated: 1" in output

    # file should remain unchanged in dry-run mode
    payload_after = _load(catalog_path)
    assert payload_after == initial_payload


def test_catalog_sync_writes_catalog(monkeypatch: Any, tmp_path: Any) -> None:
    catalog_path = tmp_path / "models.yaml"
    _save(
        catalog_path,
        {
            "version": 1,
            "models": [
                {
                    "id": "gpt-5.2",
                    "provider": "openai-codex",
                    "aliases": [],
                    "costs": {"input_per_1k": 0.0, "output_per_1k": 0.0},
                }
            ],
        },
    )

    monkeypatch.setattr(
        router_cli,
        "fetch_openrouter_models",
        lambda **_kwargs: [
            {
                "id": "openai/gpt-5.2",
                "pricing": {"prompt": "0.0000019", "completion": "0.0000067"},
            }
        ],
    )

    assert main(["catalog", "sync", "--catalog-path", str(catalog_path)]) == 0

    payload_after = _load(catalog_path)
    model = payload_after["models"][0]
    assert model["costs"]["input_per_1k"] == 0.0019
    assert model["costs"]["output_per_1k"] == 0.0067
