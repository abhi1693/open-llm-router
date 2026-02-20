from __future__ import annotations

import argparse
from typing import Any

import pytest

from open_llm_router.config_cli import main
from tests.yaml_test_utils import load_yaml_file, save_yaml_file


def test_cli_add_account_and_route(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"

    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-account",
                "--name",
                "openclaw-a",
                "--provider",
                "openai",
                "--base-url",
                "http://localhost:11434",
                "--api-key-env",
                "OPENCLAW_ACCOUNT_A_KEY",
                "--models",
                "gpt-5.2",
                "--set-default",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "set-route",
                "--task",
                "coding",
                "--tier",
                "xhigh",
                "--model",
                "gpt-5.2",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    assert config["default_model"] == "openai/gpt-5.2"
    assert config["accounts"][0]["name"] == "openclaw-a"
    assert config["accounts"][0]["provider"] == "openai"
    assert "openai/gpt-5.2" in config["accounts"][0]["models"]
    assert config["task_routes"]["coding"]["xhigh"] == ["gpt-5.2"]
    assert isinstance(config["models"], dict)
    assert config["models"]["openai/gpt-5.2"]["id"] == "gpt-5.2"


def test_cli_set_route_accepts_multiple_models(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"
    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-account",
                "--name",
                "acct-a",
                "--provider",
                "openai",
                "--base-url",
                "http://localhost:11434",
                "--api-key-env",
                "BACKEND_API_KEY",
                "--models",
                "m-1,m-2",
                "--set-default",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "set-route",
                "--task",
                "coding",
                "--tier",
                "medium",
                "--model",
                "m-1,m-2,m-3",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    assert config["task_routes"]["coding"]["medium"] == ["m-1", "m-2", "m-3"]


def test_cli_set_route_accepts_provider_qualified_models(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"

    assert (
        main(
            [
                "--path",
                str(config_path),
                "set-route",
                "--task",
                "coding",
                "--tier",
                "xhigh",
                "--model",
                "openai/gpt-5.2,openai-codex/gpt-5.2-codex",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    assert config["task_routes"]["coding"]["xhigh"] == [
        "openai/gpt-5.2",
        "openai-codex/gpt-5.2-codex",
    ]
    assert "openai/gpt-5.2" in config["models"]
    assert "openai-codex/gpt-5.2-codex" in config["models"]
    assert config["models"]["openai/gpt-5.2"]["id"] == "gpt-5.2"
    assert config["models"]["openai-codex/gpt-5.2-codex"]["id"] == "gpt-5.2-codex"


def test_cli_set_profile_candidates_and_learned_options(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"

    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-model",
                "--model",
                "gpt-5.2",
                "--set-default",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "set-profile",
                "--model",
                "openai-codex/gpt-5.2-codex",
                "--quality-bias",
                "0.65",
                "--quality-sensitivity",
                "2.1",
                "--cost-input-per-1k",
                "0.0012",
                "--cost-output-per-1k",
                "0.004",
                "--latency-ms",
                "1450",
                "--failure-rate",
                "0.028",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "set-candidates",
                "--task",
                "coding",
                "--models",
                "openai-codex/gpt-5.2-codex,gemini/gemini-2.5-flash,openai/gpt-5.2",
                "--enable",
            ]
        )
        == 0
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "set-learned",
                "--enabled",
                "true",
                "--utility-cost",
                "12",
                "--utility-latency",
                "0.2",
                "--utility-failure",
                "3",
                "--set-feature",
                "complexity_score=1.4",
                "--set-feature",
                "task_coding=1.0",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    profile = config["model_profiles"]["openai-codex/gpt-5.2-codex"]
    assert profile["quality_bias"] == 0.65
    assert profile["cost_output_per_1k"] == 0.004
    assert config["learned_routing"]["enabled"] is True
    assert (
        config["learned_routing"]["task_candidates"]["coding"][-1] == "openai/gpt-5.2"
    )
    assert config["learned_routing"]["feature_weights"]["complexity_score"] == 1.4


def test_cli_add_oauth_account(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"

    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-account",
                "--name",
                "openai-codex-work",
                "--provider",
                "openai-codex",
                "--base-url",
                "https://chatgpt.com/backend-api",
                "--auth-mode",
                "oauth",
                "--oauth-access-token-env",
                "CHATGPT_OAUTH_ACCESS_TOKEN",
                "--oauth-refresh-token-env",
                "CHATGPT_OAUTH_REFRESH_TOKEN",
                "--oauth-client-id-env",
                "CHATGPT_OAUTH_CLIENT_ID",
                "--models",
                "gpt-5.2-codex,gpt-5.2",
                "--set-default",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    account = config["accounts"][0]
    assert account["name"] == "openai-codex-work"
    assert account["auth_mode"] == "oauth"
    assert account["oauth_access_token_env"] == "CHATGPT_OAUTH_ACCESS_TOKEN"
    assert account["oauth_refresh_token_env"] == "CHATGPT_OAUTH_REFRESH_TOKEN"
    assert account["oauth_client_id_env"] == "CHATGPT_OAUTH_CLIENT_ID"
    assert account["base_url"] == "https://chatgpt.com/backend-api"


def test_cli_add_gemini_api_key_account(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"

    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-account",
                "--name",
                "gemini-work",
                "--provider",
                "gemini",
                "--base-url",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "--api-key-env",
                "GEMINI_API_KEY",
                "--models",
                "gemini-2.5-pro,gemini-2.5-flash",
                "--set-default",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    account = config["accounts"][0]
    assert account["name"] == "gemini-work"
    assert account["provider"] == "gemini"
    assert account["auth_mode"] == "api_key"
    assert account["api_key_env"] == "GEMINI_API_KEY"
    assert "gemini/gemini-2.5-pro" in account["models"]
    assert "gemini/gemini-2.5-flash" in account["models"]
    assert "gemini/gemini-2.5-pro" in config["models"]
    assert "gemini/gemini-2.5-flash" in config["models"]
    assert config["models"]["gemini/gemini-2.5-pro"]["id"] == "gemini-2.5-pro"
    assert config["models"]["gemini/gemini-2.5-flash"]["id"] == "gemini-2.5-flash"
    assert config["default_model"] == "gemini/gemini-2.5-pro"


def test_cli_login_chatgpt_saves_oauth_fields(tmp_path: Any, monkeypatch: Any) -> None:
    config_path = tmp_path / "router.yaml"

    def _fake_login(_args: Any) -> Any:
        return {
            "oauth_access_token": "access-token-1",
            "oauth_refresh_token": "refresh-token-1",
            "oauth_expires_at": 2222222222,
            "oauth_account_id": "acc_123",
            "oauth_client_id": "app_client_1",
            "oauth_token_url": "https://auth.openai.com/oauth/token",
        }

    monkeypatch.setattr(
        "open_llm_router.config_cli._run_chatgpt_oauth_login_flow",
        _fake_login,
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "login-chatgpt",
                "--account",
                "openai-codex-work",
                "--provider",
                "openai-codex",
                "--models",
                "gpt-5.2-codex,gpt-5.2",
                "--set-default",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    account = config["accounts"][0]
    assert account["name"] == "openai-codex-work"
    assert account["provider"] == "openai-codex"
    assert account["base_url"] == "https://chatgpt.com/backend-api"
    assert account["auth_mode"] == "oauth"
    assert account["oauth_access_token"] == "access-token-1"
    assert account["oauth_refresh_token"] == "refresh-token-1"
    assert account["oauth_expires_at"] == 2222222222
    assert account["oauth_account_id"] == "acc_123"
    assert account["oauth_client_id"] == "app_client_1"
    assert account["oauth_token_url"] == "https://auth.openai.com/oauth/token"
    assert "openai-codex/gpt-5.2-codex" in account["models"]
    assert "openai-codex/gpt-5.2" in account["models"]
    assert "openai-codex/gpt-5.2-codex" in config["models"]
    assert "openai-codex/gpt-5.2" in config["models"]
    assert config["models"]["openai-codex/gpt-5.2-codex"]["id"] == "gpt-5.2-codex"
    assert config["models"]["openai-codex/gpt-5.2"]["id"] == "gpt-5.2"
    assert config["default_model"] == "openai-codex/gpt-5.2-codex"


def test_cli_login_chatgpt_normalizes_existing_default_model(
    tmp_path: Any, monkeypatch: Any
) -> None:
    config_path = tmp_path / "router.yaml"
    save_yaml_file(
        config_path,
        {
            "default_model": "gpt-5.2-codex",
            "models": ["gpt-5.2-codex"],
            "accounts": [],
        },
    )

    def _fake_login(_args: Any) -> Any:
        return {
            "oauth_access_token": "access-token-1",
            "oauth_refresh_token": "refresh-token-1",
            "oauth_expires_at": 2222222222,
            "oauth_account_id": "acc_123",
            "oauth_client_id": "app_client_1",
            "oauth_token_url": "https://auth.openai.com/oauth/token",
        }

    monkeypatch.setattr(
        "open_llm_router.config_cli._run_chatgpt_oauth_login_flow",
        _fake_login,
    )

    assert (
        main(
            [
                "--path",
                str(config_path),
                "login-chatgpt",
                "--account",
                "openai-codex-work",
                "--provider",
                "openai-codex",
                "--models",
                "gpt-5.2-codex,gpt-5.2",
            ]
        )
        == 0
    )

    config = load_yaml_file(config_path)
    assert config["default_model"] == "openai-codex/gpt-5.2-codex"
    assert "openai-codex/gpt-5.2-codex" in config["models"]
    assert "openai-codex/gpt-5.2" in config["models"]


def test_cli_login_chatgpt_rejects_non_openai_codex_provider(tmp_path: Any) -> None:
    config_path = tmp_path / "router.yaml"
    with pytest.raises(SystemExit):
        main(
            [
                "--path",
                str(config_path),
                "login-chatgpt",
                "--account",
                "openai-work",
                "--provider",
                "openai",
                "--models",
                "gpt-5.2",
            ]
        )


def test_oauth_login_flow_uses_manual_paste_when_browser_unavailable(
    monkeypatch: Any,
) -> None:
    import open_llm_router.config_cli as config_cli

    class _DummyServer:
        def __init__(self) -> None:
            self.expected_state = "state-123"
            self.auth_code = None

        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    wait_called = {"value": False}

    def _fake_wait_for_callback(_server: Any, _timeout: Any) -> Any:
        wait_called["value"] = True
        return None

    class _DummyResponse:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "access_token": "header.payload.signature",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
            }

    monkeypatch.setattr(
        config_cli, "_generate_pkce", lambda: ("verifier-1", "challenge-1")
    )
    monkeypatch.setattr(config_cli.secrets, "token_hex", lambda _size: "state-123")
    monkeypatch.setattr(
        config_cli,
        "_start_callback_server",
        lambda **_kwargs: (_DummyServer(), None),
    )
    monkeypatch.setattr(config_cli, "_wait_for_callback_code", _fake_wait_for_callback)
    monkeypatch.setattr(config_cli.webbrowser, "open", lambda _url: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (
            "http://localhost:1455/auth/callback?code=abc123&state=state-123"
        ),
    )
    monkeypatch.setattr(
        config_cli.httpx, "post", lambda *_args, **_kwargs: _DummyResponse()
    )

    args = argparse.Namespace(
        client_id="client-1",
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        redirect_uri="http://localhost:1455/auth/callback",
        scope="openid profile email offline_access",
        originator="pi",
        manual_code=None,
        timeout_seconds=180,
        no_browser=False,
        no_local_callback=False,
    )

    result = config_cli._run_chatgpt_oauth_login_flow(args)
    assert wait_called["value"] is False
    assert result["oauth_refresh_token"] == "refresh-1"
    assert result["oauth_client_id"] == "client-1"
    assert result["oauth_token_url"] == "https://auth.openai.com/oauth/token"


def test_oauth_login_flow_uses_paste_url_arg_without_browser_or_callback(
    monkeypatch: Any,
) -> None:
    import open_llm_router.config_cli as config_cli

    class _DummyServer:
        def __init__(self) -> None:
            self.expected_state = "state-123"
            self.auth_code = None

        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    browser_called = {"value": False}
    wait_called = {"value": False}

    def _fake_browser_open(_url: Any) -> Any:
        browser_called["value"] = True
        return True

    def _fake_wait_for_callback(_server: Any, _timeout: Any) -> Any:
        wait_called["value"] = True
        return None

    class _DummyResponse:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "access_token": "header.payload.signature",
                "refresh_token": "refresh-2",
                "expires_in": 3600,
            }

    monkeypatch.setattr(
        config_cli, "_generate_pkce", lambda: ("verifier-1", "challenge-1")
    )
    monkeypatch.setattr(config_cli.secrets, "token_hex", lambda _size: "state-123")
    monkeypatch.setattr(
        config_cli,
        "_start_callback_server",
        lambda **_kwargs: (_DummyServer(), None),
    )
    monkeypatch.setattr(config_cli.webbrowser, "open", _fake_browser_open)
    monkeypatch.setattr(config_cli, "_wait_for_callback_code", _fake_wait_for_callback)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (
            "http://localhost:1455/auth/callback?code=abc999&state=state-123"
        ),
    )
    monkeypatch.setattr(
        config_cli.httpx, "post", lambda *_args, **_kwargs: _DummyResponse()
    )

    args = argparse.Namespace(
        client_id="client-1",
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        redirect_uri="http://localhost:1455/auth/callback",
        scope="openid profile email offline_access",
        originator="pi",
        manual_code=None,
        paste_url=True,
        timeout_seconds=180,
        no_browser=False,
        no_local_callback=False,
    )

    result = config_cli._run_chatgpt_oauth_login_flow(args)
    assert browser_called["value"] is False
    assert wait_called["value"] is False
    assert result["oauth_refresh_token"] == "refresh-2"
    assert result["oauth_client_id"] == "client-1"
    assert result["oauth_token_url"] == "https://auth.openai.com/oauth/token"


def test_cli_show_outputs_summary(tmp_path: Any, capsys: Any) -> None:
    config_path = tmp_path / "router.yaml"
    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-model",
                "--model",
                "gpt-5.2",
                "--set-default",
            ]
        )
        == 0
    )
    assert main(["--path", str(config_path), "show"]) == 0
    output = capsys.readouterr().out
    assert "default_model:" in output
    assert "learned_enabled:" in output
