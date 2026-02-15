from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smart_model_router.config_cli import main


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_cli_add_account_and_route(tmp_path):
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
                "qwen2.5-14b-instruct,qwen2.5-coder-14b-instruct",
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
                "codex-1",
            ]
        )
        == 0
    )

    config = _load(config_path)
    assert config["default_model"] == "qwen2.5-14b-instruct"
    assert config["accounts"][0]["name"] == "openclaw-a"
    assert config["accounts"][0]["provider"] == "openai"
    assert "qwen2.5-coder-14b-instruct" in config["accounts"][0]["models"]
    assert config["task_routes"]["coding"]["xhigh"] == "codex-1"
    assert "codex-1" in config["models"]


def test_cli_set_profile_candidates_and_learned_options(tmp_path):
    config_path = tmp_path / "router.yaml"

    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-model",
                "--model",
                "qwen2.5-14b-instruct",
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
                "codex-1",
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
                "qwen2.5-coder-14b-instruct,deepseek-r1,codex-1",
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

    config = _load(config_path)
    profile = config["model_profiles"]["codex-1"]
    assert profile["quality_bias"] == 0.65
    assert profile["cost_output_per_1k"] == 0.004
    assert config["learned_routing"]["enabled"] is True
    assert config["learned_routing"]["task_candidates"]["coding"][-1] == "codex-1"
    assert config["learned_routing"]["feature_weights"]["complexity_score"] == 1.4


def test_cli_add_oauth_account(tmp_path):
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

    config = _load(config_path)
    account = config["accounts"][0]
    assert account["name"] == "openai-codex-work"
    assert account["auth_mode"] == "oauth"
    assert account["oauth_access_token_env"] == "CHATGPT_OAUTH_ACCESS_TOKEN"
    assert account["oauth_refresh_token_env"] == "CHATGPT_OAUTH_REFRESH_TOKEN"
    assert account["oauth_client_id_env"] == "CHATGPT_OAUTH_CLIENT_ID"
    assert account["base_url"] == "https://chatgpt.com/backend-api"


def test_cli_login_chatgpt_saves_oauth_fields(tmp_path, monkeypatch):
    config_path = tmp_path / "router.yaml"

    def _fake_login(_args):
        return {
            "oauth_access_token": "access-token-1",
            "oauth_refresh_token": "refresh-token-1",
            "oauth_expires_at": 2222222222,
            "oauth_account_id": "acc_123",
            "oauth_client_id": "app_client_1",
            "oauth_token_url": "https://auth.openai.com/oauth/token",
        }

    monkeypatch.setattr(
        "smart_model_router.config_cli._run_chatgpt_oauth_login_flow",
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
                "--models",
                "gpt-5.2-codex,gpt-5.2",
                "--set-default",
            ]
        )
        == 0
    )

    config = _load(config_path)
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
    assert "gpt-5.2-codex" in account["models"]
    assert config["default_model"] == "gpt-5.2-codex"


def test_oauth_login_flow_uses_manual_paste_when_browser_unavailable(monkeypatch):
    import smart_model_router.config_cli as config_cli

    class _DummyServer:
        def __init__(self) -> None:
            self.expected_state = "state-123"
            self.auth_code = None

        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    wait_called = {"value": False}

    def _fake_wait_for_callback(_server, _timeout):
        wait_called["value"] = True
        return None

    class _DummyResponse:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict:
            return {
                "access_token": "header.payload.signature",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
            }

    monkeypatch.setattr(config_cli, "_generate_pkce", lambda: ("verifier-1", "challenge-1"))
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
        lambda _prompt: "http://localhost:1455/auth/callback?code=abc123&state=state-123",
    )
    monkeypatch.setattr(config_cli.httpx, "post", lambda *_args, **_kwargs: _DummyResponse())

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


def test_oauth_login_flow_uses_paste_url_arg_without_browser_or_callback(monkeypatch):
    import smart_model_router.config_cli as config_cli

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

    def _fake_browser_open(_url):
        browser_called["value"] = True
        return True

    def _fake_wait_for_callback(_server, _timeout):
        wait_called["value"] = True
        return None

    class _DummyResponse:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict:
            return {
                "access_token": "header.payload.signature",
                "refresh_token": "refresh-2",
                "expires_in": 3600,
            }

    monkeypatch.setattr(config_cli, "_generate_pkce", lambda: ("verifier-1", "challenge-1"))
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
        lambda _prompt: "http://localhost:1455/auth/callback?code=abc999&state=state-123",
    )
    monkeypatch.setattr(config_cli.httpx, "post", lambda *_args, **_kwargs: _DummyResponse())

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


def test_cli_show_outputs_summary(tmp_path, capsys):
    config_path = tmp_path / "router.yaml"
    assert (
        main(
            [
                "--path",
                str(config_path),
                "add-model",
                "--model",
                "qwen2.5-14b-instruct",
                "--set-default",
            ]
        )
        == 0
    )
    assert main(["--path", str(config_path), "show"]) == 0
    output = capsys.readouterr().out
    assert "default_model:" in output
    assert "learned_enabled:" in output
