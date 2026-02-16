from __future__ import annotations

import pytest

from open_llm_router.catalog import CatalogValidationError, load_internal_catalog
from open_llm_router.profile_compiler import compile_profile_document


def test_compile_profile_document_produces_effective_routing_config():
    raw_profile = {
        "schema_version": 1,
        "profile": {
            "default": "auto",
            "per_task": {
                "coding": "quality",
            },
        },
        "guardrails": {
            "max_latency_ms": 900,
            "per_task": {
                "coding": {
                    "max_failure_rate": 0.04,
                }
            },
        },
        "accounts": [
            {
                "name": "primary-codex",
                "provider": "openai-codex",
                "auth_mode": "oauth",
                "oauth_access_token_env": "CHATGPT_OAUTH_ACCESS_TOKEN",
                "models": ["gpt-5.2", "gpt-5.2-codex"],
            }
        ],
    }

    result = compile_profile_document(raw_profile)
    effective = result.effective_config

    assert effective["default_model"]
    assert effective["accounts"][0]["provider"] == "openai-codex"
    assert (
        effective["accounts"][0]["base_url"] == "https://chatgpt.com/backend-api"
    )
    assert "retry_statuses" in effective
    assert "complexity" in effective
    assert "learned_routing" in effective
    assert result.explain["profile_layers"][0] == "profile.default:auto"
    assert any(entry.get("context") for entry in result.explain["guardrail_pruned"])


def test_compile_profile_document_rejects_unknown_provider():
    raw_profile = {
        "profile": {"default": "auto"},
        "accounts": [
            {
                "name": "bad",
                "provider": "unknown-provider",
            }
        ],
    }

    with pytest.raises(CatalogValidationError) as exc:
        compile_profile_document(raw_profile)

    message = str(exc.value)
    assert "accounts[0].provider" in message
    assert "Suggested canonical ids" in message


def test_compile_profile_document_supports_nvidia_provider_models():
    raw_profile = {
        "schema_version": 1,
        "profile": {"default": "auto"},
        "accounts": [
            {
                "name": "nvidia-main",
                "provider": "nvidia",
                "auth_mode": "api_key",
                "api_key_env": "NVIDIA_API_KEY",
                "models": ["z-ai/glm5"],
            }
        ],
        "raw_overrides": {
            "default_model": "nvidia/z-ai/glm5",
            "task_routes": {"general": {"default": ["nvidia/z-ai/glm5"]}},
        },
    }

    result = compile_profile_document(raw_profile)
    effective = result.effective_config

    assert effective["accounts"][0]["provider"] == "nvidia"
    assert effective["accounts"][0]["base_url"] == "https://integrate.api.nvidia.com"
    assert effective["accounts"][0]["models"] == ["nvidia/z-ai/glm5"]


def test_compile_profile_document_supports_nvidia_nested_slash_model_id():
    raw_profile = {
        "schema_version": 1,
        "profile": {"default": "auto"},
        "accounts": [
            {
                "name": "nvidia-main",
                "provider": "nvidia",
                "auth_mode": "api_key",
                "api_key_env": "NVIDIA_API_KEY",
                "models": ["moonshotai/kimi-k2.5"],
            }
        ],
        "raw_overrides": {
            "default_model": "nvidia/moonshotai/kimi-k2.5",
            "task_routes": {"general": {"default": ["nvidia/moonshotai/kimi-k2.5"]}},
        },
    }

    result = compile_profile_document(raw_profile)
    effective = result.effective_config
    assert effective["accounts"][0]["models"] == ["nvidia/moonshotai/kimi-k2.5"]


def test_catalog_metadata_presets_expand_into_model_entries():
    catalog = load_internal_catalog()
    entry = catalog.get_model("openai-codex/gpt-5.2-codex")

    assert "tool_use" in entry.capabilities
    assert entry.limits.context_tokens == 200000
    assert entry.tier == "balanced"
    assert entry.type == "coding"
    assert entry.task_affinity.get("coding") == 1.0


def test_compile_profile_document_auto_aligns_routes_to_enabled_accounts():
    raw_profile = {
        "schema_version": 1,
        "profile": {"default": "auto"},
        "accounts": [
            {
                "name": "codex-main",
                "provider": "openai-codex",
                "auth_mode": "oauth",
                "oauth_access_token_env": "CHATGPT_OAUTH_ACCESS_TOKEN",
                "models": ["gpt-5.2-codex"],
            },
            {
                "name": "gemini-main",
                "provider": "gemini",
                "auth_mode": "api_key",
                "api_key_env": "GEMINI_API_KEY",
                "models": ["gemini-2.5-flash"],
            },
        ],
    }

    result = compile_profile_document(raw_profile)
    effective = result.effective_config
    available = {"openai-codex/gpt-5.2-codex", "gemini/gemini-2.5-flash"}

    assert set(effective["models"].keys()) == available
    assert effective["default_model"] in available
    assert effective["task_routes"]["coding"]["high"] == ["openai-codex/gpt-5.2-codex"]
    assert effective["task_routes"]["image"]["default"] == ["gemini/gemini-2.5-flash"]
    assert all(model in available for model in effective["fallback_models"])
    assert "account_alignment" in result.explain


def test_compile_profile_document_defaults_account_models_to_provider_catalog():
    catalog = load_internal_catalog()
    expected_nvidia = sorted(
        model_id for model_id in catalog.model_ids if model_id.startswith("nvidia/")
    )

    raw_profile = {
        "schema_version": 1,
        "profile": {"default": "auto"},
        "accounts": [
            {
                "name": "nvidia-main",
                "provider": "nvidia",
                "auth_mode": "api_key",
                "api_key_env": "NVIDIA_API_KEY",
                # Intentionally omit models to use provider defaults.
            }
        ],
    }

    result = compile_profile_document(raw_profile)
    effective = result.effective_config
    account_models = sorted(effective["accounts"][0]["models"])

    assert account_models == expected_nvidia
