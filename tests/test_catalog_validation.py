from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_llm_router.catalog import CatalogValidationError
from open_llm_router.config import load_routing_config


def _write(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_load_routing_config_rejects_unknown_model(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "openai-codex/not-a-real-model",
            "task_routes": {"general": {"default": ["openai-codex/not-a-real-model"]}},
        },
    )

    with pytest.raises(CatalogValidationError) as exc:
        load_routing_config(str(config_path))

    message = str(exc.value)
    assert "default_model" in message
    assert "Suggested canonical ids" in message


def test_load_routing_config_rejects_unknown_provider(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "openai-codex/gpt-5.2",
            "task_routes": {"general": {"default": ["openai-codex/gpt-5.2"]}},
            "accounts": [
                {
                    "name": "bad-account",
                    "provider": "not-supported",
                    "base_url": "https://example.com",
                }
            ],
        },
    )

    with pytest.raises(CatalogValidationError) as exc:
        load_routing_config(str(config_path))

    message = str(exc.value)
    assert "accounts[0].provider" in message
    assert "Suggested canonical ids" in message


def test_load_routing_config_accepts_provider_hint_for_models_with_slash(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/z-ai/glm5",
            "task_routes": {"general": {"default": ["nvidia/z-ai/glm5"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["z-ai/glm5"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/z-ai/glm5"
    assert loaded.accounts[0].models == ["z-ai/glm5"]


def test_load_routing_config_accepts_nvidia_model_id_with_nested_slash(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/moonshotai/kimi-k2.5",
            "task_routes": {"general": {"default": ["nvidia/moonshotai/kimi-k2.5"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["moonshotai/kimi-k2.5"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/moonshotai/kimi-k2.5"
    assert loaded.accounts[0].models == ["moonshotai/kimi-k2.5"]


def test_load_routing_config_accepts_nvidia_deepseek_model_id_with_nested_slash(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/deepseek-ai/deepseek-v3.2",
            "task_routes": {"general": {"default": ["nvidia/deepseek-ai/deepseek-v3.2"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["deepseek-ai/deepseek-v3.2"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/deepseek-ai/deepseek-v3.2"
    assert loaded.accounts[0].models == ["deepseek-ai/deepseek-v3.2"]


def test_load_routing_config_accepts_nvidia_deepseek_terminus_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/deepseek-ai/deepseek-v3.1-terminus",
            "task_routes": {
                "general": {"default": ["nvidia/deepseek-ai/deepseek-v3.1-terminus"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["deepseek-ai/deepseek-v3.1-terminus"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/deepseek-ai/deepseek-v3.1-terminus"
    assert loaded.accounts[0].models == ["deepseek-ai/deepseek-v3.1-terminus"]


def test_load_routing_config_accepts_nvidia_deepseek_v31_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/deepseek-ai/deepseek-v3.1",
            "task_routes": {
                "general": {"default": ["nvidia/deepseek-ai/deepseek-v3.1"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["deepseek-ai/deepseek-v3.1"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/deepseek-ai/deepseek-v3.1"
    assert loaded.accounts[0].models == ["deepseek-ai/deepseek-v3.1"]


def test_load_routing_config_accepts_nvidia_glm47_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/z-ai/glm4.7",
            "task_routes": {"general": {"default": ["nvidia/z-ai/glm4.7"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["z-ai/glm4.7"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/z-ai/glm4.7"
    assert loaded.accounts[0].models == ["z-ai/glm4.7"]


def test_load_routing_config_accepts_nvidia_minimax_m21_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/minimaxai/minimax-m2.1",
            "task_routes": {"general": {"default": ["nvidia/minimaxai/minimax-m2.1"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["minimaxai/minimax-m2.1"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/minimaxai/minimax-m2.1"
    assert loaded.accounts[0].models == ["minimaxai/minimax-m2.1"]


def test_load_routing_config_accepts_nvidia_minimax_m2_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/minimaxai/minimax-m2",
            "task_routes": {"general": {"default": ["nvidia/minimaxai/minimax-m2"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["minimaxai/minimax-m2"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/minimaxai/minimax-m2"
    assert loaded.accounts[0].models == ["minimaxai/minimax-m2"]


def test_load_routing_config_accepts_nvidia_kimi_k2_thinking_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/moonshotai/kimi-k2-thinking",
            "task_routes": {
                "general": {"default": ["nvidia/moonshotai/kimi-k2-thinking"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["moonshotai/kimi-k2-thinking"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/moonshotai/kimi-k2-thinking"
    assert loaded.accounts[0].models == ["moonshotai/kimi-k2-thinking"]


def test_load_routing_config_accepts_nvidia_kimi_k2_instruct_0905_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/moonshotai/kimi-k2-instruct-0905",
            "task_routes": {
                "general": {"default": ["nvidia/moonshotai/kimi-k2-instruct-0905"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["moonshotai/kimi-k2-instruct-0905"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/moonshotai/kimi-k2-instruct-0905"
    assert loaded.accounts[0].models == ["moonshotai/kimi-k2-instruct-0905"]


def test_load_routing_config_accepts_nvidia_kimi_k2_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/moonshotai/kimi-k2-instruct",
            "task_routes": {
                "general": {"default": ["nvidia/moonshotai/kimi-k2-instruct"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["moonshotai/kimi-k2-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/moonshotai/kimi-k2-instruct"
    assert loaded.accounts[0].models == ["moonshotai/kimi-k2-instruct"]


def test_load_routing_config_accepts_nvidia_qwen_35_397b_a17b_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen3.5-397b-a17b",
            "task_routes": {
                "general": {"default": ["nvidia/qwen/qwen3.5-397b-a17b"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen3.5-397b-a17b"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen3.5-397b-a17b"
    assert loaded.accounts[0].models == ["qwen/qwen3.5-397b-a17b"]


def test_load_routing_config_accepts_nvidia_qwen3_next_80b_a3b_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen3-next-80b-a3b-instruct",
            "task_routes": {
                "general": {"default": ["nvidia/qwen/qwen3-next-80b-a3b-instruct"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen3-next-80b-a3b-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen3-next-80b-a3b-instruct"
    assert loaded.accounts[0].models == ["qwen/qwen3-next-80b-a3b-instruct"]


def test_load_routing_config_accepts_nvidia_qwen3_next_80b_a3b_thinking_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen3-next-80b-a3b-thinking",
            "task_routes": {
                "general": {"default": ["nvidia/qwen/qwen3-next-80b-a3b-thinking"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen3-next-80b-a3b-thinking"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen3-next-80b-a3b-thinking"
    assert loaded.accounts[0].models == ["qwen/qwen3-next-80b-a3b-thinking"]


def test_load_routing_config_accepts_nvidia_qwen3_coder_480b_a35b_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen3-coder-480b-a35b-instruct",
            "task_routes": {
                "general": {"default": ["nvidia/qwen/qwen3-coder-480b-a35b-instruct"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen3-coder-480b-a35b-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen3-coder-480b-a35b-instruct"
    assert loaded.accounts[0].models == ["qwen/qwen3-coder-480b-a35b-instruct"]


def test_load_routing_config_accepts_nvidia_qwen3_235b_a22b_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen3-235b-a22b",
            "task_routes": {"general": {"default": ["nvidia/qwen/qwen3-235b-a22b"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen3-235b-a22b"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen3-235b-a22b"
    assert loaded.accounts[0].models == ["qwen/qwen3-235b-a22b"]


def test_load_routing_config_accepts_nvidia_qwen25_coder_32b_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen2.5-coder-32b-instruct",
            "task_routes": {
                "general": {"default": ["nvidia/qwen/qwen2.5-coder-32b-instruct"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen2.5-coder-32b-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen2.5-coder-32b-instruct"
    assert loaded.accounts[0].models == ["qwen/qwen2.5-coder-32b-instruct"]


def test_load_routing_config_accepts_nvidia_qwq_32b_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwq-32b",
            "task_routes": {"general": {"default": ["nvidia/qwen/qwq-32b"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwq-32b"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwq-32b"
    assert loaded.accounts[0].models == ["qwen/qwq-32b"]


def test_load_routing_config_accepts_nvidia_qwen2_7b_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen2-7b-instruct",
            "task_routes": {"general": {"default": ["nvidia/qwen/qwen2-7b-instruct"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen2-7b-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen2-7b-instruct"
    assert loaded.accounts[0].models == ["qwen/qwen2-7b-instruct"]


def test_load_routing_config_accepts_nvidia_qwen25_7b_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen2.5-7b-instruct",
            "task_routes": {"general": {"default": ["nvidia/qwen/qwen2.5-7b-instruct"]}},
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen2.5-7b-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen2.5-7b-instruct"
    assert loaded.accounts[0].models == ["qwen/qwen2.5-7b-instruct"]


def test_load_routing_config_accepts_nvidia_qwen25_coder_7b_instruct_model_id(tmp_path):
    config_path = tmp_path / "router.yaml"
    _write(
        config_path,
        {
            "default_model": "nvidia/qwen/qwen2.5-coder-7b-instruct",
            "task_routes": {
                "general": {"default": ["nvidia/qwen/qwen2.5-coder-7b-instruct"]}
            },
            "accounts": [
                {
                    "name": "nvidia-main",
                    "provider": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com",
                    "models": ["qwen/qwen2.5-coder-7b-instruct"],
                }
            ],
        },
    )

    loaded = load_routing_config(str(config_path))
    assert loaded.default_model == "nvidia/qwen/qwen2.5-coder-7b-instruct"
    assert loaded.accounts[0].models == ["qwen/qwen2.5-coder-7b-instruct"]
