from __future__ import annotations

from open_llm_router.config import RoutingConfig
from open_llm_router.server.main import _build_models_response


def _config_with_models() -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "default_model": "openai/m1",
            "task_routes": {"general": {"low": ["openai/m1"]}},
            "fallback_models": [],
            "models": {
                "openai/m1": {
                    "provider": "openai",
                    "id": "m1",
                    "costs": {"input_per_1k": 0.002, "output_per_1k": 0.006},
                    "limits": {"context_tokens": 128000, "max_output_tokens": 4096},
                    "capabilities": [
                        "chat",
                        "tool_use",
                        "reasoning",
                        "json_mode",
                        "vision",
                    ],
                    "description": "Test model",
                    "hugging_face_id": " N/A ",
                    "expiration_date": " null ",
                    "per_request_limits": {},
                },
                "plain-model": {},
            },
            "accounts": [
                {
                    "name": "acct-openai",
                    "provider": "openai",
                    "base_url": "http://provider-openai",
                    "api_key": "test-key",
                    "models": ["openai/m1", "plain-model"],
                }
            ],
        }
    )


def test_build_models_response_includes_auto_and_configured_models() -> None:
    payload = _build_models_response(_config_with_models())
    assert payload["object"] == "list"

    ids = [item["id"] for item in payload["data"]]
    assert "auto" in ids
    assert "openai/m1" in ids
    assert "plain-model" in ids


def test_build_models_response_emits_openrouter_like_model_metadata() -> None:
    payload = _build_models_response(_config_with_models())
    model = next(item for item in payload["data"] if item["id"] == "openai/m1")

    assert model["canonical_slug"] == "openai/m1"
    assert model["name"] == "openai/m1"
    assert model["description"] == "Test model"
    assert model["context_length"] == 128000
    assert model["pricing"]["prompt"] == "0.000002"
    assert model["pricing"]["completion"] == "0.000006"
    assert model["top_provider"]["max_completion_tokens"] == 4096
    assert model["top_provider"]["context_length"] == 128000
    assert "tools" in model["supported_parameters"]
    assert "tool_choice" in model["supported_parameters"]
    assert "reasoning" in model["supported_parameters"]
    assert "response_format" in model["supported_parameters"]
    assert "structured_outputs" in model["supported_parameters"]
    assert model["architecture"]["input_modalities"] == ["text", "image"]
    assert model["architecture"]["output_modalities"] == ["text"]
    assert "hugging_face_id" not in model
    assert "expiration_date" not in model
    assert "per_request_limits" not in model


def test_build_models_response_omits_fields_without_data() -> None:
    payload = _build_models_response(_config_with_models())

    auto = next(item for item in payload["data"] if item["id"] == "auto")
    assert "supported_parameters" not in auto
    assert "default_parameters" not in auto
    assert "per_request_limits" not in auto
    assert "hugging_face_id" not in auto
    assert "expiration_date" not in auto
    assert "max_completion_tokens" not in auto["top_provider"]
