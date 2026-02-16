from open_llm_router.config import RoutingConfig
import pytest

from open_llm_router.router_engine import (
    InvalidModelError,
    RoutingConstraintError,
    SmartModelRouter,
)


def _router() -> SmartModelRouter:
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "complexity": {"low_max_chars": 100, "medium_max_chars": 500, "high_max_chars": 2000},
            "task_routes": {
                "general": {
                    "low": "general-7b",
                    "medium": "general-14b",
                    "high": "general-32b",
                    "xhigh": "general-70b",
                },
                "coding": {
                    "low": "code-7b",
                    "medium": "code-14b",
                    "high": "code-32b",
                    "xhigh": "codex-1",
                },
                "thinking": {
                    "low": "think-14b",
                    "medium": "think-32b",
                    "high": "think-70b",
                    "xhigh": "think-codex",
                },
                "image": {"default": "vision-11b"},
                "instruction_following": {"default": "general-7b"},
            },
            "fallback_models": ["general-14b", "general-32b"],
        }
    )
    return SmartModelRouter(config)


def test_respects_explicit_model():
    router = _router()
    payload = {
        "model": "code-14b",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "code-14b"
    assert decision.source == "request"
    assert decision.task == "explicit"


def test_rejects_unknown_explicit_model():
    router = _router()
    payload = {
        "model": "my-custom-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(InvalidModelError) as exc:
        router.decide(payload, "/v1/chat/completions")
    assert exc.value.requested_model == "my-custom-model"
    assert "not configured" in str(exc.value)


def test_respects_explicit_provider_qualified_model():
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "models": ["openai/codex-1", "general-14b"],
            "complexity": {"low_max_chars": 100, "medium_max_chars": 500, "high_max_chars": 2000},
            "task_routes": {},
            "fallback_models": ["general-14b", "general-32b"],
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "openai/codex-1",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "openai/codex-1"
    assert decision.source == "request"


def test_treats_openrouter_auto_alias_as_auto_routing():
    router = _router()
    payload = {
        "model": "openrouter/auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.source == "auto"
    assert decision.task == "general"
    assert decision.selected_model == "general-7b"


def test_allowed_models_filters_auto_route_candidates():
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/codex-1",
            "task_routes": {
                "general": {
                    "low": ["openai/codex-1", "gemini/gemini-2.5-flash"],
                },
            },
            "fallback_models": [],
            "models": {
                "openai/codex-1": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "allowed_models": ["gemini/*"],
        "messages": [{"role": "user", "content": "hello"}],
    }

    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "gemini/gemini-2.5-flash"


def test_allowed_models_rejects_explicit_model_when_disallowed():
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/codex-1",
            "task_routes": {"general": {"default": "openai/codex-1"}},
            "models": {
                "openai/codex-1": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "openai/codex-1",
        "allowed_models": ["gemini/*"],
        "messages": [{"role": "user", "content": "hello"}],
    }

    with pytest.raises(RoutingConstraintError) as exc:
        router.decide(payload, "/v1/chat/completions")
    assert exc.value.constraint == "allowed_models"


def test_allowed_models_raises_when_no_auto_candidates_match():
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/codex-1",
            "task_routes": {
                "general": {
                    "low": ["openai/codex-1", "openai/gpt-5.2"],
                },
            },
            "fallback_models": [],
            "models": {
                "openai/codex-1": {"capabilities": ["chat"]},
                "openai/gpt-5.2": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "allowed_models": ["gemini/*"],
        "messages": [{"role": "user", "content": "hello"}],
    }

    with pytest.raises(RoutingConstraintError) as exc:
        router.decide(payload, "/v1/chat/completions")
    assert exc.value.constraint == "allowed_models"


def test_provider_preferences_are_carried_in_route_decision():
    router = _router()
    payload = {
        "model": "auto",
        "provider": {
            "order": ["gemini", "openai"],
            "sort": "price",
            "partition": "none",
            "require_parameters": True,
        },
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.provider_preferences == {
        "order": ["gemini", "openai"],
        "sort": "price",
        "partition": "none",
        "require_parameters": True,
    }


def test_routes_auto_coding_request():
    router = _router()
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "Debug this Python function and fix the bug in my SQL query.",
            }
        ],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.source == "auto"
    assert decision.task == "coding"
    assert decision.selected_model in {"code-7b", "code-14b", "code-32b"}


def test_routes_auto_coding_request_prefers_multi_model_route_order():
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "complexity": {"low_max_chars": 100, "medium_max_chars": 500, "high_max_chars": 2000},
            "task_routes": {
                "coding": {
                    "low": ["code-7b", "code-14b"],
                    "medium": ["code-14b", "code-32b"],
                    "high": ["code-32b"],
                    "xhigh": ["codex-1"],
                },
            },
            "fallback_models": ["general-14b", "general-32b"],
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "Debug this Python function and fix the bug in my SQL query.",
            }
        ],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "code-7b"
    assert decision.fallback_models == ["code-14b", "general-14b", "general-32b"]


def test_ignores_openclaw_style_system_preamble_for_simple_user_task():
    router = _router()
    long_system_prompt = (
        "You are a personal assistant running inside OpenClaw. "
        "Tooling includes code, function, bug, debug, refactor, python, typescript, sql, class, "
        "reason, analyze, evaluate, architect. "
    ) * 500
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": long_system_prompt},
            {"role": "user", "content": "Hello"},
        ],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.task == "general"
    assert decision.complexity == "low"
    assert decision.selected_model == "general-7b"
    assert decision.signals["text_scope"] == "user_messages"
    assert decision.signals["text_length_total"] > 2000


def test_routes_auto_image_request():
    router = _router()
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                ],
            }
        ],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.task == "image"
    assert decision.selected_model == "vision-11b"


def test_fallbacks_exclude_selected_model():
    router = _router()
    payload = {"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "general-7b"
    assert "general-7b" not in decision.fallback_models


def test_routes_xhigh_for_reasoning_effort_high():
    router = _router()
    payload = {
        "model": "auto",
        "reasoning": {"effort": "high"},
        "messages": [
            {
                "role": "user",
                "content": "Design a compiler architecture and reason about tradeoffs in depth.",
            }
        ],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.task == "thinking"
    assert decision.complexity == "xhigh"
    assert decision.selected_model == "think-codex"


def test_hard_constraints_filter_models_without_required_tool_capability():
    config = RoutingConfig.model_validate(
        {
            "default_model": "text-model",
            "task_routes": {
                "general": {
                    "low": ["text-model", "tool-model"],
                },
            },
            "models": {
                "text-model": {"capabilities": ["chat", "streaming"]},
                "tool-model": {"capabilities": ["chat", "streaming", "tool_use"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Call the weather tool"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
    }

    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "tool-model"


def test_hard_constraints_filter_models_that_exceed_output_token_limit():
    config = RoutingConfig.model_validate(
        {
            "default_model": "small-model",
            "task_routes": {
                "general": {
                    "low": ["small-model", "large-model"],
                },
            },
            "models": {
                "small-model": {
                    "capabilities": ["chat", "streaming"],
                    "limits": {"max_output_tokens": 128, "context_tokens": 4096},
                },
                "large-model": {
                    "capabilities": ["chat", "streaming"],
                    "limits": {"max_output_tokens": 2048, "context_tokens": 32768},
                },
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Generate a long answer"}],
        "max_tokens": 500,
    }

    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "large-model"


def test_hard_constraints_filter_models_without_enabled_account_support():
    config = RoutingConfig.model_validate(
        {
            "default_model": "supported-model",
            "task_routes": {
                "general": {
                    "low": ["unsupported-model", "supported-model"],
                },
            },
            "models": {
                "unsupported-model": {"capabilities": ["chat"]},
                "supported-model": {"capabilities": ["chat"]},
            },
            "learned_routing": {
                "enabled": True,
                "task_candidates": {
                    "general": ["unsupported-model", "supported-model"],
                },
            },
            "model_profiles": {
                "unsupported-model": {"quality_bias": 1.0, "quality_sensitivity": 2.0},
                "supported-model": {"quality_bias": 0.2, "quality_sensitivity": 1.0},
            },
            "accounts": [
                {
                    "name": "only-supported",
                    "provider": "openai",
                    "base_url": "http://localhost:8000",
                    "auth_mode": "passthrough",
                    "models": ["supported-model"],
                    "enabled": True,
                }
            ],
        }
    )
    router = SmartModelRouter(config)
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
        },
        endpoint="/v1/chat/completions",
    )
    assert decision.selected_model == "supported-model"
    assert (
        "no_enabled_account_supports_model"
        in decision.decision_trace["hard_constraint_rejections"]["unsupported-model"]
    )


def test_classifier_uses_recent_user_window_instead_of_full_history():
    router = _router()
    very_long_old_message = "reason analyze architect design plan " * 120
    messages = [{"role": "user", "content": very_long_old_message} for _ in range(30)]
    messages.extend([{"role": "user", "content": "ok"} for _ in range(10)])
    decision = router.decide(
        payload={"model": "auto", "messages": messages},
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "general"
    assert decision.complexity == "low"
