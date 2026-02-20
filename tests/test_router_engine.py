import pytest

from open_llm_router.config import RoutingConfig
from open_llm_router.routing.router_engine import (
    InvalidModelError,
    RoutingConstraintError,
    SmartModelRouter,
)


def _router() -> SmartModelRouter:
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "complexity": {
                "low_max_chars": 100,
                "medium_max_chars": 500,
                "high_max_chars": 2000,
            },
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
                    "xhigh": "code-70b",
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


def test_respects_explicit_model() -> None:
    router = _router()
    payload = {
        "model": "code-14b",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "code-14b"
    assert decision.source == "request"
    assert decision.task == "explicit"


def test_rejects_unknown_explicit_model() -> None:
    router = _router()
    payload = {
        "model": "my-custom-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(InvalidModelError) as exc:
        router.decide(payload, "/v1/chat/completions")
    assert exc.value.requested_model == "my-custom-model"
    assert "not configured" in str(exc.value)


def test_respects_explicit_provider_qualified_model() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "models": ["openai-codex/gpt-5.2-codex", "general-14b"],
            "complexity": {
                "low_max_chars": 100,
                "medium_max_chars": 500,
                "high_max_chars": 2000,
            },
            "task_routes": {},
            "fallback_models": ["general-14b", "general-32b"],
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "openai-codex/gpt-5.2-codex",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "openai-codex/gpt-5.2-codex"
    assert decision.source == "request"


def test_treats_openrouter_auto_alias_as_auto_routing() -> None:
    router = _router()
    payload = {
        "model": "openrouter/auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.source == "auto"
    assert decision.task == "general"
    assert decision.selected_model == "general-7b"


def test_allowed_models_filters_auto_route_candidates() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai-codex/gpt-5.2-codex",
            "task_routes": {
                "general": {
                    "low": ["openai-codex/gpt-5.2-codex", "gemini/gemini-2.5-flash"],
                },
            },
            "fallback_models": [],
            "models": {
                "openai-codex/gpt-5.2-codex": {"capabilities": ["chat"]},
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


def test_allowed_models_rejects_explicit_model_when_disallowed() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai-codex/gpt-5.2-codex",
            "task_routes": {"general": {"default": "openai-codex/gpt-5.2-codex"}},
            "models": {
                "openai-codex/gpt-5.2-codex": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "openai-codex/gpt-5.2-codex",
        "allowed_models": ["gemini/*"],
        "messages": [{"role": "user", "content": "hello"}],
    }

    with pytest.raises(RoutingConstraintError) as exc:
        router.decide(payload, "/v1/chat/completions")
    assert exc.value.constraint == "allowed_models"


def test_allowed_models_raises_when_no_auto_candidates_match() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai-codex/gpt-5.2-codex",
            "task_routes": {
                "general": {
                    "low": ["openai-codex/gpt-5.2-codex", "openai/gpt-5.2"],
                },
            },
            "fallback_models": [],
            "models": {
                "openai-codex/gpt-5.2-codex": {"capabilities": ["chat"]},
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


def test_provider_preferences_are_carried_in_route_decision() -> None:
    router = _router()
    payload = {
        "model": "auto",
        "provider": {
            "order": ["gemini", "openai"],
            "sort": "price",
            "partition": "none",
            "require_parameters": True,
            "allow_fallbacks": False,
        },
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.provider_preferences == {
        "order": ["gemini", "openai"],
        "sort": "price",
        "partition": "none",
        "require_parameters": True,
        "allow_fallbacks": False,
    }


def test_provider_preferences_only_filters_route_candidates() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/gpt-5.2",
            "task_routes": {
                "general": {
                    "low": ["openai/gpt-5.2", "gemini/gemini-2.5-flash"],
                },
            },
            "models": {
                "openai/gpt-5.2": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "provider": {"only": ["gemini"]},
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "gemini/gemini-2.5-flash"


def test_provider_preferences_ignore_filters_route_candidates() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/gpt-5.2",
            "task_routes": {
                "general": {
                    "low": ["openai/gpt-5.2", "gemini/gemini-2.5-flash"],
                },
            },
            "models": {
                "openai/gpt-5.2": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "provider": {"ignore": ["openai"]},
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "gemini/gemini-2.5-flash"


def test_provider_preferences_order_reorders_rule_chain_head() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/gpt-5.2",
            "task_routes": {
                "general": {
                    "low": ["openai/gpt-5.2", "gemini/gemini-2.5-flash"],
                },
            },
            "models": {
                "openai/gpt-5.2": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "provider": {"order": ["gemini", "openai"]},
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "gemini/gemini-2.5-flash"
    assert decision.decision_trace["provider_preference_filtered_chain"][0] == (
        "gemini/gemini-2.5-flash"
    )


def test_provider_preferences_raise_when_filters_exclude_all_candidates() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/gpt-5.2",
            "task_routes": {
                "general": {
                    "low": ["openai/gpt-5.2", "gemini/gemini-2.5-flash"],
                },
            },
            "models": {
                "openai/gpt-5.2": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "provider": {"only": ["anthropic"]},
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(RoutingConstraintError) as exc:
        router.decide(payload, "/v1/chat/completions")
    assert exc.value.constraint == "provider_preferences"


def test_provider_order_bias_applies_in_learned_routing() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai/gpt-5.2",
            "task_routes": {
                "general": {
                    "low": ["openai/gpt-5.2", "gemini/gemini-2.5-flash"],
                },
            },
            "learned_routing": {
                "enabled": True,
                "task_candidates": {
                    "general": ["openai/gpt-5.2", "gemini/gemini-2.5-flash"],
                },
            },
            "model_profiles": {
                "openai/gpt-5.2": {
                    "quality_bias": 0.2,
                    "quality_sensitivity": 1.0,
                    "cost_input_per_1k": 0.0002,
                    "cost_output_per_1k": 0.0006,
                    "latency_ms": 300,
                    "failure_rate": 0.01,
                },
                "gemini/gemini-2.5-flash": {
                    "quality_bias": 0.2,
                    "quality_sensitivity": 1.0,
                    "cost_input_per_1k": 0.0002,
                    "cost_output_per_1k": 0.0006,
                    "latency_ms": 300,
                    "failure_rate": 0.01,
                },
            },
            "models": {
                "openai/gpt-5.2": {"capabilities": ["chat"]},
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "provider": {"order": ["gemini", "openai"]},
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "gemini/gemini-2.5-flash"
    assert decision.decision_trace["learned_trace"]["provider_order_applied"] is True
    assert "provider_order_bonus" in decision.candidate_scores[0]


def test_routes_auto_coding_request() -> None:
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


def test_routes_auto_coding_request_prefers_multi_model_route_order() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "complexity": {
                "low_max_chars": 100,
                "medium_max_chars": 500,
                "high_max_chars": 2000,
            },
            "task_routes": {
                "coding": {
                    "low": ["code-7b", "code-14b"],
                    "medium": ["code-14b", "code-32b"],
                    "high": ["code-32b"],
                    "xhigh": ["code-70b"],
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


def test_ignores_openclaw_style_system_preamble_for_simple_user_task() -> None:
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


def test_routes_auto_image_request() -> None:
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


def test_fallbacks_exclude_selected_model() -> None:
    router = _router()
    payload = {"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "general-7b"
    assert "general-7b" not in decision.fallback_models


def test_routes_xhigh_for_reasoning_effort_high() -> None:
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


def test_hard_constraints_filter_models_without_required_tool_capability() -> None:
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


def test_hard_constraints_filter_models_that_exceed_output_token_limit() -> None:
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


def test_hard_constraints_allow_small_context_overflow_with_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "small-model",
            "task_routes": {
                "general": {
                    "default": ["small-model", "large-model"],
                },
            },
            "models": {
                "small-model": {
                    "capabilities": ["chat", "streaming"],
                    "limits": {"context_tokens": 800},
                },
                "large-model": {
                    "capabilities": ["chat", "streaming"],
                    "limits": {"context_tokens": 1000},
                },
            },
        }
    )
    monkeypatch.setattr(
        "open_llm_router.routing.router_engine._estimate_payload_tokens",
        lambda **_: (1050, "test_override"),
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }

    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "large-model"
    assert any(
        reason.startswith("context_window_exceeded:")
        for reason in decision.decision_trace["hard_constraint_rejections"][
            "small-model"
        ]
    )
    assert (
        decision.decision_trace["hard_constraint_token_estimation"] == "test_override"
    )


def test_hard_constraints_supplement_large_context_single_candidate_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "primary-model",
            "task_routes": {
                "general": {
                    "default": ["primary-model"],
                },
            },
            "fallback_models": [],
            "models": {
                "primary-model": {
                    "capabilities": ["chat", "streaming", "tool_use"],
                    "limits": {"context_tokens": 500000},
                },
                "backup-model": {
                    "capabilities": ["chat", "streaming", "tool_use"],
                    "limits": {"context_tokens": 500000},
                },
            },
        }
    )
    monkeypatch.setattr(
        "open_llm_router.routing.router_engine._estimate_payload_tokens",
        lambda **_: (130000, "test_override"),
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {"name": "noop", "parameters": {"type": "object"}},
            }
        ],
        "messages": [{"role": "user", "content": "hello"}],
    }

    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "primary-model"
    assert "backup-model" in decision.fallback_models
    assert decision.decision_trace["hard_constraint_supplemented_models"] == [
        "backup-model"
    ]


def test_hard_constraints_filter_models_without_enabled_account_support() -> None:
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


def test_classifier_uses_recent_user_window_instead_of_full_history() -> None:
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


def test_routes_single_hint_coding_intent_prompt() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python script that validates CSV rows.",
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "coding"
    assert decision.selected_model in {"code-7b", "code-14b", "code-32b"}
    assert decision.signals["task_confidence"] >= 0.0


def test_secondary_classifier_disambiguates_instruction_vs_coding() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Reword this paragraph about Python so it sounds professional.",
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "instruction_following"
    assert decision.selected_model == "general-7b"
    assert decision.signals["secondary_classifier_used"] is True


def test_semantic_classifier_routes_coding_intent_without_code_keywords() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Can you craft a helper that iterates through rows in a "
                        "comma-separated file and flags malformed records?"
                    ),
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "coding"
    assert decision.selected_model in {"code-7b", "code-14b", "code-32b"}
    assert decision.signals["semantic_classifier_used"] is True
    assert decision.signals["task_reason"] == "semantic_classifier_override"


def test_semantic_classifier_routes_thinking_intent_from_tradeoff_language() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Walk through competing approaches and justify which "
                        "direction we should take."
                    ),
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "thinking"
    assert decision.selected_model in {
        "think-14b",
        "think-32b",
        "think-70b",
        "think-codex",
    }
    assert decision.signals["semantic_classifier_used"] is True
    assert decision.signals["task_reason"] == "semantic_classifier_override"


def test_semantic_classifier_routes_rewrite_style_instruction_intent() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Tighten this paragraph so it reads professionally without "
                        "changing the meaning."
                    ),
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "instruction_following"
    assert decision.selected_model == "general-7b"
    assert decision.signals["semantic_classifier_used"] is True
    assert decision.signals["task_reason"] == "semantic_classifier_override"


def test_local_embedding_semantic_classifier_can_be_used_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "open_llm_router.routing.classifier._semantic_task_prediction_local",
        lambda **_: (
            "thinking",
            0.9,
            {
                "general": 0.1,
                "coding": 0.2,
                "thinking": 0.95,
                "instruction_following": 0.15,
            },
        ),
    )
    config = RoutingConfig.model_validate(
        {
            "default_model": "think-14b",
            "task_routes": {
                "general": {
                    "low": "general-7b",
                    "medium": "general-14b",
                    "high": "general-32b",
                    "xhigh": "general-70b",
                },
                "thinking": {"default": "think-14b"},
                "instruction_following": {"default": "general-7b"},
            },
            "semantic_classifier": {
                "enabled": True,
                "backend": "local_embedding",
                "local_model_name": "/models/local-embed",
                "local_files_only": True,
                "min_confidence": 0.1,
            },
        }
    )
    router = SmartModelRouter(config)
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [{"role": "user", "content": "help me decide a direction"}],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "thinking"
    assert decision.selected_model == "think-14b"
    assert decision.signals["semantic_classifier_source"] == "local_embedding"
    assert decision.signals["semantic_classifier_status"] is None


def test_local_embedding_semantic_classifier_falls_back_to_prototype_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "open_llm_router.routing.classifier._semantic_task_prediction_local",
        lambda **_: None,
    )
    config = RoutingConfig.model_validate(
        {
            "default_model": "general-14b",
            "task_routes": {
                "general": {
                    "low": "general-7b",
                    "medium": "general-14b",
                    "high": "general-32b",
                    "xhigh": "general-70b",
                },
                "instruction_following": {"default": "general-7b"},
            },
            "semantic_classifier": {
                "enabled": True,
                "backend": "local_embedding",
                "local_model_name": "/models/local-embed",
                "local_files_only": True,
            },
        }
    )
    router = SmartModelRouter(config)
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Tighten this paragraph so it reads professionally without "
                        "changing the meaning."
                    ),
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "instruction_following"
    assert decision.signals["semantic_classifier_source"] == "prototype"
    assert decision.signals["semantic_classifier_status"] == (
        "local_embedding_unavailable"
    )


def test_route_reranker_can_reorder_rule_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_embedding_for_text(
        *, model_name: str, local_files_only: bool, max_length: int, text: str
    ) -> tuple[float, float] | None:
        _ = (model_name, local_files_only, max_length)
        normalized = text.lower()
        if "task:coding" in normalized or "write code" in normalized:
            return (1.0, 0.0)
        if "general chit chat" in normalized:
            return (0.0, 1.0)
        if "specialty:coding" in normalized:
            return (1.0, 0.0)
        return (0.0, 1.0)

    monkeypatch.setattr(
        "open_llm_router.routing.router_engine._local_embedding_for_text",
        _fake_embedding_for_text,
    )

    config = RoutingConfig.model_validate(
        {
            "default_model": "general-7b",
            "task_routes": {
                "general": {"default": "general-7b"},
                "coding": {"low": ["general-7b", "code-7b"]},
            },
            "models": {
                "general-7b": {"capabilities": ["chat"]},
                "code-7b": {"capabilities": ["chat"]},
            },
            "route_reranker": {
                "enabled": True,
                "backend": "local_embedding",
                "local_model_name": "/models/local-reranker",
                "local_files_only": True,
                "local_max_length": 256,
                "similarity_weight": 4.0,
                "min_similarity": 0.0,
                "model_hints": {
                    "general-7b": "general chit chat summarization",
                    "code-7b": "write code debug compile",
                },
            },
        }
    )
    router = SmartModelRouter(config)
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Write code to parse CSV rows safely."}
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "coding"
    assert decision.selected_model == "code-7b"
    assert decision.ranked_models[0] == "code-7b"
    assert decision.decision_trace["rule_chain_reranker"]["status"] == "applied"


def test_route_reranker_can_shift_learned_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_embedding_for_text(
        *, model_name: str, local_files_only: bool, max_length: int, text: str
    ) -> tuple[float, float] | None:
        _ = (model_name, local_files_only, max_length)
        normalized = text.lower()
        if "task:coding" in normalized or "write code" in normalized:
            return (1.0, 0.0)
        if "general writing assistant" in normalized:
            return (0.0, 1.0)
        if "software engineering code generation" in normalized:
            return (1.0, 0.0)
        return (0.0, 1.0)

    monkeypatch.setattr(
        "open_llm_router.routing.router_engine._local_embedding_for_text",
        _fake_embedding_for_text,
    )

    base_config = RoutingConfig.model_validate(
        {
            "default_model": "general-7b",
            "task_routes": {
                "general": {"default": "general-7b"},
                "coding": {"low": ["general-7b", "code-7b"]},
            },
            "models": {
                "general-7b": {"capabilities": ["chat"]},
                "code-7b": {"capabilities": ["chat"]},
            },
            "model_profiles": {
                "general-7b": {"quality_bias": 1.2},
                "code-7b": {"quality_bias": -0.8},
            },
            "learned_routing": {
                "enabled": True,
                "task_candidates": {"coding": ["general-7b", "code-7b"]},
                "utility_weights": {"cost": 0.0, "latency": 0.0, "failure": 0.0},
            },
        }
    )
    baseline_router = SmartModelRouter(base_config)
    baseline_decision = baseline_router.decide(
        payload={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Write code to parse CSV rows safely."}
            ],
        },
        endpoint="/v1/chat/completions",
    )

    reranked_config = base_config.model_copy(deep=True)
    reranked_config.route_reranker.enabled = True
    reranked_config.route_reranker.local_model_name = "/models/local-reranker"
    reranked_config.route_reranker.similarity_weight = 5.0
    reranked_config.route_reranker.model_hints = {
        "general-7b": "general writing assistant",
        "code-7b": "software engineering code generation",
    }
    reranked_router = SmartModelRouter(reranked_config)
    reranked_decision = reranked_router.decide(
        payload={
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Write code to parse CSV rows safely."}
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert baseline_decision.selected_model == "general-7b"
    assert reranked_decision.selected_model == "code-7b"
    assert reranked_decision.decision_trace["learned_trace"]["reranker"]["status"] == (
        "applied"
    )
    assert reranked_decision.candidate_scores[0]["model"] == "code-7b"
    assert "reranker_bonus" in reranked_decision.candidate_scores[0]


def test_routes_coding_without_language_name_using_structural_signals() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Fix this crash from app/core/main.foo:42\n"
                        "Traceback: Exception while processing request\n"
                        "$ make test\n"
                        "for (i = 0; i < n; i++) { total += values[i]; }"
                    ),
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "coding"
    assert decision.selected_model in {"code-7b", "code-14b", "code-32b"}
    assert decision.signals["structural_code_score"] > 0.0


def test_latest_factual_question_overrides_stale_coding_context() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Debug this crash and fix the failing module test.",
                },
                {
                    "role": "assistant",
                    "content": "I can help with that debug flow.",
                },
                {
                    "role": "user",
                    "content": "Question: who is the president of india",
                },
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "general"
    assert decision.selected_model == "general-7b"
    assert decision.signals["task_reason"] == "latest_turn_factual_override"
    assert decision.signals["latest_turn_override_applied"] is True


def test_latest_coding_question_is_not_downgraded_to_general() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Debug this crash and fix the failing module test.",
                },
                {
                    "role": "assistant",
                    "content": "Share the latest error and trace.",
                },
                {
                    "role": "user",
                    "content": "What is this stack trace telling me?",
                },
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "coding"
    assert decision.selected_model in {"code-7b", "code-14b", "code-32b"}
    assert decision.signals["latest_turn_override_applied"] is False


def test_factual_general_query_pins_rule_chain_head_over_learned_reorder() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "openai-codex/gpt-5.2-codex",
            "task_routes": {
                "general": {
                    "low": ["gemini/gemini-2.5-flash", "openai-codex/gpt-5.2-codex"],
                },
            },
            "fallback_models": [],
            "learned_routing": {
                "enabled": True,
                "task_candidates": {
                    "general": [
                        "gemini/gemini-2.5-flash",
                        "openai-codex/gpt-5.2-codex",
                    ],
                },
            },
            "model_profiles": {
                "gemini/gemini-2.5-flash": {
                    "quality_bias": -2.0,
                    "quality_sensitivity": 0.2,
                    "cost_input_per_1k": 0.0003,
                    "cost_output_per_1k": 0.0012,
                    "latency_ms": 650,
                    "failure_rate": 0.02,
                },
                "openai-codex/gpt-5.2-codex": {
                    "quality_bias": 2.5,
                    "quality_sensitivity": 1.4,
                    "cost_input_per_1k": 0.0015,
                    "cost_output_per_1k": 0.006,
                    "latency_ms": 1300,
                    "failure_rate": 0.03,
                },
            },
            "models": {
                "gemini/gemini-2.5-flash": {"capabilities": ["chat"]},
                "openai-codex/gpt-5.2-codex": {"capabilities": ["chat"]},
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Question: who is the president of india"}
        ],
    }

    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.task == "general"
    assert decision.complexity == "low"
    assert decision.selected_model == "gemini/gemini-2.5-flash"
    assert (
        decision.decision_trace["selected_reason"]
        == "factual_query_rule_chain_guardrail"
    )


def test_factual_question_with_non_user_role_still_routes_general() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Debug this crash and inspect traceback in main.foo:42",
                },
                {
                    "role": "human",
                    "content": "Question:Who is the president of india?",
                },
            ],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.task == "general"
    assert decision.complexity == "low"
    assert decision.selected_model == "general-7b"
    assert decision.signals["latest_user_factual_query"] is True
