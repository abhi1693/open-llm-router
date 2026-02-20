from open_llm_router.config import RoutingConfig
from open_llm_router.routing.router_engine import SmartModelRouter


def _router() -> SmartModelRouter:
    config = RoutingConfig.model_validate(
        {
            "default_model": "cheap-model",
            "complexity": {
                "low_max_chars": 100,
                "medium_max_chars": 600,
                "high_max_chars": 2000,
            },
            "task_routes": {
                "general": {
                    "low": "cheap-model",
                    "medium": "cheap-model",
                    "high": "strong-model",
                },
                "coding": {
                    "low": "cheap-model",
                    "medium": "strong-model",
                    "high": "strong-model",
                    "xhigh": "strong-model",
                },
                "thinking": {"default": "strong-model"},
                "instruction_following": {"default": "cheap-model"},
                "image": {"default": "cheap-model"},
            },
            "fallback_models": ["strong-model"],
            "learned_routing": {
                "enabled": True,
                "bias": -4.0,
                "default_output_tokens": 400,
                "feature_weights": {
                    "complexity_score": 1.4,
                    "task_coding": 1.0,
                    "reasoning_effort_high": 1.2,
                    "code_score": 0.2,
                },
                "utility_weights": {"cost": 12.0, "latency": 0.2, "failure": 3.0},
                "task_candidates": {
                    "general": ["cheap-model", "strong-model"],
                    "coding": ["cheap-model", "strong-model"],
                    "thinking": ["cheap-model", "strong-model"],
                },
            },
            "model_profiles": {
                "cheap-model": {
                    "quality_bias": 0.1,
                    "quality_sensitivity": 0.6,
                    "cost_input_per_1k": 0.00012,
                    "cost_output_per_1k": 0.00035,
                    "latency_ms": 350,
                    "failure_rate": 0.02,
                },
                "strong-model": {
                    "quality_bias": 0.65,
                    "quality_sensitivity": 2.1,
                    "cost_input_per_1k": 0.0012,
                    "cost_output_per_1k": 0.004,
                    "latency_ms": 1450,
                    "failure_rate": 0.028,
                },
            },
        }
    )
    return SmartModelRouter(config)


def test_learned_router_prefers_cheaper_model_for_easy_prompt() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
        },
        endpoint="/v1/chat/completions",
    )
    assert decision.selected_model == "cheap-model"
    assert decision.signals["routing_mode"] == "learned_utility"
    assert decision.ranked_models[0] == "cheap-model"
    assert (
        decision.candidate_scores[0]["utility"]
        >= decision.candidate_scores[1]["utility"]
    )


def test_learned_router_prefers_stronger_model_for_hard_coding_prompt() -> None:
    router = _router()
    decision = router.decide(
        payload={
            "model": "auto",
            "reasoning": {"effort": "high"},
            "messages": [
                {
                    "role": "user",
                    "content": "Debug this Python function and optimize SQL performance with deep reasoning.",
                }
            ],
        },
        endpoint="/v1/chat/completions",
    )
    assert decision.task == "coding"
    assert decision.complexity == "xhigh"
    assert decision.selected_model == "strong-model"
    assert decision.ranked_models[0] == "strong-model"


def test_learned_router_keeps_candidates_within_rule_chain() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "cheap-model",
            "task_routes": {
                "coding": {
                    "low": "cheap-model",
                    "medium": "medium-model",
                    "high": "strong-model",
                    "xhigh": "xhigh-model",
                },
            },
            "fallback_models": ["medium-model"],
            "learned_routing": {
                "enabled": True,
                "task_candidates": {
                    # Includes xhigh-model, but low-complexity routing should stay on
                    # the rule-chain candidate set.
                    "coding": [
                        "cheap-model",
                        "medium-model",
                        "strong-model",
                        "xhigh-model",
                    ],
                },
            },
            "model_profiles": {
                "cheap-model": {
                    "quality_bias": 0.1,
                    "quality_sensitivity": 0.7,
                    "cost_input_per_1k": 0.0001,
                    "cost_output_per_1k": 0.0003,
                    "latency_ms": 300,
                    "failure_rate": 0.02,
                },
                "medium-model": {
                    "quality_bias": 0.35,
                    "quality_sensitivity": 1.3,
                    "cost_input_per_1k": 0.0006,
                    "cost_output_per_1k": 0.0018,
                    "latency_ms": 900,
                    "failure_rate": 0.03,
                },
                "strong-model": {
                    "quality_bias": 0.7,
                    "quality_sensitivity": 2.0,
                    "cost_input_per_1k": 0.0012,
                    "cost_output_per_1k": 0.004,
                    "latency_ms": 1400,
                    "failure_rate": 0.03,
                },
                "xhigh-model": {
                    "quality_bias": 0.9,
                    "quality_sensitivity": 2.3,
                    "cost_input_per_1k": 0.0022,
                    "cost_output_per_1k": 0.0072,
                    "latency_ms": 1600,
                    "failure_rate": 0.035,
                },
            },
        }
    )

    router = SmartModelRouter(config)
    decision = router.decide(
        payload={
            "model": "auto",
            "messages": [{"role": "user", "content": "Write a tiny python helper"}],
        },
        endpoint="/v1/chat/completions",
    )

    assert decision.complexity == "low"
    assert "xhigh-model" not in decision.ranked_models


def test_learned_router_ucb_explores_underused_models() -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "model-a",
            "task_routes": {
                "general": {
                    "low": ["model-a", "model-b"],
                },
            },
            "fallback_models": [],
            "learned_routing": {
                "enabled": True,
                "task_candidates": {
                    "general": ["model-a", "model-b"],
                },
            },
            "model_profiles": {
                "model-a": {
                    "quality_bias": 0.35,
                    "quality_sensitivity": 1.0,
                    "cost_input_per_1k": 0.0002,
                    "cost_output_per_1k": 0.0006,
                    "latency_ms": 300,
                    "failure_rate": 0.01,
                },
                "model-b": {
                    "quality_bias": 0.33,
                    "quality_sensitivity": 1.0,
                    "cost_input_per_1k": 0.0002,
                    "cost_output_per_1k": 0.0006,
                    "latency_ms": 300,
                    "failure_rate": 0.01,
                },
            },
        }
    )
    router = SmartModelRouter(config)
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }

    first = router.decide(payload, "/v1/chat/completions")
    second = router.decide(payload, "/v1/chat/completions")

    assert first.selected_model == "model-a"
    assert second.selected_model == "model-b"
    assert second.decision_trace["learned_trace"]["bandit"]["enabled"] is True
    assert "bandit_selection_score" in second.candidate_scores[0]
