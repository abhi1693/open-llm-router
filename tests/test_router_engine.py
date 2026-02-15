from smart_model_router.config import RoutingConfig
from smart_model_router.router_engine import SmartModelRouter


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
        "model": "my-custom-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    decision = router.decide(payload, "/v1/chat/completions")
    assert decision.selected_model == "my-custom-model"
    assert decision.source == "request"
    assert decision.task == "explicit"


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
