from __future__ import annotations

from typing import Any

from open_llm_router.config import ComplexityConfig

CODE_HINTS = (
    "code",
    "function",
    "bug",
    "debug",
    "stack trace",
    "refactor",
    "python",
    "javascript",
    "typescript",
    "sql",
    "algorithm",
    "class ",
    "compile",
)

THINKING_HINTS = (
    "reason",
    "analyze",
    "deeply",
    "step by step",
    "tradeoff",
    "compare",
    "evaluate",
    "architect",
    "design plan",
)

INSTRUCTION_HINTS = (
    "rewrite",
    "summarize",
    "translate",
    "classify",
    "extract",
    "format",
    "convert",
    "paraphrase",
)

SKIP_TEXT_KEYS = {
    "model",
    "id",
    "name",
    "object",
    "role",
    "tool_call_id",
    "finish_reason",
}

MAX_USER_MESSAGES_FOR_CLASSIFICATION = 8
MAX_USER_TEXT_CHARS_FOR_CLASSIFICATION = 8000


def _bump_complexity(level: str) -> str:
    if level == "low":
        return "medium"
    if level == "medium":
        return "high"
    if level == "high":
        return "xhigh"
    return "xhigh"


def _extract_reasoning_effort(payload: dict[str, Any]) -> str | None:
    direct = payload.get("reasoning_effort")
    if isinstance(direct, str):
        return direct.lower().strip()

    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str):
            return effort.lower().strip()
    return None


def _collect_text_and_signals(
    value: Any, parent_key: str, texts: list[str], signals: dict[str, Any]
) -> None:
    if value is None:
        return

    if isinstance(value, str):
        if "image" in parent_key:
            signals["has_image"] = True
            return
        texts.append(value)
        return

    if isinstance(value, list):
        for item in value:
            _collect_text_and_signals(item, parent_key, texts, signals)
        return

    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = raw_key.lower()
            if key in SKIP_TEXT_KEYS:
                continue
            if "image" in key:
                signals["has_image"] = True
            if key == "type":
                if isinstance(child, str) and "image" in child.lower():
                    signals["has_image"] = True
                continue
            _collect_text_and_signals(child, key, texts, signals)


def _collect_user_message_texts(
    payload: dict[str, Any], signals: dict[str, Any]
) -> list[str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []

    user_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str) or role.lower().strip() != "user":
            continue
        _collect_text_and_signals(
            message.get("content"), "content", user_texts, signals
        )
    return user_texts


def _scoped_user_text_blob(user_texts: list[str]) -> tuple[str, bool]:
    if not user_texts:
        return "", False
    considered = user_texts[-MAX_USER_MESSAGES_FOR_CLASSIFICATION:]
    blob = " ".join(considered).strip()
    if len(blob) <= MAX_USER_TEXT_CHARS_FOR_CLASSIFICATION:
        return blob, len(considered) != len(user_texts)

    trimmed_blob = blob[-MAX_USER_TEXT_CHARS_FOR_CLASSIFICATION:]
    return trimmed_blob, True


def classify_request(
    payload: dict[str, Any],
    endpoint: str,
    complexity_cfg: ComplexityConfig,
) -> tuple[str, str, dict[str, Any]]:
    texts: list[str] = []
    signals: dict[str, Any] = {"has_image": False}
    _collect_text_and_signals(payload, "", texts, signals)

    full_text_blob = " ".join(texts).strip()
    user_texts = _collect_user_message_texts(payload, signals)
    if user_texts:
        text_scope = "user_messages"
        text_blob, user_scope_truncated = _scoped_user_text_blob(user_texts)
    else:
        text_scope = "payload"
        text_blob = full_text_blob
        user_scope_truncated = False

    text_lower = text_blob.lower()
    text_length = len(text_blob)
    full_text_length = len(full_text_blob)
    reasoning_effort = _extract_reasoning_effort(payload)

    matched_code_hints = [hint for hint in CODE_HINTS if hint in text_lower]
    matched_thinking_hints = [hint for hint in THINKING_HINTS if hint in text_lower]
    matched_instruction_hints = [
        hint for hint in INSTRUCTION_HINTS if hint in text_lower
    ]

    code_score = len(matched_code_hints)
    think_score = len(matched_thinking_hints)
    instruction_score = len(matched_instruction_hints)
    if "```" in text_blob:
        code_score += 2

    if text_length <= complexity_cfg.low_max_chars:
        complexity = "low"
    elif text_length <= complexity_cfg.medium_max_chars:
        complexity = "medium"
    elif text_length <= complexity_cfg.high_max_chars:
        complexity = "high"
    else:
        complexity = "xhigh"

    complexity_base = complexity
    complexity_adjustments: list[str] = []

    if endpoint.startswith("/v1/images") or signals["has_image"]:
        task = "image"
        task_reason = "image_endpoint_or_image_signal"
    elif code_score >= 2:
        task = "coding"
        task_reason = "code_score>=2"
    elif think_score >= 2 or reasoning_effort is not None:
        task = "thinking"
        task_reason = "think_score>=2_or_reasoning_effort_set"
    elif instruction_score >= 1 and text_length <= complexity_cfg.medium_max_chars:
        task = "instruction_following"
        task_reason = "instruction_score>=1_and_length_within_medium"
    else:
        task = "general"
        task_reason = "default_general"

    if think_score >= 2:
        complexity = _bump_complexity(complexity)
        complexity_adjustments.append("bump:think_score>=2")
    if think_score >= 4:
        complexity = _bump_complexity(complexity)
        complexity_adjustments.append("bump:think_score>=4")
    # Avoid escalating short coding prompts too aggressively; keep low-tier routing for
    # concise requests and reserve medium-tier bumps for longer prompts.
    if (
        task == "coding"
        and complexity == "low"
        and text_length > int(complexity_cfg.low_max_chars * 0.8)
    ):
        complexity = "medium"
        complexity_adjustments.append("set:low->medium_for_coding_mid_length")
    if reasoning_effort in {"high", "xhigh", "max"}:
        complexity = "xhigh"
        complexity_adjustments.append("set:xhigh_for_reasoning_effort_high")
    elif reasoning_effort in {"medium"}:
        complexity = _bump_complexity(complexity)
        complexity_adjustments.append("bump:reasoning_effort_medium")

    signals.update(
        {
            "text_scope": text_scope,
            "text_length": text_length,
            "text_length_total": full_text_length,
            "user_messages_count_total": len(user_texts),
            "user_messages_count_considered": (
                min(len(user_texts), MAX_USER_MESSAGES_FOR_CLASSIFICATION)
                if user_texts
                else 0
            ),
            "user_text_scope_truncated": user_scope_truncated,
            "code_score": code_score,
            "think_score": think_score,
            "instruction_score": instruction_score,
            "reasoning_effort": reasoning_effort,
            "task_reason": task_reason,
            "complexity_base": complexity_base,
            "complexity_adjustments": complexity_adjustments,
            "complexity_final": complexity,
            "matched_code_hints": matched_code_hints,
            "matched_thinking_hints": matched_thinking_hints,
            "matched_instruction_hints": matched_instruction_hints,
            "text_preview": text_blob[:500],
            "text_preview_total": full_text_blob[:500],
            "endpoint": endpoint,
        }
    )
    return task, complexity, signals
