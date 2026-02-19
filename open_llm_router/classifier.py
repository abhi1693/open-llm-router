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

CODING_ACTION_HINTS = (
    "write",
    "implement",
    "build",
    "fix",
    "optimize",
    "debug",
    "refactor",
    "create",
)

CODING_OBJECT_HINTS = (
    "script",
    "function",
    "class",
    "api",
    "endpoint",
    "query",
    "sql",
    "test",
    "algorithm",
)

THINKING_INTENT_HINTS = (
    "why",
    "tradeoff",
    "pros and cons",
    "architecture",
    "strategy",
    "evaluate",
    "compare",
    "analyze",
    "reason",
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


def _collect_message_texts(
    payload: dict[str, Any], signals: dict[str, Any]
) -> tuple[list[str], list[str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return [], []

    user_texts: list[str] = []
    all_message_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower().strip()
        message_texts: list[str] = []
        _collect_text_and_signals(message.get("content"), "content", message_texts, signals)
        all_message_texts.extend(message_texts)
        if role == "user":
            user_texts.extend(message_texts)
    return user_texts, all_message_texts


def _scoped_user_text_blob(user_texts: list[str]) -> tuple[str, bool]:
    if not user_texts:
        return "", False
    considered = user_texts[-MAX_USER_MESSAGES_FOR_CLASSIFICATION:]
    blob = " ".join(considered).strip()
    if len(blob) <= MAX_USER_TEXT_CHARS_FOR_CLASSIFICATION:
        return blob, len(considered) != len(user_texts)

    trimmed_blob = blob[-MAX_USER_TEXT_CHARS_FOR_CLASSIFICATION:]
    return trimmed_blob, True


def _contains_any(text_lower: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text_lower for hint in hints)


def _task_scores(
    *,
    text_blob: str,
    text_lower: str,
    text_length: int,
    code_score: int,
    think_score: int,
    instruction_score: int,
    reasoning_effort: str | None,
    payload: dict[str, Any],
) -> dict[str, float]:
    coding_action = _contains_any(text_lower, CODING_ACTION_HINTS)
    coding_object = _contains_any(text_lower, CODING_OBJECT_HINTS)
    thinking_intent = _contains_any(text_lower, THINKING_INTENT_HINTS)

    response_format = payload.get("response_format")
    has_structured_output_hint = False
    if isinstance(response_format, dict):
        response_type = str(response_format.get("type") or "").strip().lower()
        has_structured_output_hint = response_type in {"json_object", "json_schema"}

    scores: dict[str, float] = {
        "general": 0.4,
        "coding": (float(code_score) * 1.2),
        "thinking": (float(think_score) * 1.25),
        "instruction_following": (float(instruction_score) * 1.25),
    }
    if "```" in text_blob:
        scores["coding"] += 1.0
    if coding_action and coding_object:
        scores["coding"] += 1.0
    elif coding_action or coding_object:
        scores["coding"] += 0.4
    if "stack trace" in text_lower:
        scores["coding"] += 0.6
    if payload.get("tools") or payload.get("functions"):
        scores["coding"] += 0.2

    if reasoning_effort is not None:
        scores["thinking"] += 1.2
    if thinking_intent:
        scores["thinking"] += 0.7

    if has_structured_output_hint:
        scores["instruction_following"] += 0.6

    if text_length <= 80:
        scores["general"] += 0.25
    if (
        max(
            scores["coding"],
            scores["thinking"],
            scores["instruction_following"],
        )
        < 0.9
    ):
        scores["general"] += 0.25

    return scores


def _select_task_from_scores(
    *,
    scores: dict[str, float],
    text_lower: str,
    text_length: int,
    code_score: int,
    instruction_score: int,
    reasoning_effort: str | None,
    medium_max_chars: int,
) -> tuple[str, str, float]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_task, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = max(0.0, top_score - second_score)
    reason = "score_top1"

    if top_task == "instruction_following" and text_length > medium_max_chars:
        for candidate_task, _ in ordered[1:]:
            if candidate_task != "instruction_following":
                top_task = candidate_task
                reason = "instruction_length_guardrail"
                break

    if confidence < 0.45:
        coding_action = _contains_any(text_lower, CODING_ACTION_HINTS)
        coding_object = _contains_any(text_lower, CODING_OBJECT_HINTS)
        if reasoning_effort is not None:
            top_task = "thinking"
            reason = "low_confidence_reasoning_effort_override"
        elif code_score >= 1 and (coding_action or coding_object):
            top_task = "coding"
            reason = "low_confidence_coding_intent_override"
        elif instruction_score >= 1 and text_length <= medium_max_chars:
            top_task = "instruction_following"
            reason = "low_confidence_instruction_override"

    return top_task, reason, confidence


def classify_request(
    payload: dict[str, Any],
    endpoint: str,
    complexity_cfg: ComplexityConfig,
) -> tuple[str, str, dict[str, Any]]:
    signals: dict[str, Any] = {"has_image": False}
    user_texts, message_texts = _collect_message_texts(payload, signals)
    if message_texts:
        full_text_blob = " ".join(message_texts).strip()
    else:
        texts: list[str] = []
        _collect_text_and_signals(payload, "", texts, signals)
        full_text_blob = " ".join(texts).strip()
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

    task_scores = _task_scores(
        text_blob=text_blob,
        text_lower=text_lower,
        text_length=text_length,
        code_score=code_score,
        think_score=think_score,
        instruction_score=instruction_score,
        reasoning_effort=reasoning_effort,
        payload=payload,
    )
    task_confidence = 1.0

    if endpoint.startswith("/v1/images") or signals["has_image"]:
        task = "image"
        task_reason = "image_endpoint_or_image_signal"
        task_scores["image"] = 1.0
    else:
        task, task_reason, task_confidence = _select_task_from_scores(
            scores=task_scores,
            text_lower=text_lower,
            text_length=text_length,
            code_score=code_score,
            instruction_score=instruction_score,
            reasoning_effort=reasoning_effort,
            medium_max_chars=complexity_cfg.medium_max_chars,
        )

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
            "task_confidence": round(task_confidence, 6),
            "task_scores": {
                name: round(score, 6) for name, score in sorted(task_scores.items())
            },
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
