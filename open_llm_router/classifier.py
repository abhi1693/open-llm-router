from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from open_llm_router.config import (
    ClassifierCalibrationConfig,
    ComplexityConfig,
    SemanticClassifierConfig,
)

CODE_HINTS = (
    "code",
    "function",
    "bug",
    "debug",
    "stack trace",
    "refactor",
    "compile",
    "syntax error",
    "exception",
    "traceback",
    "test case",
    "algorithm",
    "class ",
    "repository",
    "codebase",
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
    "reword",
    "rephrase",
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
    "test",
    "algorithm",
    "module",
    "library",
    "code",
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

SECONDARY_INSTRUCTION_HINTS = (
    "rewrite",
    "reword",
    "paraphrase",
    "summarize",
    "translate",
    "polish",
    "clean up",
    "fix grammar",
    "improve wording",
    "rephrase",
)

SECONDARY_TASK_CUES: dict[str, tuple[tuple[str, float], ...]] = {
    "coding": (
        ("write code", 1.0),
        ("fix bug", 1.0),
        ("debug", 0.9),
        ("stack trace", 1.0),
        ("traceback", 0.9),
        ("syntax error", 0.8),
        ("compile error", 0.8),
        ("unit test", 0.6),
        ("test case", 0.55),
        ("```", 0.7),
        ("function", 0.35),
        ("api", 0.3),
    ),
    "thinking": (
        ("tradeoff", 1.0),
        ("pros and cons", 1.0),
        ("compare", 0.8),
        ("evaluate", 0.8),
        ("which is better", 0.8),
        ("should i", 0.7),
        ("architecture", 0.7),
        ("analyze", 0.7),
        ("reason", 0.7),
    ),
    "instruction_following": (
        ("rewrite", 0.95),
        ("reword", 0.95),
        ("paraphrase", 0.95),
        ("summarize", 0.95),
        ("translate", 0.95),
        ("polish", 0.8),
        ("clean up", 0.8),
        ("fix grammar", 0.85),
        ("improve wording", 0.85),
        ("rephrase", 0.9),
        ("format", 0.7),
        ("convert", 0.65),
        ("extract", 0.65),
    ),
    "general": (
        ("hello", 0.4),
        ("hi", 0.3),
        ("thanks", 0.3),
        ("what is", 0.3),
        ("tell me about", 0.3),
    ),
}

SECONDARY_TASK_TOKEN_HINTS: dict[str, frozenset[str]] = {
    "coding": frozenset(
        {
            "code",
            "function",
            "debug",
            "bug",
            "traceback",
            "exception",
            "compile",
            "syntax",
            "query",
            "endpoint",
            "script",
            "api",
            "algorithm",
            "test",
            "module",
            "library",
            "class",
        }
    ),
    "thinking": frozenset(
        {
            "tradeoff",
            "compare",
            "evaluate",
            "architecture",
            "strategy",
            "analyze",
            "reason",
            "decide",
            "choose",
            "better",
        }
    ),
    "instruction_following": frozenset(
        {
            "rewrite",
            "reword",
            "paraphrase",
            "summarize",
            "translate",
            "polish",
            "format",
            "convert",
            "extract",
            "rephrase",
            "grammar",
        }
    ),
    "general": frozenset({"hello", "hi", "thanks", "explain", "what", "tell"}),
}

SEMANTIC_TASK_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "coding": (
        "build a utility to parse records and validate data",
        "write program logic and implement function behavior",
        "diagnose runtime failure and repair broken code",
        "improve query performance and optimize implementation",
        "create script for data processing automation",
        "iterate rows in a comma separated file and flag malformed records",
    ),
    "thinking": (
        "compare competing approaches and justify recommendation",
        "evaluate alternatives and reason through tradeoffs",
        "analyze strategy options and pick a direction",
        "assess strengths weaknesses and decision criteria",
        "walk through architecture choices and implications",
    ),
    "instruction_following": (
        "tighten wording while preserving original meaning",
        "revise paragraph to sound professional and clear",
        "clean up writing style and improve readability",
        "rephrase content without changing intent",
        "transform text into concise polished language",
    ),
    "general": (
        "ask for factual information",
        "general conversation and lightweight help",
        "simple greeting or broad explanation request",
    ),
}

_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "we",
        "what",
        "with",
        "you",
        "your",
    }
)

_SECONDARY_TOKEN_PATTERN = re.compile(r"[a-z0-9_+#.-]+")
_CODE_SYMBOL_PATTERN = re.compile(r"[{}();<>]")
_CODE_LINE_PATTERN = re.compile(
    r"^\s*(?:"
    r"if|for|while|return|class|def|fn|func|let|const|var|"
    r"import|from|public|private|protected|switch|case|try|catch|"
    r"select|insert|update|delete|create|alter|with|begin|end|"
    r"#include|package|module|namespace"
    r")\b",
    re.IGNORECASE,
)
_STACK_TRACE_PATTERN = re.compile(
    r"\b(?:traceback|stack trace|exception|segmentation fault|segfault|"
    r"syntaxerror|typeerror|referenceerror|nullpointerexception|compile error|compiler error)\b",
    re.IGNORECASE,
)
_SOURCE_LOCATION_PATTERN = re.compile(
    r"\b[a-z0-9_./\\-]+\.[a-z][a-z0-9]{0,7}:\d+\b",
    re.IGNORECASE,
)
_BUILD_OR_TEST_COMMAND_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[$>]\s*)?(?:"
    r"npm|yarn|pnpm|pip|poetry|cargo|go|mvn|gradle|dotnet|make|cmake|"
    r"pytest|jest|vitest|ruff|eslint|tsc|gcc|clang"
    r")\b",
    re.IGNORECASE,
)
_LATEST_FACTUAL_QUERY_PATTERN = re.compile(
    r"(?:^|[\s.,:;!?])(?:question\s*:\s*)?(?:who|what|when|where|which|whom)\s+"
    r"(?:is|are|was|were|did|do|does|has|have|can|will)\b",
    re.IGNORECASE,
)
_LATEST_CONTEXT_REFERENCE_HINTS = (
    "above",
    "earlier",
    "previous",
    "prior message",
    "from before",
    "same as before",
    "continue",
    "that code",
    "this code",
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
) -> tuple[list[str], list[str], str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return [], [], "", ""

    user_texts: list[str] = []
    all_message_texts: list[str] = []
    latest_user_text = ""
    latest_conversation_text = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower().strip()
        message_texts: list[str] = []
        _collect_text_and_signals(message.get("content"), "content", message_texts, signals)
        all_message_texts.extend(message_texts)
        message_blob = " ".join(message_texts).strip()
        if role == "user":
            user_texts.extend(message_texts)
            if message_blob:
                latest_user_text = message_blob
        if message_blob and role not in {"system"}:
            latest_conversation_text = message_blob
    return user_texts, all_message_texts, latest_user_text, latest_conversation_text


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


@lru_cache(maxsize=2048)
def _secondary_tokens(text_lower: str) -> tuple[str, ...]:
    return tuple(_SECONDARY_TOKEN_PATTERN.findall(text_lower))


def _secondary_task_prediction(text_lower: str) -> tuple[str, float, dict[str, float]]:
    scores: dict[str, float] = {
        "general": 0.1,
        "coding": 0.0,
        "thinking": 0.0,
        "instruction_following": 0.0,
    }

    for task, cues in SECONDARY_TASK_CUES.items():
        for cue, weight in cues:
            if cue in text_lower:
                scores[task] += weight

    token_set = set(_secondary_tokens(text_lower))
    for task, hints in SECONDARY_TASK_TOKEN_HINTS.items():
        overlap = len(token_set.intersection(hints))
        if overlap > 0:
            scores[task] += min(0.9, overlap * 0.22)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_task, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = (top_score - second_score) / max(1.0, top_score)
    return top_task, max(0.0, confidence), scores


def _normalize_semantic_token(token: str) -> str:
    normalized = token.lower().strip()
    if not normalized or normalized in _SEMANTIC_STOPWORDS:
        return ""
    if len(normalized) <= 2:
        return ""
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ing") and len(normalized) > 5:
        normalized = normalized[:-3]
    elif normalized.endswith("ed") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("es") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]
    return normalized


def _semantic_token_set(text_lower: str) -> set[str]:
    tokens = set()
    for raw in _SECONDARY_TOKEN_PATTERN.findall(text_lower):
        normalized = _normalize_semantic_token(raw)
        if normalized:
            tokens.add(normalized)
    return tokens


@lru_cache(maxsize=1)
def _semantic_prototype_tokens() -> dict[str, set[str]]:
    prototype_tokens: dict[str, set[str]] = {}
    for task, phrases in SEMANTIC_TASK_PROTOTYPES.items():
        task_tokens: set[str] = set()
        for phrase in phrases:
            task_tokens.update(_semantic_token_set(phrase.lower()))
        prototype_tokens[task] = task_tokens
    return prototype_tokens


def _semantic_task_prediction_prototype(
    text_lower: str,
) -> tuple[str, float, dict[str, float]]:
    query_tokens = _semantic_token_set(text_lower)
    if not query_tokens:
        return "general", 0.0, {"general": 0.05}

    prototype_tokens = _semantic_prototype_tokens()
    scores: dict[str, float] = {
        "general": 0.05,
        "coding": 0.0,
        "thinking": 0.0,
        "instruction_following": 0.0,
    }
    for task, task_tokens in prototype_tokens.items():
        if not task_tokens:
            continue
        overlap = len(query_tokens.intersection(task_tokens))
        if overlap <= 0:
            continue
        coverage = overlap / max(1.0, float(len(query_tokens)))
        relevance = overlap / max(1.0, float(len(task_tokens)))
        scores[task] += (coverage * 1.4) + (relevance * 0.9)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_task, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = (top_score - second_score) / max(1.0, top_score)
    return top_task, max(0.0, confidence), scores


@lru_cache(maxsize=4)
def _load_local_embedding_runtime(
    model_name: str,
    local_files_only: bool,
) -> tuple[Any, Any, Any] | None:
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model.eval()
    except Exception:
        return None
    return tokenizer, model, torch


@lru_cache(maxsize=4096)
def _local_embedding_for_text(
    *,
    model_name: str,
    local_files_only: bool,
    max_length: int,
    text: str,
) -> tuple[float, ...] | None:
    runtime = _load_local_embedding_runtime(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    if runtime is None:
        return None
    tokenizer, model, torch = runtime

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked = hidden * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    values = normalized[0].detach().cpu().tolist()
    return tuple(float(value) for value in values)


def _mean_normalized_vector(vectors: list[tuple[float, ...]]) -> tuple[float, ...] | None:
    if not vectors:
        return None
    dimensions = len(vectors[0])
    if dimensions <= 0:
        return None
    if any(len(vector) != dimensions for vector in vectors):
        return None

    sums = [0.0] * dimensions
    for vector in vectors:
        for idx, value in enumerate(vector):
            sums[idx] += value
    count = float(len(vectors))
    mean = [value / count for value in sums]
    norm = sum(value * value for value in mean) ** 0.5
    if norm <= 1e-12:
        return None
    return tuple(value / norm for value in mean)


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(left * right for left, right in zip(a, b))


@lru_cache(maxsize=16)
def _local_semantic_task_vectors(
    *,
    model_name: str,
    local_files_only: bool,
    max_length: int,
) -> dict[str, tuple[float, ...]] | None:
    vectors: dict[str, tuple[float, ...]] = {}
    for task, phrases in SEMANTIC_TASK_PROTOTYPES.items():
        phrase_vectors: list[tuple[float, ...]] = []
        for phrase in phrases:
            embedded = _local_embedding_for_text(
                model_name=model_name,
                local_files_only=local_files_only,
                max_length=max_length,
                text=phrase.lower(),
            )
            if embedded is not None:
                phrase_vectors.append(embedded)
        task_vector = _mean_normalized_vector(phrase_vectors)
        if task_vector is None:
            return None
        vectors[task] = task_vector
    return vectors


def _semantic_task_prediction_local(
    *,
    text_lower: str,
    semantic_cfg: SemanticClassifierConfig,
) -> tuple[str, float, dict[str, float]] | None:
    if not bool(semantic_cfg.enabled):
        return None
    if semantic_cfg.backend != "local_embedding":
        return None

    model_name = str(semantic_cfg.local_model_name or "").strip()
    if not model_name:
        return None

    query_vector = _local_embedding_for_text(
        model_name=model_name,
        local_files_only=bool(semantic_cfg.local_files_only),
        max_length=max(16, int(semantic_cfg.local_max_length)),
        text=text_lower,
    )
    if query_vector is None:
        return None

    task_vectors = _local_semantic_task_vectors(
        model_name=model_name,
        local_files_only=bool(semantic_cfg.local_files_only),
        max_length=max(16, int(semantic_cfg.local_max_length)),
    )
    if task_vectors is None:
        return None

    scores: dict[str, float] = {
        "general": 0.05,
        "coding": 0.0,
        "thinking": 0.0,
        "instruction_following": 0.0,
    }
    for task, task_vector in task_vectors.items():
        cosine = _cosine_similarity(query_vector, task_vector)
        score = (cosine + 1.0) / 2.0
        scores[task] = max(scores.get(task, 0.0), float(score))

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_task, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = (top_score - second_score) / max(0.1, top_score)
    return top_task, max(0.0, confidence), scores


def _semantic_task_prediction(
    *,
    text_lower: str,
    semantic_cfg: SemanticClassifierConfig | None,
) -> tuple[str, float, dict[str, float], str, str | None]:
    if semantic_cfg is not None and semantic_cfg.enabled:
        local_prediction = _semantic_task_prediction_local(
            text_lower=text_lower,
            semantic_cfg=semantic_cfg,
        )
        if local_prediction is not None:
            task, confidence, scores = local_prediction
            return task, confidence, scores, "local_embedding", None
        task, confidence, scores = _semantic_task_prediction_prototype(text_lower)
        return (
            task,
            confidence,
            scores,
            "prototype",
            "local_embedding_unavailable",
        )

    task, confidence, scores = _semantic_task_prediction_prototype(text_lower)
    return task, confidence, scores, "prototype", None


def _collect_structural_code_signals(
    text_blob: str, text_lower: str
) -> tuple[list[str], float]:
    matches: list[str] = []
    score = 0.0

    if _STACK_TRACE_PATTERN.search(text_lower):
        matches.append("stack_trace")
        score += 0.9
    if _SOURCE_LOCATION_PATTERN.search(text_blob):
        matches.append("source_location")
        score += 0.6
    if _BUILD_OR_TEST_COMMAND_PATTERN.search(text_blob):
        matches.append("build_or_test_command")
        score += 0.6

    symbol_hits = len(_CODE_SYMBOL_PATTERN.findall(text_blob))
    if symbol_hits >= 6:
        matches.append("code_symbols")
        score += min(1.0, symbol_hits / 16.0)

    lines = [line for line in text_blob.splitlines() if line.strip()]
    if lines:
        code_like_lines = sum(1 for line in lines if _CODE_LINE_PATTERN.search(line))
        if code_like_lines >= 2:
            matches.append("code_like_lines")
            score += min(1.2, code_like_lines * 0.35)

    return matches, score


def _maybe_override_task_with_latest_user_turn(
    *,
    task: str,
    reason: str,
    confidence: float,
    latest_user_text: str,
    latest_user_text_lower: str,
    latest_user_structural_code_score: float,
) -> tuple[str, str, float, bool]:
    if task not in {"coding", "thinking"}:
        return task, reason, confidence, False
    if not latest_user_text:
        return task, reason, confidence, False

    latest_code_like = (
        "```" in latest_user_text
        or latest_user_structural_code_score > 0.0
        or _contains_any(latest_user_text_lower, CODE_HINTS)
        or _contains_any(latest_user_text_lower, CODING_OBJECT_HINTS)
    )
    latest_references_context = any(
        hint in latest_user_text_lower for hint in _LATEST_CONTEXT_REFERENCE_HINTS
    )
    latest_factual_query = bool(_LATEST_FACTUAL_QUERY_PATTERN.search(latest_user_text))

    if latest_factual_query and not latest_code_like and not latest_references_context:
        return "general", "latest_turn_factual_override", min(confidence, 0.35), True

    return task, reason, confidence, False


def _task_scores(
    *,
    text_blob: str,
    text_lower: str,
    text_length: int,
    code_score: int,
    think_score: int,
    instruction_score: int,
    structural_code_score: float,
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
        "coding": (float(code_score) * 1.2) + float(structural_code_score),
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
    structural_code_score: float,
    instruction_score: int,
    reasoning_effort: str | None,
    medium_max_chars: int,
    semantic_task: str,
    semantic_confidence: float,
    semantic_scores: dict[str, float],
    semantic_min_confidence: float,
    calibration_cfg: ClassifierCalibrationConfig,
) -> tuple[
    str,
    str,
    float,
    bool,
    dict[str, float],
    float | None,
    bool,
]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_task, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = max(0.0, top_score - second_score)
    reason = "score_top1"
    secondary_used = False
    secondary_scores: dict[str, float] = {}
    secondary_confidence: float | None = None
    semantic_used = False

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

    semantic_override_allowed = (
        semantic_task != "general"
        and semantic_confidence >= semantic_min_confidence
        and not (semantic_task == "instruction_following" and text_length > medium_max_chars)
    )
    if semantic_override_allowed and (
        confidence < 0.45
        or top_task == "general"
        or (
            top_task == "instruction_following"
            and semantic_task in {"coding", "thinking"}
            and text_length > medium_max_chars
        )
    ):
        semantic_top = semantic_scores.get(semantic_task, 0.0)
        current_top = semantic_scores.get(top_task, 0.0)
        if semantic_top >= current_top:
            top_task = semantic_task
            reason = "semantic_classifier_override"
            confidence = max(confidence, semantic_confidence)
            semantic_used = True

    secondary_trigger = confidence < 0.45 or _contains_any(
        text_lower, SECONDARY_INSTRUCTION_HINTS
    )
    if secondary_trigger:
        (
            secondary_task,
            secondary_confidence_value,
            secondary_scores,
        ) = _secondary_task_prediction(text_lower)
        if structural_code_score > 0.0:
            secondary_scores["coding"] = secondary_scores.get("coding", 0.0) + min(
                1.2, structural_code_score
            )
            ordered_secondary = sorted(
                secondary_scores.items(), key=lambda item: item[1], reverse=True
            )
            secondary_task = ordered_secondary[0][0]
            secondary_top = ordered_secondary[0][1]
            secondary_second = (
                ordered_secondary[1][1] if len(ordered_secondary) > 1 else 0.0
            )
            secondary_confidence_value = (secondary_top - secondary_second) / max(
                1.0, secondary_top
            )
        secondary_confidence = secondary_confidence_value
        min_secondary_confidence = (
            float(calibration_cfg.secondary_low_confidence_min_confidence)
            if confidence < 0.45
            else float(calibration_cfg.secondary_mixed_signal_min_confidence)
        )
        if (
            secondary_task != "general"
            and secondary_confidence_value >= min_secondary_confidence
            and not (
                secondary_task == "instruction_following"
                and text_length > medium_max_chars
            )
        ):
            top_task = secondary_task
            reason = "secondary_classifier_override"
            confidence = max(confidence, secondary_confidence_value)
            secondary_used = True

    return (
        top_task,
        reason,
        confidence,
        secondary_used,
        secondary_scores,
        secondary_confidence,
        semantic_used,
    )


def classify_request(
    payload: dict[str, Any],
    endpoint: str,
    complexity_cfg: ComplexityConfig,
    calibration_cfg: ClassifierCalibrationConfig | None = None,
    semantic_cfg: SemanticClassifierConfig | None = None,
) -> tuple[str, str, dict[str, Any]]:
    effective_calibration_cfg = calibration_cfg or ClassifierCalibrationConfig()
    signals: dict[str, Any] = {"has_image": False}
    (
        user_texts,
        message_texts,
        latest_user_text_from_messages,
        latest_conversation_text,
    ) = _collect_message_texts(payload, signals)
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
    matched_structural_code_hints, structural_code_score = (
        _collect_structural_code_signals(text_blob, text_lower)
    )
    latest_user_text = latest_user_text_from_messages or latest_conversation_text
    latest_user_text_lower = latest_user_text.lower()
    (
        latest_user_structural_code_hints,
        latest_user_structural_code_score,
    ) = _collect_structural_code_signals(latest_user_text, latest_user_text_lower)

    code_score = len(matched_code_hints) + int(round(structural_code_score))
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
        structural_code_score=structural_code_score,
        think_score=think_score,
        instruction_score=instruction_score,
        reasoning_effort=reasoning_effort,
        payload=payload,
    )
    task_confidence = 1.0
    effective_semantic_cfg = semantic_cfg or SemanticClassifierConfig()
    (
        semantic_task,
        semantic_task_confidence,
        semantic_task_scores,
        semantic_classifier_source,
        semantic_classifier_status,
    ) = _semantic_task_prediction(
        text_lower=text_lower,
        semantic_cfg=effective_semantic_cfg,
    )
    semantic_classifier_used = False
    secondary_classifier_used = False
    secondary_task_scores: dict[str, float] = {}
    secondary_task_confidence: float | None = None
    latest_turn_override_applied = False

    if endpoint.startswith("/v1/images") or signals["has_image"]:
        task = "image"
        task_reason = "image_endpoint_or_image_signal"
        task_scores["image"] = 1.0
    else:
        (
            task,
            task_reason,
            task_confidence,
            secondary_classifier_used,
            secondary_task_scores,
            secondary_task_confidence,
            semantic_classifier_used,
        ) = _select_task_from_scores(
            scores=task_scores,
            text_lower=text_lower,
            text_length=text_length,
            code_score=code_score,
            structural_code_score=structural_code_score,
            instruction_score=instruction_score,
            reasoning_effort=reasoning_effort,
            medium_max_chars=complexity_cfg.medium_max_chars,
            semantic_task=semantic_task,
            semantic_confidence=semantic_task_confidence,
            semantic_scores=semantic_task_scores,
            semantic_min_confidence=max(
                0.0, float(effective_semantic_cfg.min_confidence)
            ),
            calibration_cfg=effective_calibration_cfg,
        )
        (
            task,
            task_reason,
            task_confidence,
            latest_turn_override_applied,
        ) = _maybe_override_task_with_latest_user_turn(
            task=task,
            reason=task_reason,
            confidence=task_confidence,
            latest_user_text=latest_user_text,
            latest_user_text_lower=latest_user_text_lower,
            latest_user_structural_code_score=latest_user_structural_code_score,
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
            "semantic_task_prediction": semantic_task,
            "semantic_classifier_used": semantic_classifier_used,
            "semantic_classifier_source": semantic_classifier_source,
            "semantic_classifier_status": semantic_classifier_status,
            "semantic_task_confidence": round(semantic_task_confidence, 6),
            "semantic_task_scores": {
                name: round(score, 6)
                for name, score in sorted(semantic_task_scores.items())
            },
            "secondary_classifier_used": secondary_classifier_used,
            "secondary_min_confidence_low": float(
                effective_calibration_cfg.secondary_low_confidence_min_confidence
            ),
            "secondary_min_confidence_mixed": float(
                effective_calibration_cfg.secondary_mixed_signal_min_confidence
            ),
            "secondary_task_confidence": (
                None
                if secondary_task_confidence is None
                else round(secondary_task_confidence, 6)
            ),
            "secondary_task_scores": {
                name: round(score, 6)
                for name, score in sorted(secondary_task_scores.items())
            },
            "latest_turn_override_applied": latest_turn_override_applied,
            "latest_user_text_length": len(latest_user_text),
            "latest_user_factual_query": bool(
                _LATEST_FACTUAL_QUERY_PATTERN.search(latest_user_text)
            ),
            "latest_user_references_context": any(
                hint in latest_user_text_lower for hint in _LATEST_CONTEXT_REFERENCE_HINTS
            ),
            "latest_user_structural_code_hints": latest_user_structural_code_hints,
            "latest_user_structural_code_score": round(
                latest_user_structural_code_score, 6
            ),
            "complexity_base": complexity_base,
            "complexity_adjustments": complexity_adjustments,
            "complexity_final": complexity,
            "matched_code_hints": matched_code_hints,
            "matched_thinking_hints": matched_thinking_hints,
            "matched_instruction_hints": matched_instruction_hints,
            "matched_structural_code_hints": matched_structural_code_hints,
            "structural_code_score": round(structural_code_score, 6),
            "text_preview": text_blob[:500],
            "text_preview_total": full_text_blob[:500],
            "endpoint": endpoint,
        }
    )
    return task, complexity, signals
