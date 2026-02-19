from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, cast

import httpx
from fastapi import status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import Headers

from open_llm_router.circuit_breaker import CircuitBreakerRegistry
from open_llm_router.config import BackendAccount
from open_llm_router.persistence import YamlFileStore
from open_llm_router.router_engine import RouteDecision

HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

logger = logging.getLogger("uvicorn.error")


def _request_error_details(exc: httpx.RequestError) -> dict[str, Any]:
    error_repr = repr(exc)
    error_message = str(exc).strip() or error_repr
    error_type = exc.__class__.__name__.strip() or "RequestError"
    details: dict[str, Any] = {
        "error": error_message,
        "error_type": error_type,
        "error_repr": error_repr,
        "is_timeout": isinstance(exc, httpx.TimeoutException),
        "status_code": None,
    }
    request = getattr(exc, "request", None)
    if isinstance(request, httpx.Request):
        details["request_method"] = request.method
        details["request_url"] = str(request.url)
    response = getattr(exc, "response", None)
    if isinstance(response, httpx.Response):
        details["status_code"] = response.status_code
    return details


@dataclass(slots=True)
class BackendTarget:
    account: BackendAccount
    account_name: str
    provider: str
    base_url: str
    model: str
    auth_mode: str
    organization: str | None
    project: str | None
    upstream_model: str
    metadata: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        if "/" in self.model:
            provider, _, model_id = self.model.partition("/")
            if (
                provider.strip().lower() == self.provider.strip().lower()
                and self.upstream_model == model_id.strip()
            ):
                return f"{self.account_name}:{model_id.strip()}"
        return f"{self.account_name}:{self.model}"


@dataclass(slots=True)
class OAuthRuntimeState:
    access_token: str
    refresh_token: str | None
    expires_at: int | None
    account_id: str | None = None


@dataclass(slots=True)
class UpstreamRequestSpec:
    path: str
    payload: dict[str, Any]
    stream: bool
    adapter: Literal["passthrough", "chat_completions"] = "passthrough"


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() not in HOP_BY_HOP_RESPONSE_HEADERS:
            filtered[name] = value
    return filtered


def _response_content_type(response_headers: dict[str, str]) -> str | None:
    for name, value in response_headers.items():
        if name.lower() == "content-type":
            return value
    return None


def _is_json_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _build_routing_diagnostics(
    *,
    request_id: str,
    target: BackendTarget,
    route_decision: RouteDecision,
    request_latency_ms: float,
    attempted_targets: list[str],
    attempted_upstream_models: list[str],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "request_id": request_id,
        "selected_model": target.model,
        "upstream_model": target.upstream_model,
        "provider": target.provider,
        "account": target.account_name,
        "auth_mode": target.auth_mode,
        "task": route_decision.task,
        "complexity": route_decision.complexity,
        "source": route_decision.source,
        "request_latency_ms": round(request_latency_ms, 3),
        "attempted_targets": list(attempted_targets),
        "attempted_upstream_models": list(attempted_upstream_models),
    }
    if route_decision.ranked_models:
        diagnostics["ranked_models"] = list(route_decision.ranked_models)
    if route_decision.candidate_scores:
        top = route_decision.candidate_scores[0]
        diagnostics["top_candidate_model"] = top.get("model")
        diagnostics["top_utility"] = top.get("utility")
    if route_decision.provider_preferences:
        diagnostics["provider_preferences"] = dict(route_decision.provider_preferences)
    return diagnostics


def _json_response_with_routing_diagnostics(
    *,
    status_code: int,
    body: bytes,
    response_headers: dict[str, str],
    routing_diagnostics: dict[str, Any],
) -> JSONResponse | None:
    if not _is_json_content_type(_response_content_type(response_headers)):
        return None
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed["_router"] = routing_diagnostics
    for name in list(response_headers.keys()):
        if name.lower() == "content-type":
            response_headers.pop(name, None)
    return JSONResponse(
        status_code=status_code, content=parsed, headers=response_headers
    )


def _build_upstream_headers(
    incoming_headers: Headers,
    bearer_token: str | None,
    provider: str,
    oauth_account_id: str | None,
    organization: str | None,
    project: str | None,
    stream: bool = False,
    allow_passthrough_auth: bool = False,
) -> dict[str, str]:
    passthrough = {
        "accept",
        "baggage",
        "content-type",
        "idempotency-key",
        "openai-organization",
        "openai-project",
        "traceparent",
        "tracestate",
    }
    safe_x_headers = {
        "x-openai-client-source",
        "x-openai-client-user-agent",
        "x-request-id",
        "x-stainless-arch",
        "x-stainless-async",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-read-timeout",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
    }
    canonical_names = {
        "accept": "Accept",
        "baggage": "baggage",
        "content-type": "Content-Type",
        "idempotency-key": "Idempotency-Key",
        "openai-organization": "OpenAI-Organization",
        "openai-project": "OpenAI-Project",
        "traceparent": "traceparent",
        "tracestate": "tracestate",
    }
    headers: dict[str, str] = {}
    for name, value in incoming_headers.items():
        lower = name.lower()
        if lower in {"host", "content-length", "connection", "authorization"}:
            continue
        if lower in passthrough or lower in safe_x_headers:
            key = canonical_names.get(lower, name)
            headers[key] = value

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif allow_passthrough_auth:
        incoming_auth = incoming_headers.get("authorization", "").strip()
        if incoming_auth.lower().startswith("bearer "):
            headers["Authorization"] = incoming_auth

    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project

    if provider.strip().lower() == "openai-codex":
        if oauth_account_id:
            headers["chatgpt-account-id"] = oauth_account_id
        headers.setdefault("OpenAI-Beta", "responses=experimental")
        headers.setdefault("originator", "pi")
        headers["Accept"] = "text/event-stream"

    if not any(key.lower() == "content-type" for key in headers):
        headers["Content-Type"] = "application/json"
    if not any(key.lower() == "accept" for key in headers):
        headers["Accept"] = "text/event-stream" if stream else "application/json"
    return headers


def _as_input_text_part(value: str) -> dict[str, str]:
    return {"type": "input_text", "text": value}


def _as_output_text_part(value: str) -> dict[str, str]:
    return {"type": "output_text", "text": value}


def _normalize_codex_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized == "tool":
        return "user"
    if normalized not in {"assistant", "system", "developer", "user"}:
        return "user"
    return normalized


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")).strip()
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return _coerce_text(content)

    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                chunks.append(item)
            continue
        if not isinstance(item, dict):
            continue
        raw_text = item.get("text")
        if isinstance(raw_text, str) and raw_text.strip():
            chunks.append(raw_text)
            continue
        if item.get("type") == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url.strip():
                chunks.append(image_url)
    text = "\n".join(chunks).strip()
    return text if text else _coerce_text(content)


def _to_codex_input_parts(content: Any, role: str) -> list[dict[str, Any]]:
    normalized_role = _normalize_codex_role(role)

    def make_text_part(value: str) -> dict[str, str]:
        if normalized_role == "assistant":
            return _as_output_text_part(value)
        return _as_input_text_part(value)

    if isinstance(content, str):
        text = content.strip()
        return [make_text_part(text)] if text else []

    if not isinstance(content, list):
        text = _coerce_text(content)
        return [make_text_part(text)] if text else []

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(make_text_part(text))
            continue

        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).strip().lower()
        if item_type in {"text", "input_text", "output_text"}:
            raw_text = item.get("text")
            if isinstance(raw_text, str):
                text = raw_text.strip()
                if text:
                    parts.append(make_text_part(text))
            continue

        if item_type in {"image_url", "input_image"}:
            # Codex accepts input_image for user/system/developer, not assistant output.
            if normalized_role == "assistant":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url.strip():
                parts.append({"type": "input_image", "image_url": image_url.strip()})
            continue

        fallback_text = _coerce_text(item.get("content"))
        if fallback_text:
            parts.append(make_text_part(fallback_text))

    if not parts:
        fallback = _coerce_text(content)
        if fallback:
            parts.append(make_text_part(fallback))
    return parts


def _extract_codex_instructions(
    payload: dict[str, Any], messages: list[dict[str, Any]]
) -> str:
    explicit = payload.get("instructions")
    instructions: list[str] = []
    if isinstance(explicit, str) and explicit.strip():
        instructions.append(explicit.strip())

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        if role not in {"system", "developer"}:
            continue
        text = _extract_text_content(message.get("content"))
        if text:
            instructions.append(text)

    return "\n\n".join(instructions) if instructions else "You are a helpful assistant."


def _chat_messages_to_codex_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        raw_role = str(message.get("role", "user")).strip().lower()
        role = _normalize_codex_role(raw_role)
        if raw_role in {"system", "developer"}:
            continue
        parts = _to_codex_input_parts(message.get("content"), role=role)
        if raw_role == "tool":
            for part in parts:
                if part.get("type") == "input_text":
                    text = str(part.get("text", "")).strip()
                    if text and not text.startswith("Tool output:"):
                        part["text"] = f"Tool output:\n{text}"
        if parts:
            result.append({"role": role, "content": parts})
    return result


def _normalize_responses_input_for_codex(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        text = input_value.strip()
        return (
            [{"role": "user", "content": [_as_input_text_part(text)]}] if text else []
        )

    if isinstance(input_value, dict):
        raw_role = str(input_value.get("role", "user")).strip().lower() or "user"
        role = _normalize_codex_role(raw_role)
        content = _to_codex_input_parts(input_value.get("content"), role=role)
        if not content:
            return []
        if raw_role == "tool":
            for part in content:
                if part.get("type") == "input_text":
                    text = str(part.get("text", "")).strip()
                    if text and not text.startswith("Tool output:"):
                        part["text"] = f"Tool output:\n{text}"
        return [{"role": role, "content": content}]

    if isinstance(input_value, list):
        output: list[dict[str, Any]] = []
        for item in input_value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    output.append(
                        {"role": "user", "content": [_as_input_text_part(text)]}
                    )
                continue
            if isinstance(item, dict):
                if "role" in item:
                    raw_role = str(item.get("role", "user")).strip().lower() or "user"
                    role = _normalize_codex_role(raw_role)
                    parts = _to_codex_input_parts(item.get("content"), role=role)
                    if raw_role == "tool":
                        for part in parts:
                            if part.get("type") == "input_text":
                                text = str(part.get("text", "")).strip()
                                if text and not text.startswith("Tool output:"):
                                    part["text"] = f"Tool output:\n{text}"
                    if parts:
                        output.append({"role": role, "content": parts})
                else:
                    parts = _to_codex_input_parts([item], role="user")
                    if parts:
                        output.append({"role": "user", "content": parts})
        return output

    return []


def _normalize_codex_tools(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools

    normalized: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue

        tool_type = str(tool.get("type", "")).strip().lower()
        if tool_type == "function" and isinstance(tool.get("function"), dict):
            function_def = tool["function"]
            mapped: dict[str, Any] = {
                "type": "function",
                "name": function_def.get("name"),
            }
            if "description" in function_def:
                mapped["description"] = function_def.get("description")
            if "parameters" in function_def:
                mapped["parameters"] = function_def.get("parameters")
            if "strict" in function_def:
                mapped["strict"] = function_def.get("strict")
            normalized.append(mapped)
            continue

        normalized.append(tool)

    return normalized


def _normalize_codex_tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    if not isinstance(choice, dict):
        return choice

    choice_type = str(choice.get("type", "")).strip().lower()
    if choice_type != "function":
        return choice

    if "name" in choice:
        return choice

    function_obj = choice.get("function")
    if isinstance(function_obj, dict):
        name = function_obj.get("name")
        if isinstance(name, str) and name.strip():
            return {"type": "function", "name": name.strip()}
    return choice


def _prepare_codex_payload(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model", "")).strip()
    messages = [m for m in payload.get("messages", []) if isinstance(m, dict)]
    codex_payload: dict[str, Any] = {
        "model": model,
        "store": False,
        # ChatGPT Codex backend currently requires stream=true.
        "stream": True,
        "instructions": _extract_codex_instructions(payload, messages),
    }

    if path == "/v1/chat/completions":
        input_messages = _chat_messages_to_codex_input(messages)
        if not input_messages:
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                input_messages = [
                    {"role": "user", "content": [_as_input_text_part(prompt.strip())]}
                ]
        codex_payload["input"] = input_messages
    else:
        if "input" in payload:
            codex_payload["input"] = _normalize_responses_input_for_codex(
                payload.get("input")
            )
        elif messages:
            codex_payload["input"] = _chat_messages_to_codex_input(messages)
        else:
            codex_payload["input"] = []

    passthrough_fields = {
        "temperature",
        "top_p",
        "max_output_tokens",
        "reasoning",
        "metadata",
        "parallel_tool_calls",
    }
    for field in passthrough_fields:
        if field in payload:
            codex_payload[field] = payload[field]

    if "tools" in payload:
        codex_payload["tools"] = _normalize_codex_tools(payload.get("tools"))
    elif "functions" in payload:
        # Legacy Chat Completions function-calling schema.
        legacy_functions = payload.get("functions")
        if isinstance(legacy_functions, list):
            codex_payload["tools"] = [
                {"type": "function", **function_def}
                for function_def in legacy_functions
                if isinstance(function_def, dict)
            ]

    if "tool_choice" in payload:
        codex_payload["tool_choice"] = _normalize_codex_tool_choice(
            payload.get("tool_choice")
        )
    elif "function_call" in payload:
        function_call = payload.get("function_call")
        if isinstance(function_call, str):
            codex_payload["tool_choice"] = function_call
        elif isinstance(function_call, dict):
            name = function_call.get("name")
            if isinstance(name, str) and name.strip():
                codex_payload["tool_choice"] = {
                    "type": "function",
                    "name": name.strip(),
                }

    max_tokens = payload.get("max_tokens")
    if max_tokens is not None and "max_output_tokens" not in codex_payload:
        codex_payload["max_output_tokens"] = max_tokens

    return codex_payload


def _drop_none_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned_item = _drop_none_fields(item)
            if cleaned_item is None:
                continue
            cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [_drop_none_fields(item) for item in value if item is not None]
    return value


def _prepare_gemini_chat_payload(
    payload: dict[str, Any], stream: bool
) -> dict[str, Any]:
    allowed_fields = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    }
    prepared: dict[str, Any] = {
        "model": str(payload.get("model", "")).strip(),
        "stream": stream,
    }
    for field in allowed_fields:
        if field in {"model", "stream"}:
            continue
        if field not in payload:
            continue
        value = payload.get(field)
        if value is None:
            continue
        prepared[field] = value

    # Normalize max_output_tokens to OpenAI chat-completions max_tokens for Gemini compatibility.
    max_output_tokens = payload.get("max_output_tokens")
    if "max_tokens" not in prepared and isinstance(max_output_tokens, int):
        prepared["max_tokens"] = max_output_tokens

    return cast(dict[str, Any], _drop_none_fields(prepared))


def _prepare_upstream_request(
    path: str,
    payload: dict[str, Any],
    provider: str,
    stream: bool,
) -> UpstreamRequestSpec:
    if provider.strip().lower() == "openai-codex" and path in {
        "/v1/chat/completions",
        "/v1/responses",
    }:
        adapter: Literal["passthrough", "chat_completions"] = (
            "chat_completions" if path == "/v1/chat/completions" else "passthrough"
        )
        return UpstreamRequestSpec(
            path="/codex/responses",
            payload=_prepare_codex_payload(path, payload),
            stream=True,
            adapter=adapter,
        )

    if provider.strip().lower() == "gemini" and path == "/v1/chat/completions":
        return UpstreamRequestSpec(
            path=path,
            payload=_prepare_gemini_chat_payload(payload=payload, stream=stream),
            stream=stream,
        )

    if provider.strip().lower() == "github" and path == "/v1/chat/completions":
        return UpstreamRequestSpec(
            path="/inference/chat/completions",
            payload=_prepare_gemini_chat_payload(payload=payload, stream=stream),
            stream=stream,
        )

    return UpstreamRequestSpec(path=path, payload=payload, stream=stream)


@dataclass(slots=True)
class ProxyExecutionStats:
    attempted_targets: list[str]
    attempted_upstream_models: list[str]
    skipped_rate_limited: int = 0
    skipped_circuit_open: int = 0
    skipped_oauth_token_missing: int = 0


@dataclass(slots=True)
class ProxyAttemptRunContext:
    path: str
    trial_payload: dict[str, Any]
    incoming_headers: Headers
    route_decision: RouteDecision
    stream: bool
    request_id: str
    total_attempts: int
    request_started: float


@dataclass(slots=True)
class PreparedProxyAttempt:
    request_spec: UpstreamRequestSpec
    headers: dict[str, str]


@dataclass(slots=True)
class ProxyExecutionPlan:
    candidate_targets: list[BackendTarget]
    context: ProxyAttemptRunContext | None
    preflight_response: Response | None = None


class ProxyResponseAdapter:
    @staticmethod
    async def to_fastapi_response(
        *,
        upstream: httpx.Response,
        stream: bool,
        upstream_stream: bool,
        target: BackendTarget,
        attempted_targets: list[str],
        attempted_upstream_models: list[str],
        request_latency_ms: float,
        route_decision: RouteDecision,
        request_id: str,
        adapter: Literal["passthrough", "chat_completions"],
        audit_hook: Callable[[dict[str, Any]], None] | None,
    ) -> Response:
        response_headers = _filter_response_headers(upstream.headers)
        response_headers["x-router-request-id"] = request_id
        response_headers["x-router-model"] = target.model
        response_headers["x-router-account"] = target.account_name
        response_headers["x-router-provider"] = target.provider
        response_headers["x-router-auth-mode"] = target.auth_mode
        response_headers["x-router-upstream-model"] = target.upstream_model
        response_headers["x-router-task"] = route_decision.task
        response_headers["x-router-complexity"] = route_decision.complexity
        response_headers["x-router-source"] = route_decision.source
        response_headers["x-router-request-latency-ms"] = f"{request_latency_ms:.3f}"
        response_headers["x-router-attempted-targets"] = ",".join(attempted_targets)
        if route_decision.ranked_models:
            response_headers["x-router-ranked-models"] = ",".join(
                route_decision.ranked_models
            )
        if route_decision.candidate_scores:
            top = route_decision.candidate_scores[0]
            response_headers["x-router-top-utility"] = str(top.get("utility", ""))
        routing_diagnostics = _build_routing_diagnostics(
            request_id=request_id,
            target=target,
            route_decision=route_decision,
            request_latency_ms=request_latency_ms,
            attempted_targets=attempted_targets,
            attempted_upstream_models=attempted_upstream_models,
        )

        logger.info(
            "proxy_response request_id=%s target=%s status=%d attempts=%d",
            request_id,
            target.label,
            upstream.status_code,
            len(attempted_targets),
        )
        if audit_hook is not None:
            try:
                audit_hook(
                    {
                        "event": "proxy_response",
                        "request_id": request_id,
                        "target": target.label,
                        "account": target.account_name,
                        "provider": target.provider,
                        "model": target.model,
                        "upstream_model": target.upstream_model,
                        "status": upstream.status_code,
                        "request_latency_ms": request_latency_ms,
                        "attempts": len(attempted_targets),
                        "attempted_targets": attempted_targets,
                        "attempted_upstream_models": attempted_upstream_models,
                    }
                )
            except Exception:
                pass

        if adapter == "chat_completions":
            return await BackendProxy._to_chat_completions_response(
                upstream=upstream,
                stream=stream,
                response_headers=response_headers,
                model=target.model,
                request_id=request_id,
                audit_hook=audit_hook,
                routing_diagnostics=routing_diagnostics,
            )

        if stream:
            media_type = response_headers.pop("content-type", "text/event-stream")

            async def stream_generator() -> AsyncIterator[bytes]:
                try:
                    if upstream_stream:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                    else:
                        body = await upstream.aread()
                        if body:
                            yield body
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                content=stream_generator(),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=media_type,
            )

        body = await upstream.aread()
        await upstream.aclose()
        diagnostics_response = _json_response_with_routing_diagnostics(
            status_code=upstream.status_code,
            body=body,
            response_headers=response_headers,
            routing_diagnostics=routing_diagnostics,
        )
        if diagnostics_response is not None:
            return diagnostics_response
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers=response_headers,
        )


class ProxyAttemptProcessor:
    def __init__(self, *, proxy: BackendProxy) -> None:
        self._proxy = proxy

    def routing_exhausted_response(
        self,
        *,
        request_id: str,
        candidate_targets_total: int,
        stats: ProxyExecutionStats,
    ) -> JSONResponse:
        logger.error(
            (
                "proxy_exhausted request_id=%s attempted_targets=%s "
                "candidate_targets_total=%d skipped_rate_limited=%d "
                "skipped_circuit_open=%d skipped_oauth_token_missing=%d"
            ),
            request_id,
            ",".join(stats.attempted_targets),
            candidate_targets_total,
            stats.skipped_rate_limited,
            stats.skipped_circuit_open,
            stats.skipped_oauth_token_missing,
        )
        self._proxy.audit(
            "proxy_exhausted",
            request_id=request_id,
            candidate_targets_total=candidate_targets_total,
            attempted_count=len(stats.attempted_targets),
            attempted_targets=stats.attempted_targets,
            attempted_upstream_models=stats.attempted_upstream_models,
            skipped_rate_limited=stats.skipped_rate_limited,
            skipped_circuit_open=stats.skipped_circuit_open,
            skipped_oauth_token_missing=stats.skipped_oauth_token_missing,
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": {
                    "type": "routing_exhausted",
                    "message": "All model/account targets failed.",
                    "candidate_targets_total": candidate_targets_total,
                    "attempted_count": len(stats.attempted_targets),
                    "attempted_targets": stats.attempted_targets,
                    "attempted_upstream_models": stats.attempted_upstream_models,
                    "skipped_rate_limited": stats.skipped_rate_limited,
                    "skipped_circuit_open": stats.skipped_circuit_open,
                    "skipped_oauth_token_missing": stats.skipped_oauth_token_missing,
                }
            },
        )

    def skip_target_for_circuit_breaker(
        self,
        *,
        target: BackendTarget,
        breaker_key: str,
        request_id: str,
        stats: ProxyExecutionStats,
    ) -> bool:
        if self._proxy.circuit_breakers is None:
            return False
        if self._proxy.circuit_breakers.allow_request(breaker_key):
            return False

        snapshot = self._proxy.circuit_breakers.snapshot(breaker_key)
        logger.info(
            "proxy_skip_circuit_open request_id=%s target=%s state=%s failures=%s",
            request_id,
            target.label,
            snapshot.get("state"),
            snapshot.get("failure_count"),
        )
        self._proxy.audit(
            "proxy_skip_circuit_open",
            request_id=request_id,
            target=target.label,
            account=target.account_name,
            model=target.model,
            breaker=snapshot,
        )
        stats.skipped_circuit_open += 1
        return True

    def skip_target_for_rate_limit(
        self,
        *,
        target: BackendTarget,
        request_id: str,
        stats: ProxyExecutionStats,
    ) -> bool:
        if not self._proxy.is_temporarily_rate_limited(target.account_name):
            return False

        logger.info(
            "proxy_skip_rate_limited request_id=%s target=%s",
            request_id,
            target.label,
        )
        self._proxy.audit(
            "proxy_skip_rate_limited",
            request_id=request_id,
            target=target.label,
            account=target.account_name,
            model=target.model,
        )
        stats.skipped_rate_limited += 1
        return True

    def record_attempt(
        self,
        *,
        target: BackendTarget,
        request_id: str,
        attempt: int,
        total_attempts: int,
        stats: ProxyExecutionStats,
    ) -> None:
        stats.attempted_targets.append(target.label)
        stats.attempted_upstream_models.append(target.upstream_model)
        logger.info(
            "proxy_attempt request_id=%s attempt=%d/%d target=%s provider=%s auth_mode=%s",
            request_id,
            attempt,
            total_attempts,
            target.label,
            target.provider,
            target.auth_mode,
        )
        self._proxy.audit(
            "proxy_attempt",
            request_id=request_id,
            attempt=attempt,
            total_attempts=total_attempts,
            target=target.label,
            account=target.account_name,
            provider=target.provider,
            model=target.model,
            upstream_model=target.upstream_model,
            auth_mode=target.auth_mode,
        )

    def handle_missing_oauth_token(
        self,
        *,
        target: BackendTarget,
        bearer_token: str | None,
        request_id: str,
        has_more_targets: bool,
        stats: ProxyExecutionStats,
    ) -> tuple[bool, JSONResponse | None]:
        if target.auth_mode != "oauth" or bearer_token:
            return False, None

        logger.warning(
            "proxy_oauth_token_missing request_id=%s target=%s",
            request_id,
            target.label,
        )
        self._proxy.audit(
            "proxy_oauth_token_missing",
            request_id=request_id,
            target=target.label,
            account=target.account_name,
            model=target.model,
            upstream_model=target.upstream_model,
        )
        stats.skipped_oauth_token_missing += 1
        if has_more_targets:
            return True, None

        return False, JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "type": "oauth_credentials_unavailable",
                    "message": (
                        "Selected OAuth-backed target has no usable access token "
                        "and could not refresh one."
                    ),
                    "attempted_targets": stats.attempted_targets,
                    "attempted_upstream_models": stats.attempted_upstream_models,
                }
            },
        )

    async def send_upstream_request(
        self,
        *,
        target: BackendTarget,
        request_spec: UpstreamRequestSpec,
        headers: dict[str, str],
        breaker_key: str,
        request_id: str,
        attempt: int,
        total_attempts: int,
        has_more_targets: bool,
        stats: ProxyExecutionStats,
    ) -> tuple[httpx.Response | None, JSONResponse | None]:
        attempt_started = time.perf_counter()
        try:
            request = self._proxy.client.build_request(
                method="POST",
                url=f"{target.base_url.rstrip('/')}{request_spec.path}",
                json=request_spec.payload,
                headers=headers,
            )
            upstream = await self._proxy.client.send(request, stream=request_spec.stream)
            connect_latency_ms = (time.perf_counter() - attempt_started) * 1000.0
            logger.info(
                "proxy_upstream_connected request_id=%s target=%s connect_ms=%.2f status=%d",
                request_id,
                target.label,
                connect_latency_ms,
                upstream.status_code,
            )
            self._proxy.audit(
                "proxy_upstream_connected",
                request_id=request_id,
                target=target.label,
                account=target.account_name,
                provider=target.provider,
                model=target.model,
                upstream_model=target.upstream_model,
                connect_ms=round(connect_latency_ms, 3),
                status=upstream.status_code,
            )
            return upstream, None
        except httpx.RequestError as exc:
            error_details = _request_error_details(exc)
            if self._proxy.circuit_breakers is not None:
                self._proxy.circuit_breakers.on_failure(breaker_key)
            logger.warning(
                (
                    "proxy_request_error request_id=%s target=%s "
                    "attempt=%d/%d error_type=%s status_code=%s error=%s"
                ),
                request_id,
                target.label,
                attempt,
                total_attempts,
                error_details["error_type"],
                error_details.get("status_code"),
                error_details["error"],
            )
            self._proxy.audit(
                "proxy_request_error",
                request_id=request_id,
                target=target.label,
                account=target.account_name,
                model=target.model,
                upstream_model=target.upstream_model,
                attempt=attempt,
                total_attempts=total_attempts,
                attempt_latency_ms=round(
                    (time.perf_counter() - attempt_started) * 1000.0, 3
                ),
                **error_details,
            )
            if has_more_targets:
                return None, None

            return None, JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={
                    "error": {
                        "type": "upstream_connection_error",
                        "message": (
                            "Could not reach backend "
                            f"({error_details['error_type']}): {error_details['error']}"
                        ),
                        "error_type": error_details["error_type"],
                        "attempted_targets": stats.attempted_targets,
                        "attempted_upstream_models": stats.attempted_upstream_models,
                    }
                },
            )

    def should_retry_upstream_response(
        self,
        *,
        upstream: httpx.Response,
        target: BackendTarget,
        breaker_key: str,
        request_id: str,
        has_more_targets: bool,
    ) -> bool:
        should_retry = (
            upstream.status_code in self._proxy.retry_statuses and has_more_targets
        )
        if upstream.status_code < 500 and upstream.status_code != 429:
            if self._proxy.circuit_breakers is not None:
                self._proxy.circuit_breakers.on_success(breaker_key)
        elif (
            upstream.status_code in self._proxy.retry_statuses
            and self._proxy.circuit_breakers is not None
        ):
            self._proxy.circuit_breakers.on_failure(breaker_key)

        if upstream.status_code == 429:
            self._proxy.mark_rate_limited(
                target.account_name,
                upstream.headers,
                request_id=request_id,
                target=target.label,
                model=target.model,
                upstream_model=target.upstream_model,
            )
        if not should_retry:
            return False

        logger.info(
            "proxy_retry request_id=%s target=%s status=%d",
            request_id,
            target.label,
            upstream.status_code,
        )
        self._proxy.audit(
            "proxy_retry",
            request_id=request_id,
            target=target.label,
            account=target.account_name,
            model=target.model,
            upstream_model=target.upstream_model,
            status=upstream.status_code,
        )
        return True


class ProxyPreflightPlanner:
    def __init__(self, *, proxy: BackendProxy) -> None:
        self._proxy = proxy

    def prepare_candidate_targets(
        self,
        *,
        route_decision: RouteDecision,
        payload: dict[str, Any],
    ) -> tuple[list[BackendTarget], dict[str, list[str]], list[str]]:
        candidate_targets = self._proxy.build_candidate_targets(route_decision)
        parameter_rejections: dict[str, list[str]] = {}
        requested_parameters: list[str] = []
        if self._proxy.requires_parameter_compatibility(
            route_decision.provider_preferences
        ):
            (
                candidate_targets,
                parameter_rejections,
                requested_parameters,
            ) = self._proxy.filter_targets_by_parameter_support(
                candidate_targets=candidate_targets,
                payload=payload,
            )
        return candidate_targets, parameter_rejections, requested_parameters

    def apply_fallback_policy(
        self,
        *,
        candidate_targets: list[BackendTarget],
        route_decision: RouteDecision,
        request_id: str,
    ) -> list[BackendTarget]:
        allow_fallbacks = self._proxy.allows_fallbacks(
            route_decision.provider_preferences
        )
        if allow_fallbacks or len(candidate_targets) <= 1:
            return candidate_targets

        primary_target = candidate_targets[0]
        self._proxy.audit(
            "proxy_fallbacks_disabled",
            request_id=request_id,
            selected_model=primary_target.model,
            primary_target=primary_target.label,
        )
        return [primary_target]

    @staticmethod
    def effective_model_selection(
        *,
        candidate_targets: list[BackendTarget],
        route_decision: RouteDecision,
    ) -> tuple[str, list[str]]:
        effective_model_chain = _dedupe_preserving_order(
            [target.model for target in candidate_targets]
        )
        effective_selected_model = (
            effective_model_chain[0]
            if effective_model_chain
            else route_decision.selected_model
        )
        effective_fallback_models = (
            effective_model_chain[1:] if effective_model_chain else []
        )
        return effective_selected_model, effective_fallback_models

    def emit_proxy_start(
        self,
        *,
        path: str,
        route_decision: RouteDecision,
        stream: bool,
        request_id: str,
        selected_model: str,
        fallback_models: list[str],
        candidate_targets: int,
    ) -> None:
        logger.info(
            (
                "proxy_start request_id=%s path=%s selected_model=%s stream=%s "
                "candidate_targets=%d"
            ),
            request_id,
            path,
            selected_model,
            stream,
            candidate_targets,
        )
        self._proxy.audit(
            "proxy_start",
            request_id=request_id,
            path=path,
            selected_model=selected_model,
            source=route_decision.source,
            task=route_decision.task,
            complexity=route_decision.complexity,
            requested_model=route_decision.requested_model,
            fallback_models=fallback_models,
            provider_preferences=route_decision.provider_preferences,
            stream=stream,
            candidate_targets=candidate_targets,
        )

    def emit_parameter_filter_audit(
        self,
        *,
        request_id: str,
        requested_parameters: list[str],
        parameter_rejections: dict[str, list[str]],
    ) -> None:
        self._proxy.audit(
            "proxy_parameter_compatibility_filter",
            request_id=request_id,
            requested_parameters=requested_parameters,
            rejections=parameter_rejections,
        )

    def preflight_constraint_response(
        self,
        *,
        route_decision: RouteDecision,
        candidate_targets: list[BackendTarget],
        requested_parameters: list[str],
        parameter_rejections: dict[str, list[str]],
        request_id: str,
    ) -> JSONResponse | None:
        if candidate_targets:
            return None

        if self._proxy.requires_parameter_compatibility(
            route_decision.provider_preferences
        ):
            return self._require_parameters_unsatisfied_response(
                request_id=request_id,
                requested_parameters=requested_parameters,
                parameter_rejections=parameter_rejections,
            )

        if self._proxy.has_provider_filters(route_decision.provider_preferences):
            return self._provider_filters_unsatisfied_response(
                route_decision=route_decision,
                request_id=request_id,
            )

        return self._no_backend_target_response(
            route_decision=route_decision,
            request_id=request_id,
        )

    def _require_parameters_unsatisfied_response(
        self,
        *,
        request_id: str,
        requested_parameters: list[str],
        parameter_rejections: dict[str, list[str]],
    ) -> JSONResponse:
        logger.warning(
            "proxy_require_parameters_no_target request_id=%s requested_parameters=%s",
            request_id,
            ",".join(requested_parameters),
        )
        self._proxy.audit(
            "proxy_require_parameters_no_target",
            request_id=request_id,
            requested_parameters=requested_parameters,
            rejections=parameter_rejections,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "type": "routing_constraints_unsatisfied",
                    "message": (
                        "No provider/account targets satisfy require_parameters constraints."
                    ),
                    "constraint": "require_parameters",
                    "details": {
                        "requested_parameters": requested_parameters,
                        "rejections": parameter_rejections,
                    },
                }
            },
        )

    def _provider_filters_unsatisfied_response(
        self,
        *,
        route_decision: RouteDecision,
        request_id: str,
    ) -> JSONResponse:
        only = self._proxy.normalized_provider_filter_values(
            route_decision.provider_preferences.get("only")
        )
        ignore = self._proxy.normalized_provider_filter_values(
            route_decision.provider_preferences.get("ignore")
        )
        logger.warning(
            "proxy_provider_filters_no_target request_id=%s only=%s ignore=%s",
            request_id,
            ",".join(only),
            ",".join(ignore),
        )
        self._proxy.audit(
            "proxy_provider_filters_no_target",
            request_id=request_id,
            selected_model=route_decision.selected_model,
            fallback_models=route_decision.fallback_models,
            only=only,
            ignore=ignore,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "type": "routing_constraints_unsatisfied",
                    "message": (
                        "No provider/account targets satisfy provider only/ignore constraints."
                    ),
                    "constraint": "provider",
                    "details": {
                        "only": only,
                        "ignore": ignore,
                    },
                }
            },
        )

    def _no_backend_target_response(
        self,
        *,
        route_decision: RouteDecision,
        request_id: str,
    ) -> JSONResponse:
        logger.warning(
            "proxy_no_target request_id=%s selected_model=%s fallback_models=%s",
            request_id,
            route_decision.selected_model,
            ",".join(route_decision.fallback_models),
        )
        self._proxy.audit(
            "proxy_no_target",
            request_id=request_id,
            selected_model=route_decision.selected_model,
            fallback_models=route_decision.fallback_models,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "type": "no_backend_target",
                    "message": "No backend account supports the selected model set.",
                    "selected_model": route_decision.selected_model,
                    "fallback_models": route_decision.fallback_models,
                }
            },
        )


class ProxyRequestExecutor:
    def __init__(self, *, proxy: BackendProxy) -> None:
        self._proxy = proxy
        self._attempt_processor = ProxyAttemptProcessor(proxy=proxy)
        self._preflight_planner = ProxyPreflightPlanner(proxy=proxy)

    async def execute(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        incoming_headers: Headers,
        route_decision: RouteDecision,
        stream: bool,
        request_id: str | None = None,
    ) -> Response:
        plan = self._build_execution_plan(
            path=path,
            payload=payload,
            incoming_headers=incoming_headers,
            route_decision=route_decision,
            stream=stream,
            request_id=request_id,
        )
        if plan.preflight_response is not None:
            return plan.preflight_response
        if plan.context is None:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": {"type": "internal_error", "message": "Missing execution context."}},
            )

        return await self._execute_target_attempts(
            candidate_targets=plan.candidate_targets,
            context=plan.context,
        )

    def _build_execution_plan(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        incoming_headers: Headers,
        route_decision: RouteDecision,
        stream: bool,
        request_id: str | None,
    ) -> ProxyExecutionPlan:
        rid = request_id or "-"
        (
            candidate_targets,
            parameter_rejections,
            requested_parameters,
        ) = self._preflight_planner.prepare_candidate_targets(
            route_decision=route_decision,
            payload=payload,
        )
        candidate_targets = self._preflight_planner.apply_fallback_policy(
            candidate_targets=candidate_targets,
            route_decision=route_decision,
            request_id=rid,
        )
        effective_selected_model, effective_fallback_models = (
            self._preflight_planner.effective_model_selection(
                candidate_targets=candidate_targets,
                route_decision=route_decision,
            )
        )
        request_started = time.perf_counter()
        self._preflight_planner.emit_proxy_start(
            path=path,
            route_decision=route_decision,
            stream=stream,
            request_id=rid,
            selected_model=effective_selected_model,
            fallback_models=effective_fallback_models,
            candidate_targets=len(candidate_targets),
        )
        if parameter_rejections:
            self._preflight_planner.emit_parameter_filter_audit(
                request_id=rid,
                requested_parameters=requested_parameters,
                parameter_rejections=parameter_rejections,
            )

        preflight_response = self._preflight_planner.preflight_constraint_response(
            route_decision=route_decision,
            candidate_targets=candidate_targets,
            requested_parameters=requested_parameters,
            parameter_rejections=parameter_rejections,
            request_id=rid,
        )
        if preflight_response is not None:
            return ProxyExecutionPlan(
                candidate_targets=candidate_targets,
                context=None,
                preflight_response=preflight_response,
            )

        attempt_context = ProxyAttemptRunContext(
            path=path,
            trial_payload=dict(payload),
            incoming_headers=incoming_headers,
            route_decision=route_decision,
            stream=stream,
            request_id=rid,
            total_attempts=len(candidate_targets),
            request_started=request_started,
        )
        return ProxyExecutionPlan(
            candidate_targets=candidate_targets,
            context=attempt_context,
        )

    async def _execute_target_attempts(
        self,
        *,
        candidate_targets: list[BackendTarget],
        context: ProxyAttemptRunContext,
    ) -> Response:
        stats = ProxyExecutionStats(
            attempted_targets=[],
            attempted_upstream_models=[],
        )
        for index, target in enumerate(candidate_targets):
            response = await self._attempt_single_target(
                target=target,
                context=context,
                attempt_number=index + 1,
                has_more_targets=index < context.total_attempts - 1,
                stats=stats,
            )
            if response is not None:
                return response

        return self._attempt_processor.routing_exhausted_response(
            request_id=context.request_id,
            candidate_targets_total=context.total_attempts,
            stats=stats,
        )

    async def _attempt_single_target(
        self,
        *,
        target: BackendTarget,
        context: ProxyAttemptRunContext,
        attempt_number: int,
        has_more_targets: bool,
        stats: ProxyExecutionStats,
    ) -> Response | None:
        breaker_key = f"{target.account_name}:{target.provider}"
        if self._attempt_processor.skip_target_for_circuit_breaker(
            target=target,
            breaker_key=breaker_key,
            request_id=context.request_id,
            stats=stats,
        ):
            return None
        if self._attempt_processor.skip_target_for_rate_limit(
            target=target,
            request_id=context.request_id,
            stats=stats,
        ):
            return None

        self._attempt_processor.record_attempt(
            target=target,
            request_id=context.request_id,
            attempt=attempt_number,
            total_attempts=context.total_attempts,
            stats=stats,
        )
        prepared_attempt = await self._prepare_attempt_request(
            target=target,
            context=context,
            has_more_targets=has_more_targets,
            stats=stats,
        )
        if isinstance(prepared_attempt, Response):
            return prepared_attempt
        if prepared_attempt is None:
            return None

        upstream, request_error_response = await self._attempt_processor.send_upstream_request(
            target=target,
            request_spec=prepared_attempt.request_spec,
            headers=prepared_attempt.headers,
            breaker_key=breaker_key,
            request_id=context.request_id,
            attempt=attempt_number,
            total_attempts=context.total_attempts,
            has_more_targets=has_more_targets,
            stats=stats,
        )
        if request_error_response is not None:
            return request_error_response
        if upstream is None:
            return None

        return await self._finalize_attempt_response(
            upstream=upstream,
            target=target,
            breaker_key=breaker_key,
            request_spec=prepared_attempt.request_spec,
            context=context,
            has_more_targets=has_more_targets,
            stats=stats,
        )

    async def _prepare_attempt_request(
        self,
        *,
        target: BackendTarget,
        context: ProxyAttemptRunContext,
        has_more_targets: bool,
        stats: ProxyExecutionStats,
    ) -> PreparedProxyAttempt | Response | None:
        context.trial_payload["model"] = target.upstream_model
        request_spec = _prepare_upstream_request(
            path=context.path,
            payload=context.trial_payload,
            provider=target.provider,
            stream=context.stream,
        )
        bearer_token = await self._proxy.resolve_bearer_token(target.account)
        skip_for_oauth, oauth_response = self._attempt_processor.handle_missing_oauth_token(
            target=target,
            bearer_token=bearer_token,
            request_id=context.request_id,
            has_more_targets=has_more_targets,
            stats=stats,
        )
        if oauth_response is not None:
            return oauth_response
        if skip_for_oauth:
            return None

        headers = _build_upstream_headers(
            incoming_headers=context.incoming_headers,
            bearer_token=bearer_token,
            provider=target.provider,
            oauth_account_id=self._proxy.resolve_oauth_account_id(target.account),
            organization=target.organization,
            project=target.project,
            stream=request_spec.stream,
            allow_passthrough_auth=target.account.allows_passthrough_auth(),
        )
        return PreparedProxyAttempt(
            request_spec=request_spec,
            headers=headers,
        )

    async def _finalize_attempt_response(
        self,
        *,
        upstream: httpx.Response,
        target: BackendTarget,
        breaker_key: str,
        request_spec: UpstreamRequestSpec,
        context: ProxyAttemptRunContext,
        has_more_targets: bool,
        stats: ProxyExecutionStats,
    ) -> Response | None:
        if self._attempt_processor.should_retry_upstream_response(
            upstream=upstream,
            target=target,
            breaker_key=breaker_key,
            request_id=context.request_id,
            has_more_targets=has_more_targets,
        ):
            await upstream.aclose()
            return None

        return await BackendProxy._to_fastapi_response(
            upstream=upstream,
            stream=context.stream,
            upstream_stream=request_spec.stream,
            target=target,
            attempted_targets=stats.attempted_targets,
            attempted_upstream_models=stats.attempted_upstream_models,
            request_latency_ms=round(
                (time.perf_counter() - context.request_started) * 1000.0, 3
            ),
            route_decision=context.route_decision,
            request_id=context.request_id,
            adapter=request_spec.adapter,
            audit_hook=self._proxy.audit_hook,
        )


class ModelRegistryResolver:
    def __init__(self, *, model_registry: dict[str, dict[str, Any]]) -> None:
        self._model_registry = model_registry

    @staticmethod
    def split_model_ref(model: str) -> tuple[str | None, str]:
        normalized = model.strip()
        if not normalized:
            return None, ""
        if "/" not in normalized:
            return None, normalized
        provider, model_id = normalized.split("/", 1)
        provider = provider.strip()
        model_id = model_id.strip()
        if not provider or not model_id:
            return None, normalized
        return provider, model_id

    def resolve_model_metadata(
        self, account: BackendAccount, model: str
    ) -> dict[str, Any]:
        metadata = self._model_registry.get(model)
        if isinstance(metadata, dict):
            return metadata

        # Fallback for metadata maps that only key by provider/modelId with empty metadata.
        provider, model_id = self.split_model_ref(model)
        account_provider = account.provider.strip().lower()
        if provider and provider.lower() == account_provider:
            provider_key = f"{provider.lower()}/{model_id}"
            provider_metadata = self._model_registry.get(provider_key)
            if isinstance(provider_metadata, dict):
                return provider_metadata

        if "/" not in model:
            provider_key = f"{account_provider}/{model.strip()}"
            provider_metadata = self._model_registry.get(provider_key)
            if isinstance(provider_metadata, dict):
                return provider_metadata

        return {}

    def resolve_upstream_model(
        self,
        account: BackendAccount,
        model: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resolved_metadata = (
            metadata
            if isinstance(metadata, dict)
            else self.resolve_model_metadata(account, model)
        )
        metadata_id = (
            resolved_metadata.get("id") if isinstance(resolved_metadata, dict) else None
        )
        if isinstance(metadata_id, str) and metadata_id.strip():
            return metadata_id.strip()

        # Fallback for metadata maps keyed by provider/modelId.
        provider, model_id = self.split_model_ref(model)
        if provider and provider.lower() == account.provider.strip().lower():
            return model_id

        # Backward-compatible fallback for plain model ids.
        return account.upstream_model(model)

    def resolve_target_metadata(self, target: BackendTarget) -> dict[str, Any]:
        if isinstance(target.metadata, dict):
            return target.metadata
        metadata = self._model_registry.get(target.model)
        if isinstance(metadata, dict):
            return metadata

        if "/" not in target.model:
            provider_key = f"{target.provider.strip().lower()}/{target.model.strip()}"
            provider_metadata = self._model_registry.get(provider_key)
            if isinstance(provider_metadata, dict):
                return provider_metadata

        return {}


class TargetSelectionPolicy:
    def __init__(
        self,
        *,
        target_metadata_resolver: Callable[[BackendTarget], dict[str, Any]],
    ) -> None:
        self._target_metadata_resolver = target_metadata_resolver

    @staticmethod
    def provider_partition_mode(provider_preferences: dict[str, Any]) -> str:
        partition = str(provider_preferences.get("partition") or "").strip().lower()
        if partition == "none":
            return "none"
        return "model"

    @staticmethod
    def requires_parameter_compatibility(provider_preferences: dict[str, Any]) -> bool:
        return bool(provider_preferences.get("require_parameters"))

    @staticmethod
    def allows_fallbacks(provider_preferences: dict[str, Any]) -> bool:
        value = provider_preferences.get("allow_fallbacks")
        if isinstance(value, bool):
            return value
        return True

    @staticmethod
    def normalized_provider_filter_values(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        output: list[str] = []
        seen: set[str] = set()
        for value in raw:
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output

    @staticmethod
    def has_provider_filters(provider_preferences: dict[str, Any]) -> bool:
        return bool(provider_preferences.get("only")) or bool(
            provider_preferences.get("ignore")
        )

    def filter_targets_by_provider_preferences(
        self,
        *,
        model_targets: list[BackendTarget],
        provider_preferences: dict[str, Any],
    ) -> list[BackendTarget]:
        only = set(
            self.normalized_provider_filter_values(provider_preferences.get("only"))
        )
        ignore = set(
            self.normalized_provider_filter_values(provider_preferences.get("ignore"))
        )
        if not only and not ignore:
            return model_targets

        filtered: list[BackendTarget] = []
        for target in model_targets:
            provider = target.provider.strip().lower()
            account = target.account_name.strip().lower()

            if only and provider not in only and account not in only:
                continue
            if provider in ignore or account in ignore:
                continue

            filtered.append(target)
        return filtered

    @staticmethod
    def provider_order_index_map(raw_order: Any) -> dict[str, int]:
        if not isinstance(raw_order, list):
            return {}
        output: dict[str, int] = {}
        for index, item in enumerate(raw_order):
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower()
            if not normalized:
                continue
            output.setdefault(normalized, index)
        return output

    def target_metadata(self, target: BackendTarget) -> dict[str, Any]:
        return self._target_metadata_resolver(target)

    @staticmethod
    def as_non_negative_float(value: Any, default: float = 0.0) -> float:
        if not isinstance(value, (int, float)):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0.0 else default

    def target_effective_price(self, target: BackendTarget) -> float:
        metadata = self.target_metadata(target)
        costs = metadata.get("costs")
        if not isinstance(costs, dict):
            return float("inf")
        input_cost = self.as_non_negative_float(costs.get("input_per_1k"), default=0.0)
        output_cost = self.as_non_negative_float(costs.get("output_per_1k"), default=0.0)
        if input_cost <= 0.0 and output_cost <= 0.0:
            return float("inf")
        return input_cost + output_cost

    def target_latency_ms(self, target: BackendTarget) -> float:
        metadata = self.target_metadata(target)
        priors = metadata.get("priors")
        if not isinstance(priors, dict):
            return float("inf")
        latency_ms = self.as_non_negative_float(priors.get("latency_ms"), default=0.0)
        return latency_ms if latency_ms > 0.0 else float("inf")

    def target_throughput(self, target: BackendTarget) -> float:
        metadata = self.target_metadata(target)
        priors = metadata.get("priors")
        if isinstance(priors, dict):
            throughput = self.as_non_negative_float(
                priors.get("throughput_tps"), default=0.0
            )
            if throughput > 0.0:
                return throughput

        latency_ms = self.target_latency_ms(target)
        if latency_ms == float("inf") or latency_ms <= 0.0:
            return 0.0
        return 1000.0 / latency_ms

    def sort_model_targets(
        self,
        *,
        model_targets: list[BackendTarget],
        provider_preferences: dict[str, Any],
    ) -> list[BackendTarget]:
        if len(model_targets) <= 1 or not provider_preferences:
            return model_targets

        order_index = self.provider_order_index_map(provider_preferences.get("order"))
        sort_by = str(provider_preferences.get("sort") or "").strip().lower()
        if sort_by not in {"price", "latency", "throughput"}:
            sort_by = ""

        def _sort_key(target: BackendTarget) -> tuple[float, float, str]:
            order_rank = float(order_index.get(target.provider.strip().lower(), 10_000))
            if order_rank >= 10_000:
                order_rank = float(
                    order_index.get(target.account_name.strip().lower(), 10_000)
                )

            metric = 0.0
            if sort_by == "price":
                metric = self.target_effective_price(target)
            elif sort_by == "latency":
                metric = self.target_latency_ms(target)
            elif sort_by == "throughput":
                metric = -self.target_throughput(target)

            return (
                order_rank,
                metric,
                target.account_name,
            )

        return sorted(model_targets, key=_sort_key)

    @staticmethod
    def extract_request_parameter_names(payload: dict[str, Any]) -> set[str]:
        ignored_keys = {
            "model",
            "messages",
            "input",
            "prompt",
            "stream",
            "provider",
            "allowed_models",
            "plugins",
        }
        output: set[str] = set()
        for raw_key in payload:
            if not isinstance(raw_key, str):
                continue
            key = raw_key.strip().lower()
            if not key or key in ignored_keys:
                continue
            output.add(key)
        return output

    @staticmethod
    def normalize_parameter_set(raw: Any) -> set[str]:
        if not isinstance(raw, list):
            return set()
        output: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower()
            if normalized:
                output.add(normalized)
        return output

    def filter_targets_by_parameter_support(
        self,
        *,
        candidate_targets: list[BackendTarget],
        payload: dict[str, Any],
    ) -> tuple[list[BackendTarget], dict[str, list[str]], list[str]]:
        requested = self.extract_request_parameter_names(payload)
        requested_parameters = sorted(requested)
        if not requested:
            return candidate_targets, {}, requested_parameters

        accepted: list[BackendTarget] = []
        rejected: dict[str, list[str]] = {}
        for target in candidate_targets:
            metadata = self.target_metadata(target)
            supported = self.normalize_parameter_set(metadata.get("supported_parameters"))
            unsupported = self.normalize_parameter_set(
                metadata.get("unsupported_parameters")
            )

            reasons: list[str] = []
            if supported:
                missing = sorted(requested - supported)
                if missing:
                    reasons.append(f"missing_supported_parameters:{','.join(missing)}")

            blocked = sorted(requested & unsupported)
            if blocked:
                reasons.append(f"unsupported_parameters:{','.join(blocked)}")

            if reasons:
                rejected[target.label] = reasons
                continue
            accepted.append(target)

        if accepted:
            return accepted, rejected, requested_parameters
        return [], rejected, requested_parameters


class BackendTargetPlanner:
    def __init__(self, *, proxy: BackendProxy) -> None:
        self._proxy = proxy

    def build_candidate_targets(
        self, route_decision: RouteDecision
    ) -> list[BackendTarget]:
        model_chain = self._candidate_model_chain(route_decision)
        provider_preferences = route_decision.provider_preferences or {}
        partition = self._proxy.provider_partition_mode(provider_preferences)
        grouped_targets = [
            self._build_model_targets(
                model=model,
                provider_preferences=provider_preferences,
            )
            for model in model_chain
        ]

        if (
            route_decision.source != "request"
            and self._proxy.allows_fallbacks(provider_preferences)
        ):
            grouped_targets = self._proxy.prioritize_model_groups_by_rate_limit(
                grouped_targets
            )

        return self._collect_targets_by_partition(
            grouped_targets=grouped_targets,
            provider_preferences=provider_preferences,
            partition=partition,
        )

    @staticmethod
    def _candidate_model_chain(route_decision: RouteDecision) -> list[str]:
        if route_decision.source == "request":
            return [route_decision.selected_model]
        return _dedupe_preserving_order(
            [route_decision.selected_model, *route_decision.fallback_models]
        )

    def _build_model_targets(
        self,
        *,
        model: str,
        provider_preferences: dict[str, Any],
    ) -> list[BackendTarget]:
        model_targets: list[BackendTarget] = []
        for account in self._proxy.accounts:
            if account.enabled and account.supports_model(model):
                metadata = self._proxy.resolve_model_metadata(account, model)
                model_targets.append(
                    BackendTarget(
                        account=account,
                        account_name=account.name,
                        provider=account.provider,
                        base_url=account.base_url,
                        model=model,
                        upstream_model=self._proxy.resolve_upstream_model(
                            account,
                            model,
                            metadata=metadata,
                        ),
                        auth_mode=account.auth_mode,
                        organization=account.organization,
                        project=account.project,
                        metadata=metadata,
                    )
                )
        return self._proxy.filter_targets_by_provider_preferences(
            model_targets=model_targets,
            provider_preferences=provider_preferences,
        )

    def _collect_targets_by_partition(
        self,
        *,
        grouped_targets: list[list[BackendTarget]],
        provider_preferences: dict[str, Any],
        partition: str,
    ) -> list[BackendTarget]:
        if partition == "none":
            flattened_targets = [target for group in grouped_targets for target in group]
            sorted_targets = self._proxy.sort_model_targets(
                model_targets=flattened_targets,
                provider_preferences=provider_preferences,
            )
            return self._proxy.prioritize_targets_by_rate_limit(sorted_targets)

        targets: list[BackendTarget] = []
        for model_targets in grouped_targets:
            sorted_targets = self._proxy.sort_model_targets(
                model_targets=model_targets,
                provider_preferences=provider_preferences,
            )
            targets.extend(self._proxy.prioritize_targets_by_rate_limit(sorted_targets))
        return targets


class RateLimitTracker:
    def __init__(
        self,
        *,
        rate_limited_until: dict[str, float],
        audit_emitter: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._rate_limited_until = rate_limited_until
        self._audit_emitter = audit_emitter

    def is_temporarily_rate_limited(self, account_name: str) -> bool:
        until = self._rate_limited_until.get(account_name)
        if not until:
            return False
        now = time.time()
        if now >= until:
            self._rate_limited_until.pop(account_name, None)
            return False
        return True

    def mark_rate_limited(
        self,
        account_name: str,
        headers: httpx.Headers,
        *,
        request_id: str | None = None,
        target: str | None = None,
        model: str | None = None,
        upstream_model: str | None = None,
    ) -> None:
        retry_after_seconds = _parse_retry_after_seconds(headers)
        until = time.time() + retry_after_seconds
        current = self._rate_limited_until.get(account_name, 0.0)
        if until > current:
            self._rate_limited_until[account_name] = until
        logger.info(
            (
                "proxy_rate_limited request_id=%s account=%s target=%s "
                "retry_after_seconds=%.1f"
            ),
            request_id,
            account_name,
            target,
            retry_after_seconds,
        )
        fields: dict[str, Any] = {
            "account": account_name,
            "retry_after_seconds": retry_after_seconds,
            "until_epoch": until,
        }
        if request_id:
            fields["request_id"] = request_id
        if target:
            fields["target"] = target
        if model:
            fields["model"] = model
        if upstream_model:
            fields["upstream_model"] = upstream_model
        if self._audit_emitter is not None:
            self._audit_emitter("proxy_rate_limited", fields)


class OAuthStateManager:
    def __init__(
        self,
        *,
        oauth_runtime: dict[str, OAuthRuntimeState],
        oauth_refresh_locks: dict[str, asyncio.Lock],
        oauth_persistence_lock: asyncio.Lock,
        oauth_state_persistence_path: Path | None,
        client_getter: Callable[[], httpx.AsyncClient],
    ) -> None:
        self._oauth_runtime = oauth_runtime
        self._oauth_refresh_locks = oauth_refresh_locks
        self._oauth_persistence_lock = oauth_persistence_lock
        self._oauth_state_persistence_path = oauth_state_persistence_path
        self._client_getter = client_getter

    async def resolve_bearer_token(self, account: BackendAccount) -> str | None:
        if account.auth_mode == "api_key":
            return account.resolved_api_key()
        if account.auth_mode == "passthrough":
            return None
        if account.auth_mode != "oauth":
            return account.resolved_api_key()

        state = self._oauth_runtime.get(account.name)
        if not state:
            state = OAuthRuntimeState(
                access_token=account.resolved_oauth_access_token() or "",
                refresh_token=account.resolved_oauth_refresh_token(),
                expires_at=account.resolved_oauth_expires_at(),
                account_id=account.resolved_oauth_account_id(),
            )
            self._oauth_runtime[account.name] = state

        if state.access_token and not _is_token_expiring(state.expires_at):
            if not state.account_id:
                state.account_id = _extract_chatgpt_account_id(state.access_token)
            return state.access_token

        if state.refresh_token:
            refreshed = await self.refresh_oauth_state(account, fallback_state=state)
            if refreshed and refreshed.access_token:
                return refreshed.access_token

        return state.access_token or None

    async def refresh_oauth_state(
        self,
        account: BackendAccount,
        fallback_state: OAuthRuntimeState | None = None,
    ) -> OAuthRuntimeState | None:
        token_url = account.effective_oauth_token_url()
        if not token_url:
            return fallback_state

        lock = self._oauth_refresh_locks.setdefault(account.name, asyncio.Lock())
        async with lock:
            current = self._oauth_runtime.get(account.name) or fallback_state
            if (
                current
                and current.access_token
                and not _is_token_expiring(current.expires_at)
            ):
                return current

            refresh_token = (
                current.refresh_token if current else None
            ) or account.resolved_oauth_refresh_token()
            if not refresh_token:
                logger.warning(
                    "oauth_refresh_skipped account=%s reason=missing_refresh_token",
                    account.name,
                )
                return current

            logger.info(
                "oauth_refresh_start account=%s token_url=%s", account.name, token_url
            )
            payload: dict[str, str] = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            client_id = account.resolved_oauth_client_id()
            client_secret = account.resolved_oauth_client_secret()
            if client_id:
                payload["client_id"] = client_id
            if client_secret:
                payload["client_secret"] = client_secret

            try:
                response = await self._client_getter().post(
                    token_url,
                    data=payload,
                    headers={"Accept": "application/json"},
                )
            except httpx.RequestError:
                logger.warning(
                    "oauth_refresh_error account=%s reason=request_error", account.name
                )
                return current

            if response.status_code >= 400:
                logger.warning(
                    "oauth_refresh_error account=%s status=%d",
                    account.name,
                    response.status_code,
                )
                return current

            try:
                body = response.json()
            except ValueError:
                logger.warning(
                    "oauth_refresh_error account=%s reason=invalid_json", account.name
                )
                return current

            raw_access = body.get("access_token")
            access_token = str(raw_access).strip() if raw_access is not None else ""
            if not access_token:
                logger.warning(
                    "oauth_refresh_error account=%s reason=missing_access_token",
                    account.name,
                )
                return current

            raw_refresh = body.get("refresh_token")
            next_refresh = (
                str(raw_refresh).strip() if raw_refresh is not None else refresh_token
            ) or None
            expires_at = _extract_expires_at(body)
            if expires_at is None and current:
                expires_at = current.expires_at

            state = OAuthRuntimeState(
                access_token=access_token,
                refresh_token=next_refresh,
                expires_at=expires_at,
                account_id=_extract_chatgpt_account_id(access_token),
            )
            self._oauth_runtime[account.name] = state
            await self.persist_oauth_state(account, state)
            logger.info(
                "oauth_refresh_success account=%s expires_at=%s",
                account.name,
                state.expires_at,
            )
            return state

    async def persist_oauth_state(
        self,
        account: BackendAccount,
        state: OAuthRuntimeState,
    ) -> None:
        config_path = self._oauth_state_persistence_path
        if config_path is None:
            return

        # Keep env-backed fields as source of truth.
        can_persist_access = account.oauth_access_token_env is None
        can_persist_refresh = account.oauth_refresh_token_env is None
        can_persist_expires = account.oauth_expires_at_env is None
        can_persist_account_id = account.oauth_account_id_env is None

        if not any(
            (
                can_persist_access,
                can_persist_refresh,
                can_persist_expires,
                can_persist_account_id,
            )
        ):
            return

        async with self._oauth_persistence_lock:
            await asyncio.to_thread(
                self.persist_oauth_state_sync,
                config_path,
                account.name,
                state,
                can_persist_access,
                can_persist_refresh,
                can_persist_expires,
                can_persist_account_id,
            )

    @staticmethod
    def persist_oauth_state_sync(
        config_path: Path,
        account_name: str,
        state: OAuthRuntimeState,
        can_persist_access: bool,
        can_persist_refresh: bool,
        can_persist_expires: bool,
        can_persist_account_id: bool,
    ) -> None:
        store = YamlFileStore(config_path)
        if not store.exists():
            logger.warning(
                "oauth_refresh_persist_skipped account=%s reason=config_not_found path=%s",
                account_name,
                config_path,
            )
            return

        try:
            raw = store.load(default={})
        except Exception:
            logger.warning(
                "oauth_refresh_persist_skipped account=%s reason=config_read_error path=%s",
                account_name,
                config_path,
            )
            return

        if not isinstance(raw, dict):
            logger.warning(
                "oauth_refresh_persist_skipped account=%s reason=invalid_config_root path=%s",
                account_name,
                config_path,
            )
            return

        accounts = raw.get("accounts")
        if not isinstance(accounts, list):
            logger.warning(
                "oauth_refresh_persist_skipped account=%s reason=missing_accounts path=%s",
                account_name,
                config_path,
            )
            return

        entry: dict[str, Any] | None = None
        for candidate in accounts:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("name", "")).strip() != account_name:
                continue
            entry = candidate
            break

        if entry is None:
            logger.warning(
                "oauth_refresh_persist_skipped account=%s reason=account_not_found path=%s",
                account_name,
                config_path,
            )
            return

        changed = False

        if can_persist_access and state.access_token:
            if entry.get("oauth_access_token") != state.access_token:
                entry["oauth_access_token"] = state.access_token
                changed = True

        if can_persist_refresh and state.refresh_token:
            if entry.get("oauth_refresh_token") != state.refresh_token:
                entry["oauth_refresh_token"] = state.refresh_token
                changed = True

        if can_persist_expires and state.expires_at is not None:
            if entry.get("oauth_expires_at") != state.expires_at:
                entry["oauth_expires_at"] = state.expires_at
                changed = True

        if can_persist_account_id and state.account_id:
            if entry.get("oauth_account_id") != state.account_id:
                entry["oauth_account_id"] = state.account_id
                changed = True

        if not changed:
            return

        try:
            store.write(raw, sort_keys=False)
        except Exception:
            logger.warning(
                "oauth_refresh_persist_skipped account=%s reason=config_write_error path=%s",
                account_name,
                config_path,
            )
            return

        logger.info(
            "oauth_refresh_persist_success account=%s path=%s",
            account_name,
            config_path,
        )

    def resolve_oauth_account_id(self, account: BackendAccount) -> str | None:
        if account.auth_mode != "oauth":
            return None
        runtime_state = self._oauth_runtime.get(account.name)
        if runtime_state and runtime_state.account_id:
            return runtime_state.account_id

        configured = account.resolved_oauth_account_id()
        if configured:
            return configured

        if runtime_state and runtime_state.access_token:
            account_id = _extract_chatgpt_account_id(runtime_state.access_token)
            runtime_state.account_id = account_id
            return account_id
        return None


class ChatCompletionsResponseAdapter:
    @staticmethod
    async def to_chat_completions_response(
        *,
        upstream: httpx.Response,
        stream: bool,
        response_headers: dict[str, str],
        model: str,
        request_id: str,
        audit_hook: Callable[[dict[str, Any]], None] | None,
        routing_diagnostics: dict[str, Any],
    ) -> Response:
        if upstream.status_code >= 400:
            body = await upstream.aread()
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                headers=response_headers,
            )

        created = int(time.time())
        state: dict[str, Any] = {
            "completion_id": f"chatcmpl-{request_id}",
            "created": created,
            "sent_role": False,
            "text_parts": [],
            "tool_calls": [],
            "tool_calls_by_item_id": {},
            "tool_calls_by_output_index": {},
            "saw_tool_call": False,
        }

        if stream:
            response_headers.pop("content-type", None)

            async def stream_generator() -> AsyncIterator[bytes]:
                try:
                    async for event in BackendProxy._iter_sse_data_json(upstream):
                        event_type = str(event.get("type", ""))
                        if event_type == "response.created":
                            response_obj = event.get("response")
                            if isinstance(response_obj, dict):
                                resp_id = response_obj.get("id")
                                if isinstance(resp_id, str) and resp_id.strip():
                                    state["completion_id"] = resp_id.strip()
                                resp_created = response_obj.get("created_at")
                                if isinstance(resp_created, (int, float)):
                                    state["created"] = int(resp_created)
                            continue

                        if event_type == "response.output_item.added":
                            item = event.get("item")
                            output_index = event.get("output_index")
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "function_call"
                                and isinstance(output_index, int)
                            ):
                                call_id = item.get("call_id")
                                name = item.get("name")
                                item_id = item.get("id")
                                if not isinstance(call_id, str):
                                    call_id = None
                                if not isinstance(name, str):
                                    name = None
                                if not isinstance(item_id, str):
                                    item_id = None

                                state["saw_tool_call"] = True
                                if not state["sent_role"]:
                                    state["sent_role"] = True
                                    yield BackendProxy._chat_completion_chunk(
                                        completion_id=state["completion_id"],
                                        created=state["created"],
                                        model=model,
                                        delta={"role": "assistant"},
                                        finish_reason=None,
                                    )

                                call_state: dict[str, Any] = {
                                    "index": output_index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name or "", "arguments": ""},
                                }
                                state["tool_calls"].append(call_state)
                                tool_calls_by_output_index = state[
                                    "tool_calls_by_output_index"
                                ]
                                tool_calls_by_output_index[output_index] = call_state
                                if item_id:
                                    state["tool_calls_by_item_id"][item_id] = call_state

                                yield BackendProxy._chat_completion_tool_call_chunk(
                                    completion_id=state["completion_id"],
                                    created=state["created"],
                                    model=model,
                                    index=output_index,
                                    call_id=call_id,
                                    name=name,
                                    arguments="",
                                )
                            continue

                        if event_type == "response.function_call_arguments.delta":
                            item_id = event.get("item_id")
                            output_index = event.get("output_index")
                            delta = event.get("delta")
                            if not isinstance(delta, str) or not delta:
                                continue
                            matched_call_state: dict[str, Any] | None = None
                            if isinstance(item_id, str):
                                matched_call_state = state["tool_calls_by_item_id"].get(
                                    item_id
                                )
                            if matched_call_state is None and isinstance(
                                output_index, int
                            ):
                                matched_call_state = state[
                                    "tool_calls_by_output_index"
                                ].get(output_index)
                            if matched_call_state is None:
                                continue
                            func = matched_call_state.get("function")
                            if isinstance(func, dict):
                                existing_args = func.get("arguments")
                                if not isinstance(existing_args, str):
                                    existing_args = ""
                                func["arguments"] = existing_args + delta
                            if not state["sent_role"]:
                                state["sent_role"] = True
                                yield BackendProxy._chat_completion_chunk(
                                    completion_id=state["completion_id"],
                                    created=state["created"],
                                    model=model,
                                    delta={"role": "assistant"},
                                    finish_reason=None,
                                )
                            call_index = matched_call_state.get("index")
                            if not isinstance(call_index, int):
                                continue
                            yield BackendProxy._chat_completion_tool_call_chunk(
                                completion_id=state["completion_id"],
                                created=state["created"],
                                model=model,
                                index=call_index,
                                arguments=delta,
                            )
                            continue

                        if event_type == "response.function_call_arguments.done":
                            item_id = event.get("item_id")
                            output_index = event.get("output_index")
                            arguments = event.get("arguments")
                            if not isinstance(arguments, str):
                                continue
                            matched_done_call_state: dict[str, Any] | None = None
                            if isinstance(item_id, str):
                                matched_done_call_state = state[
                                    "tool_calls_by_item_id"
                                ].get(item_id)
                            if matched_done_call_state is None and isinstance(
                                output_index, int
                            ):
                                matched_done_call_state = state[
                                    "tool_calls_by_output_index"
                                ].get(output_index)
                            if matched_done_call_state is None:
                                continue
                            func = matched_done_call_state.get("function")
                            if isinstance(func, dict):
                                func["arguments"] = arguments
                            continue

                        if event_type == "response.output_text.delta":
                            delta = event.get("delta")
                            if not isinstance(delta, str) or not delta:
                                continue
                            state["text_parts"].append(delta)
                            if not state["sent_role"]:
                                state["sent_role"] = True
                                yield BackendProxy._chat_completion_chunk(
                                    completion_id=state["completion_id"],
                                    created=state["created"],
                                    model=model,
                                    delta={"role": "assistant"},
                                    finish_reason=None,
                                )
                            yield BackendProxy._chat_completion_chunk(
                                completion_id=state["completion_id"],
                                created=state["created"],
                                model=model,
                                delta={"content": delta},
                                finish_reason=None,
                            )
                            continue

                        if event_type == "response.completed":
                            finish_reason = "stop"
                            if state["saw_tool_call"] and not state["text_parts"]:
                                finish_reason = "tool_calls"
                            if not state["sent_role"]:
                                yield BackendProxy._chat_completion_chunk(
                                    completion_id=state["completion_id"],
                                    created=state["created"],
                                    model=model,
                                    delta={"role": "assistant"},
                                    finish_reason=None,
                                )
                            yield BackendProxy._chat_completion_chunk(
                                completion_id=state["completion_id"],
                                created=state["created"],
                                model=model,
                                delta={},
                                finish_reason=finish_reason,
                            )
                            logger.info(
                                "proxy_chat_result request_id=%s model=%s stream=%s text_chars=%d tool_calls=%d finish_reason=%s",
                                request_id,
                                model,
                                True,
                                len("".join(state["text_parts"])),
                                len(state["tool_calls"]),
                                finish_reason,
                            )
                            if audit_hook is not None:
                                try:
                                    audit_hook(
                                        {
                                            "event": "proxy_chat_result",
                                            "request_id": request_id,
                                            "model": model,
                                            "stream": True,
                                            "text_chars": len(
                                                "".join(state["text_parts"])
                                            ),
                                            "tool_calls": len(state["tool_calls"]),
                                            "finish_reason": finish_reason,
                                            "text_preview": "".join(
                                                state["text_parts"]
                                            )[:500],
                                            "tool_call_summary": BackendProxy._tool_call_debug_summary(
                                                state["tool_calls"]
                                            ),
                                        }
                                    )
                                except Exception:
                                    pass
                            yield b"data: [DONE]\n\n"
                            return
                finally:
                    await upstream.aclose()

                # Fallback close for unusual upstream termination without completed event.
                finish_reason = "stop"
                if state["saw_tool_call"] and not state["text_parts"]:
                    finish_reason = "tool_calls"
                if not state["sent_role"]:
                    yield BackendProxy._chat_completion_chunk(
                        completion_id=state["completion_id"],
                        created=state["created"],
                        model=model,
                        delta={"role": "assistant"},
                        finish_reason=None,
                    )
                yield BackendProxy._chat_completion_chunk(
                    completion_id=state["completion_id"],
                    created=state["created"],
                    model=model,
                    delta={},
                    finish_reason=finish_reason,
                )
                logger.info(
                    "proxy_chat_result request_id=%s model=%s stream=%s text_chars=%d tool_calls=%d finish_reason=%s",
                    request_id,
                    model,
                    True,
                    len("".join(state["text_parts"])),
                    len(state["tool_calls"]),
                    finish_reason,
                )
                if audit_hook is not None:
                    try:
                        audit_hook(
                            {
                                "event": "proxy_chat_result",
                                "request_id": request_id,
                                "model": model,
                                "stream": True,
                                "text_chars": len("".join(state["text_parts"])),
                                "tool_calls": len(state["tool_calls"]),
                                "finish_reason": finish_reason,
                                "text_preview": "".join(state["text_parts"])[:500],
                                "tool_call_summary": BackendProxy._tool_call_debug_summary(
                                    state["tool_calls"]
                                ),
                            }
                        )
                    except Exception:
                        pass
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                content=stream_generator(),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type="text/event-stream",
            )

        try:
            async for event in BackendProxy._iter_sse_data_json(upstream):
                event_type = str(event.get("type", ""))
                if event_type == "response.created":
                    response_obj = event.get("response")
                    if isinstance(response_obj, dict):
                        resp_id = response_obj.get("id")
                        if isinstance(resp_id, str) and resp_id.strip():
                            state["completion_id"] = resp_id.strip()
                        resp_created = response_obj.get("created_at")
                        if isinstance(resp_created, (int, float)):
                            state["created"] = int(resp_created)
                    continue
                if event_type == "response.output_item.added":
                    item = event.get("item")
                    output_index = event.get("output_index")
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "function_call"
                        and isinstance(output_index, int)
                    ):
                        call_id = item.get("call_id")
                        name = item.get("name")
                        item_id = item.get("id")
                        if not isinstance(call_id, str):
                            call_id = None
                        if not isinstance(name, str):
                            name = ""
                        if not isinstance(item_id, str):
                            item_id = None

                        call_state: dict[str, Any] = {
                            "index": output_index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                        state["tool_calls"].append(call_state)
                        state["tool_calls_by_output_index"][output_index] = call_state
                        if item_id:
                            state["tool_calls_by_item_id"][item_id] = call_state
                        state["saw_tool_call"] = True
                    continue
                if event_type == "response.function_call_arguments.delta":
                    item_id = event.get("item_id")
                    output_index = event.get("output_index")
                    delta = event.get("delta")
                    if not isinstance(delta, str):
                        continue
                    matched_done_call_state: dict[str, Any] | None = None
                    if isinstance(item_id, str):
                        matched_done_call_state = state["tool_calls_by_item_id"].get(
                            item_id
                        )
                    if matched_done_call_state is None and isinstance(
                        output_index, int
                    ):
                        matched_done_call_state = state[
                            "tool_calls_by_output_index"
                        ].get(output_index)
                    if matched_done_call_state is None:
                        continue
                    func = matched_done_call_state.get("function")
                    if isinstance(func, dict):
                        existing_args = func.get("arguments")
                        if not isinstance(existing_args, str):
                            existing_args = ""
                        func["arguments"] = existing_args + delta
                    continue
                if event_type == "response.function_call_arguments.done":
                    item_id = event.get("item_id")
                    output_index = event.get("output_index")
                    arguments = event.get("arguments")
                    if not isinstance(arguments, str):
                        continue
                    matched_call_state: dict[str, Any] | None = None
                    if isinstance(item_id, str):
                        matched_call_state = state["tool_calls_by_item_id"].get(item_id)
                    if matched_call_state is None and isinstance(output_index, int):
                        matched_call_state = state["tool_calls_by_output_index"].get(
                            output_index
                        )
                    if matched_call_state is None:
                        continue
                    func = matched_call_state.get("function")
                    if isinstance(func, dict):
                        func["arguments"] = arguments
                    continue
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        state["text_parts"].append(delta)
                    continue
                if event_type == "response.output_text.done":
                    if not state["text_parts"]:
                        final_text = event.get("text")
                        if isinstance(final_text, str):
                            state["text_parts"].append(final_text)
                    continue
                if event_type == "response.completed":
                    break
        finally:
            await upstream.aclose()

        content = "".join(state["text_parts"])
        finish_reason = "stop"
        if state["saw_tool_call"] and not content:
            finish_reason = "tool_calls"
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if state["tool_calls"]:
            message["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": (call.get("function") or {}).get("name", ""),
                        "arguments": (call.get("function") or {}).get("arguments", ""),
                    },
                }
                for call in state["tool_calls"]
            ]
        response_body: dict[str, Any] = {
            "id": state["completion_id"],
            "object": "chat.completion",
            "created": state["created"],
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        response_body["_router"] = routing_diagnostics
        logger.info(
            "proxy_chat_result request_id=%s model=%s stream=%s text_chars=%d tool_calls=%d finish_reason=%s",
            request_id,
            model,
            False,
            len(content),
            len(state["tool_calls"]),
            finish_reason,
        )
        if audit_hook is not None:
            try:
                audit_hook(
                    {
                        "event": "proxy_chat_result",
                        "request_id": request_id,
                        "model": model,
                        "stream": False,
                        "text_chars": len(content),
                        "tool_calls": len(state["tool_calls"]),
                        "finish_reason": finish_reason,
                        "text_preview": content[:500],
                        "tool_call_summary": BackendProxy._tool_call_debug_summary(
                            state["tool_calls"]
                        ),
                    }
                )
            except Exception:
                pass
        response_headers.pop("content-type", None)
        return JSONResponse(
            status_code=upstream.status_code,
            content=response_body,
            headers=response_headers,
        )


class BackendProxy:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        backend_api_key: str | None,
        retry_statuses: list[int],
        accounts: list[BackendAccount] | None = None,
        model_registry: dict[str, dict[str, Any]] | None = None,
        audit_hook: Callable[[dict[str, Any]], None] | None = None,
        circuit_breakers: CircuitBreakerRegistry | None = None,
        oauth_state_persistence_path: str | Path | None = None,
        connect_timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        write_timeout_seconds: float | None = None,
        pool_timeout_seconds: float | None = None,
    ) -> None:
        self.retry_statuses = set(retry_statuses)
        connect_timeout = (
            max(0.1, float(connect_timeout_seconds))
            if connect_timeout_seconds is not None
            else max(0.1, min(5.0, timeout_seconds))
        )
        read_timeout = (
            max(0.1, float(read_timeout_seconds))
            if read_timeout_seconds is not None
            else max(0.1, float(timeout_seconds))
        )
        write_timeout = (
            max(0.1, float(write_timeout_seconds))
            if write_timeout_seconds is not None
            else max(0.1, float(timeout_seconds))
        )
        pool_timeout = (
            max(0.1, float(pool_timeout_seconds))
            if pool_timeout_seconds is not None
            else connect_timeout
        )
        http2_enabled = _can_enable_http2()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=None,
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
            http2=http2_enabled,
        )
        self._oauth_runtime: dict[str, OAuthRuntimeState] = {}
        self._oauth_refresh_locks: dict[str, asyncio.Lock] = {}
        self._oauth_persistence_lock = asyncio.Lock()
        self._oauth_state_persistence_path = (
            Path(oauth_state_persistence_path)
            if oauth_state_persistence_path is not None
            else None
        )
        self._oauth_state_manager = OAuthStateManager(
            oauth_runtime=self._oauth_runtime,
            oauth_refresh_locks=self._oauth_refresh_locks,
            oauth_persistence_lock=self._oauth_persistence_lock,
            oauth_state_persistence_path=self._oauth_state_persistence_path,
            client_getter=lambda: self.client,
        )
        self._model_registry = model_registry or {}
        self._model_registry_resolver = ModelRegistryResolver(
            model_registry=self._model_registry
        )
        self._target_selection_policy = TargetSelectionPolicy(
            target_metadata_resolver=self._model_registry_resolver.resolve_target_metadata
        )
        self._backend_target_planner = BackendTargetPlanner(proxy=self)
        self._request_executor = ProxyRequestExecutor(proxy=self)
        self._audit_hook = audit_hook
        self._circuit_breakers = circuit_breakers
        self._rate_limited_until: dict[str, float] = {}
        self._rate_limit_tracker = RateLimitTracker(
            rate_limited_until=self._rate_limited_until,
            audit_emitter=lambda event, fields: self._audit(event, **fields),
        )
        self.accounts = self._initialize_accounts(
            accounts=accounts,
            base_url=base_url,
            backend_api_key=backend_api_key,
        )

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _default_account(
        *,
        base_url: str,
        backend_api_key: str | None,
    ) -> BackendAccount:
        return BackendAccount(
            name="default",
            provider="default",
            base_url=base_url,
            api_key=backend_api_key,
            models=[],
            enabled=True,
        )

    @classmethod
    def _initialize_accounts(
        cls,
        *,
        accounts: list[BackendAccount] | None,
        base_url: str,
        backend_api_key: str | None,
    ) -> list[BackendAccount]:
        if accounts:
            return [account for account in accounts if account.enabled]
        return [
            cls._default_account(
                base_url=base_url,
                backend_api_key=backend_api_key,
            )
        ]

    def _audit(self, event: str, **fields: Any) -> None:
        if self._audit_hook is None:
            return
        try:
            self._audit_hook({"event": event, **fields})
        except Exception as exc:
            logger.debug("audit_write_failed event=%s error=%s", event, exc)

    async def forward_with_fallback(
        self,
        path: str,
        payload: dict[str, Any],
        incoming_headers: Headers,
        route_decision: RouteDecision,
        stream: bool,
        request_id: str | None = None,
    ) -> Response:
        return await self._request_executor.execute(
            path=path,
            payload=payload,
            incoming_headers=incoming_headers,
            route_decision=route_decision,
            stream=stream,
            request_id=request_id,
        )

    @staticmethod
    async def _to_fastapi_response(
        upstream: httpx.Response,
        stream: bool,
        upstream_stream: bool,
        target: BackendTarget,
        attempted_targets: list[str],
        attempted_upstream_models: list[str],
        request_latency_ms: float,
        route_decision: RouteDecision,
        request_id: str,
        adapter: Literal["passthrough", "chat_completions"],
        audit_hook: Callable[[dict[str, Any]], None] | None,
    ) -> Response:
        return await ProxyResponseAdapter.to_fastapi_response(
            upstream=upstream,
            stream=stream,
            upstream_stream=upstream_stream,
            target=target,
            attempted_targets=attempted_targets,
            attempted_upstream_models=attempted_upstream_models,
            request_latency_ms=request_latency_ms,
            route_decision=route_decision,
            request_id=request_id,
            adapter=adapter,
            audit_hook=audit_hook,
        )

    @staticmethod
    async def _iter_sse_data_json(
        upstream: httpx.Response,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            async for line in upstream.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    yield parsed
        except httpx.RequestError as exc:
            upstream_url = "<unknown>"
            try:
                upstream_url = str(upstream.request.url)
            except Exception:
                pass
            logger.warning(
                "proxy_upstream_stream_error url=%s error=%s",
                upstream_url,
                exc,
            )

    @staticmethod
    def _chat_completion_chunk(
        completion_id: str,
        created: int,
        model: str,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> bytes:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode("utf-8")

    @staticmethod
    def _chat_completion_tool_call_chunk(
        completion_id: str,
        created: int,
        model: str,
        *,
        index: int,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> bytes:
        function_delta: dict[str, Any] = {}
        if name is not None:
            function_delta["name"] = name
        if arguments is not None:
            function_delta["arguments"] = arguments

        tool_call_delta: dict[str, Any] = {
            "index": index,
            "function": function_delta,
        }
        if call_id is not None:
            tool_call_delta["id"] = call_id
            tool_call_delta["type"] = "function"

        return BackendProxy._chat_completion_chunk(
            completion_id=completion_id,
            created=created,
            model=model,
            delta={"tool_calls": [tool_call_delta]},
            finish_reason=None,
        )

    @staticmethod
    def _tool_call_debug_summary(
        tool_calls: list[dict[str, Any]],
        max_items: int = 3,
        max_args_chars: int = 240,
    ) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for call in tool_calls[:max_items]:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                function = {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = ""
            summary.append(
                {
                    "id": call.get("id") if isinstance(call, dict) else None,
                    "name": function.get("name"),
                    "arguments_preview": arguments[:max_args_chars],
                }
            )
        return summary

    @staticmethod
    async def _to_chat_completions_response(
        upstream: httpx.Response,
        stream: bool,
        response_headers: dict[str, str],
        model: str,
        request_id: str,
        audit_hook: Callable[[dict[str, Any]], None] | None,
        routing_diagnostics: dict[str, Any],
    ) -> Response:
        return await ChatCompletionsResponseAdapter.to_chat_completions_response(
            upstream=upstream,
            stream=stream,
            response_headers=response_headers,
            model=model,
            request_id=request_id,
            audit_hook=audit_hook,
            routing_diagnostics=routing_diagnostics,
        )

    def _build_candidate_targets(
        self, route_decision: RouteDecision
    ) -> list[BackendTarget]:
        return self._backend_target_planner.build_candidate_targets(route_decision)

    @staticmethod
    def allows_fallbacks(provider_preferences: dict[str, Any]) -> bool:
        return TargetSelectionPolicy.allows_fallbacks(provider_preferences)

    def _filter_targets_by_parameter_support(
        self,
        *,
        candidate_targets: list[BackendTarget],
        payload: dict[str, Any],
    ) -> tuple[list[BackendTarget], dict[str, list[str]], list[str]]:
        return self._target_selection_policy.filter_targets_by_parameter_support(
            candidate_targets=candidate_targets,
            payload=payload,
        )

    def filter_targets_by_parameter_support(
        self,
        *,
        candidate_targets: list[BackendTarget],
        payload: dict[str, Any],
    ) -> tuple[list[BackendTarget], dict[str, list[str]], list[str]]:
        return self._filter_targets_by_parameter_support(
            candidate_targets=candidate_targets,
            payload=payload,
        )

    async def _resolve_bearer_token(self, account: BackendAccount) -> str | None:
        return await self._oauth_state_manager.resolve_bearer_token(account)

    def _resolve_oauth_account_id(self, account: BackendAccount) -> str | None:
        return self._oauth_state_manager.resolve_oauth_account_id(account)

    @property
    def circuit_breakers(self) -> CircuitBreakerRegistry | None:
        return self._circuit_breakers

    @property
    def audit_hook(self) -> Callable[[dict[str, Any]], None] | None:
        return self._audit_hook

    def audit(self, event: str, **fields: Any) -> None:
        self._audit(event, **fields)

    def build_candidate_targets(
        self, route_decision: RouteDecision
    ) -> list[BackendTarget]:
        return self._build_candidate_targets(route_decision)

    @staticmethod
    def provider_partition_mode(provider_preferences: dict[str, Any]) -> str:
        return TargetSelectionPolicy.provider_partition_mode(provider_preferences)

    @staticmethod
    def requires_parameter_compatibility(
        provider_preferences: dict[str, Any],
    ) -> bool:
        return TargetSelectionPolicy.requires_parameter_compatibility(
            provider_preferences
        )

    @staticmethod
    def normalized_provider_filter_values(raw: Any) -> list[str]:
        return TargetSelectionPolicy.normalized_provider_filter_values(raw)

    @staticmethod
    def has_provider_filters(provider_preferences: dict[str, Any]) -> bool:
        return TargetSelectionPolicy.has_provider_filters(provider_preferences)

    def resolve_model_metadata(
        self, account: BackendAccount, model: str
    ) -> dict[str, Any]:
        return self._model_registry_resolver.resolve_model_metadata(account, model)

    def resolve_upstream_model(
        self,
        account: BackendAccount,
        model: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._model_registry_resolver.resolve_upstream_model(
            account,
            model,
            metadata=metadata,
        )

    def filter_targets_by_provider_preferences(
        self,
        *,
        model_targets: list[BackendTarget],
        provider_preferences: dict[str, Any],
    ) -> list[BackendTarget]:
        return self._target_selection_policy.filter_targets_by_provider_preferences(
            model_targets=model_targets,
            provider_preferences=provider_preferences,
        )

    def sort_model_targets(
        self,
        *,
        model_targets: list[BackendTarget],
        provider_preferences: dict[str, Any],
    ) -> list[BackendTarget]:
        return self._target_selection_policy.sort_model_targets(
            model_targets=model_targets,
            provider_preferences=provider_preferences,
        )

    def prioritize_model_groups_by_rate_limit(
        self, grouped_targets: list[list[BackendTarget]]
    ) -> list[list[BackendTarget]]:
        if len(grouped_targets) <= 1:
            return grouped_targets

        healthy_groups: list[list[BackendTarget]] = []
        limited_groups: list[list[BackendTarget]] = []
        for model_targets in grouped_targets:
            if self._group_has_available_targets(model_targets):
                healthy_groups.append(model_targets)
            else:
                limited_groups.append(model_targets)

        if not healthy_groups or not limited_groups:
            return grouped_targets
        return [*healthy_groups, *limited_groups]

    def prioritize_targets_by_rate_limit(
        self, model_targets: list[BackendTarget]
    ) -> list[BackendTarget]:
        if len(model_targets) <= 1:
            return model_targets

        healthy_targets, limited_targets = self._partition_targets_by_rate_limit(
            model_targets
        )
        if not healthy_targets or not limited_targets:
            return model_targets
        return [*healthy_targets, *limited_targets]

    def _group_has_available_targets(self, model_targets: list[BackendTarget]) -> bool:
        return any(
            not self._rate_limit_tracker.is_temporarily_rate_limited(target.account_name)
            for target in model_targets
        )

    def _partition_targets_by_rate_limit(
        self, model_targets: list[BackendTarget]
    ) -> tuple[list[BackendTarget], list[BackendTarget]]:
        healthy_targets: list[BackendTarget] = []
        limited_targets: list[BackendTarget] = []
        for target in model_targets:
            if self._rate_limit_tracker.is_temporarily_rate_limited(target.account_name):
                limited_targets.append(target)
            else:
                healthy_targets.append(target)
        return healthy_targets, limited_targets

    def is_temporarily_rate_limited(self, account_name: str) -> bool:
        return self._rate_limit_tracker.is_temporarily_rate_limited(account_name)

    async def resolve_bearer_token(self, account: BackendAccount) -> str | None:
        return await self._resolve_bearer_token(account)

    def resolve_oauth_account_id(self, account: BackendAccount) -> str | None:
        return self._resolve_oauth_account_id(account)

    def mark_rate_limited(
        self,
        account_name: str,
        headers: httpx.Headers,
        *,
        request_id: str | None = None,
        target: str | None = None,
        model: str | None = None,
        upstream_model: str | None = None,
    ) -> None:
        self._rate_limit_tracker.mark_rate_limited(
            account_name,
            headers,
            request_id=request_id,
            target=target,
            model=model,
            upstream_model=upstream_model,
        )


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _can_enable_http2() -> bool:
    try:
        import h2  # noqa: F401
    except Exception:
        return False
    return True


def _is_token_expiring(expires_at: int | None, skew_seconds: int = 60) -> bool:
    if expires_at is None:
        return False
    return expires_at <= int(time.time()) + skew_seconds


def _extract_expires_at(token_response: dict[str, Any]) -> int | None:
    now = int(time.time())

    raw_expires_in = token_response.get("expires_in")
    if raw_expires_in is not None:
        try:
            return now + int(float(raw_expires_in))
        except (TypeError, ValueError):
            pass

    raw_expires_at = token_response.get("expires_at")
    if raw_expires_at is not None:
        try:
            return int(float(raw_expires_at))
        except (TypeError, ValueError):
            pass

    return None


def _extract_chatgpt_account_id(token: str | None) -> str | None:
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload_b64 = parts[1]
    padding = "=" * ((4 - len(payload_b64) % 4) % 4)

    try:
        import base64
        import json

        payload_raw = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_raw.decode("utf-8"))
    except Exception:
        return None

    auth_claim = payload.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        account_id = auth_claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id.strip():
            return account_id.strip()
    return None


def _parse_retry_after_seconds(
    headers: httpx.Headers, default_seconds: float = 30.0
) -> float:
    raw = headers.get("retry-after")
    if not raw:
        return default_seconds

    value = raw.strip()
    if not value:
        return default_seconds

    try:
        seconds = float(value)
        if seconds > 0:
            return seconds
    except (TypeError, ValueError):
        pass

    try:
        retry_dt = parsedate_to_datetime(value)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        delta = (retry_dt - datetime.now(timezone.utc)).total_seconds()
        if delta > 0:
            return float(delta)
    except Exception:
        pass

    return default_seconds
