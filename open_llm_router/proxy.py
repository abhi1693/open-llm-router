from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import time
from typing import Any, AsyncIterator, Callable, Literal

import httpx
from fastapi import status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import Headers

from open_llm_router.config import BackendAccount
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


def _build_upstream_headers(
    incoming_headers: Headers,
    bearer_token: str | None,
    provider: str,
    oauth_account_id: str | None,
    organization: str | None,
    project: str | None,
    allow_passthrough_auth: bool = False,
) -> dict[str, str]:
    passthrough = {
        "accept",
        "content-type",
        "openai-organization",
        "openai-project",
    }
    canonical_names = {
        "accept": "Accept",
        "content-type": "Content-Type",
        "openai-organization": "OpenAI-Organization",
        "openai-project": "OpenAI-Project",
    }
    headers: dict[str, str] = {}
    for name, value in incoming_headers.items():
        lower = name.lower()
        if lower in {"host", "content-length", "connection", "authorization"}:
            continue
        if lower in passthrough or lower.startswith("x-"):
            key = canonical_names.get(lower, name)
            headers[key] = value

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif allow_passthrough_auth:
        incoming_auth = incoming_headers.get("authorization")
        if incoming_auth:
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
        headers["Accept"] = "application/json"
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


def _extract_codex_instructions(payload: dict[str, Any], messages: list[dict[str, Any]]) -> str:
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


def _chat_messages_to_codex_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        return [{"role": "user", "content": [_as_input_text_part(text)]}] if text else []

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
                    output.append({"role": "user", "content": [_as_input_text_part(text)]})
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
            codex_payload["input"] = _normalize_responses_input_for_codex(payload.get("input"))
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
        codex_payload["tool_choice"] = _normalize_codex_tool_choice(payload.get("tool_choice"))
    elif "function_call" in payload:
        function_call = payload.get("function_call")
        if isinstance(function_call, str):
            codex_payload["tool_choice"] = function_call
        elif isinstance(function_call, dict):
            name = function_call.get("name")
            if isinstance(name, str) and name.strip():
                codex_payload["tool_choice"] = {"type": "function", "name": name.strip()}

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


def _prepare_gemini_chat_payload(payload: dict[str, Any], stream: bool) -> dict[str, Any]:
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

    return _drop_none_fields(prepared)


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
        adapter = "chat_completions" if path == "/v1/chat/completions" else "passthrough"
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

    return UpstreamRequestSpec(path=path, payload=payload, stream=stream)


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
    ) -> None:
        self.retry_statuses = set(retry_statuses)
        self.client = httpx.AsyncClient(timeout=timeout_seconds)
        self._oauth_runtime: dict[str, OAuthRuntimeState] = {}
        self._oauth_refresh_locks: dict[str, asyncio.Lock] = {}
        self._rate_limited_until: dict[str, float] = {}
        self._model_registry = model_registry or {}
        self._audit_hook = audit_hook
        if accounts:
            self.accounts = [account for account in accounts if account.enabled]
        else:
            self.accounts = [
                BackendAccount(
                    name="default",
                    provider="default",
                    base_url=base_url,
                    api_key=backend_api_key,
                    models=[],
                    enabled=True,
                )
            ]

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _split_model_ref(model: str) -> tuple[str | None, str]:
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

    def _resolve_upstream_model(self, account: BackendAccount, model: str) -> str:
        metadata = self._model_registry.get(model)
        if isinstance(metadata, dict):
            metadata_id = metadata.get("id")
            if isinstance(metadata_id, str) and metadata_id.strip():
                return metadata_id.strip()

        # Fallback for metadata maps that only key by provider/modelId with empty metadata.
        provider, model_id = self._split_model_ref(model)
        if provider and provider.lower() == account.provider.strip().lower():
            return model_id

        # Backward-compatible fallback for plain model ids.
        return account.upstream_model(model)

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
        payload: dict,
        incoming_headers: Headers,
        route_decision: RouteDecision,
        stream: bool,
        request_id: str | None = None,
    ) -> Response:
        candidate_targets = self._build_candidate_targets(route_decision)
        rid = request_id or "-"
        logger.info(
            (
                "proxy_start request_id=%s path=%s selected_model=%s stream=%s "
                "candidate_targets=%d"
            ),
            rid,
            path,
            route_decision.selected_model,
            stream,
            len(candidate_targets),
        )
        self._audit(
            "proxy_start",
            request_id=rid,
            path=path,
            selected_model=route_decision.selected_model,
            source=route_decision.source,
            task=route_decision.task,
            complexity=route_decision.complexity,
            requested_model=route_decision.requested_model,
            fallback_models=route_decision.fallback_models,
            stream=stream,
            candidate_targets=len(candidate_targets),
        )
        if not candidate_targets:
            logger.warning(
                "proxy_no_target request_id=%s selected_model=%s fallback_models=%s",
                rid,
                route_decision.selected_model,
                ",".join(route_decision.fallback_models),
            )
            self._audit(
                "proxy_no_target",
                request_id=rid,
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

        attempted_targets: list[str] = []

        for index, target in enumerate(candidate_targets):
            if self._is_temporarily_rate_limited(target.account_name):
                logger.info(
                    "proxy_skip_rate_limited request_id=%s target=%s",
                    rid,
                    target.label,
                )
                self._audit(
                    "proxy_skip_rate_limited",
                    request_id=rid,
                    target=target.label,
                    account=target.account_name,
                    model=target.model,
                )
                continue
            attempted_targets.append(target.label)
            logger.info(
                "proxy_attempt request_id=%s attempt=%d/%d target=%s provider=%s auth_mode=%s",
                rid,
                index + 1,
                len(candidate_targets),
                target.label,
                target.provider,
                target.auth_mode,
            )
            self._audit(
                "proxy_attempt",
                request_id=rid,
                attempt=index + 1,
                total_attempts=len(candidate_targets),
                target=target.label,
                account=target.account_name,
                provider=target.provider,
                model=target.model,
                auth_mode=target.auth_mode,
            )
            trial_payload = deepcopy(payload)
            trial_payload["model"] = target.upstream_model
            request_spec = _prepare_upstream_request(
                path=path,
                payload=trial_payload,
                provider=target.provider,
                stream=stream,
            )
            bearer_token = await self._resolve_bearer_token(target.account)
            if target.auth_mode == "oauth" and not bearer_token:
                logger.warning(
                    "proxy_oauth_token_missing request_id=%s target=%s",
                    rid,
                    target.label,
                )
                self._audit(
                    "proxy_oauth_token_missing",
                    request_id=rid,
                    target=target.label,
                    account=target.account_name,
                    model=target.model,
                )
                if index < len(candidate_targets) - 1:
                    continue
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": {
                            "type": "oauth_credentials_unavailable",
                            "message": (
                                "Selected OAuth-backed target has no usable access token "
                                "and could not refresh one."
                            ),
                            "attempted_targets": attempted_targets,
                        }
                    },
                )
            headers = _build_upstream_headers(
                incoming_headers=incoming_headers,
                bearer_token=bearer_token,
                provider=target.provider,
                oauth_account_id=self._resolve_oauth_account_id(target.account),
                organization=target.organization,
                project=target.project,
                allow_passthrough_auth=target.account.allows_passthrough_auth(),
            )

            try:
                request = self.client.build_request(
                    method="POST",
                    url=f"{target.base_url.rstrip('/')}{request_spec.path}",
                    json=request_spec.payload,
                    headers=headers,
                )
                upstream = await self.client.send(request, stream=request_spec.stream)
            except httpx.RequestError as exc:
                logger.warning(
                    "proxy_request_error request_id=%s target=%s error=%s",
                    rid,
                    target.label,
                    exc,
                )
                self._audit(
                    "proxy_request_error",
                    request_id=rid,
                    target=target.label,
                    account=target.account_name,
                    model=target.model,
                    error=str(exc),
                )
                if index < len(candidate_targets) - 1:
                    continue
                return JSONResponse(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    content={
                        "error": {
                            "type": "upstream_connection_error",
                            "message": f"Could not reach backend: {exc}",
                            "attempted_targets": attempted_targets,
                        }
                    },
                )

            should_retry = (
                upstream.status_code in self.retry_statuses
                and index < len(candidate_targets) - 1
            )
            if upstream.status_code == 429:
                self._mark_rate_limited(target.account_name, upstream.headers)
            if should_retry:
                logger.info(
                    "proxy_retry request_id=%s target=%s status=%d",
                    rid,
                    target.label,
                    upstream.status_code,
                )
                self._audit(
                    "proxy_retry",
                    request_id=rid,
                    target=target.label,
                    account=target.account_name,
                    model=target.model,
                    status=upstream.status_code,
                )
                await upstream.aclose()
                continue

            return await self._to_fastapi_response(
                upstream=upstream,
                stream=stream,
                upstream_stream=request_spec.stream,
                target=target,
                attempted_targets=attempted_targets,
                route_decision=route_decision,
                request_id=rid,
                adapter=request_spec.adapter,
                audit_hook=self._audit_hook,
            )

        logger.error(
            "proxy_exhausted request_id=%s attempted_targets=%s",
            rid,
            ",".join(attempted_targets),
        )
        self._audit(
            "proxy_exhausted",
            request_id=rid,
            attempted_targets=attempted_targets,
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": {
                    "type": "routing_exhausted",
                    "message": "All model/account targets failed.",
                    "attempted_targets": attempted_targets,
                }
            },
        )

    @staticmethod
    async def _to_fastapi_response(
        upstream: httpx.Response,
        stream: bool,
        upstream_stream: bool,
        target: BackendTarget,
        attempted_targets: list[str],
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
        response_headers["x-router-task"] = route_decision.task
        response_headers["x-router-complexity"] = route_decision.complexity
        response_headers["x-router-source"] = route_decision.source
        response_headers["x-router-attempted-targets"] = ",".join(attempted_targets)
        if route_decision.ranked_models:
            response_headers["x-router-ranked-models"] = ",".join(route_decision.ranked_models)
        if route_decision.candidate_scores:
            top = route_decision.candidate_scores[0]
            response_headers["x-router-top-utility"] = str(top.get("utility", ""))

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
                        "status": upstream.status_code,
                        "attempts": len(attempted_targets),
                        "attempted_targets": attempted_targets,
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
            )

        if stream:
            media_type = response_headers.pop("content-type", "text/event-stream")

            async def stream_generator():
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
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    @staticmethod
    async def _iter_sse_data_json(upstream: httpx.Response) -> AsyncIterator[dict[str, Any]]:
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
                                state["tool_calls_by_output_index"][output_index] = call_state
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
                            call_state = None
                            if isinstance(item_id, str):
                                call_state = state["tool_calls_by_item_id"].get(item_id)
                            if call_state is None and isinstance(output_index, int):
                                call_state = state["tool_calls_by_output_index"].get(output_index)
                            if call_state is None:
                                continue
                            func = call_state.get("function")
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
                            call_index = call_state.get("index")
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
                            call_state = None
                            if isinstance(item_id, str):
                                call_state = state["tool_calls_by_item_id"].get(item_id)
                            if call_state is None and isinstance(output_index, int):
                                call_state = state["tool_calls_by_output_index"].get(output_index)
                            if call_state is None:
                                continue
                            func = call_state.get("function")
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
                    call_state = None
                    if isinstance(item_id, str):
                        call_state = state["tool_calls_by_item_id"].get(item_id)
                    if call_state is None and isinstance(output_index, int):
                        call_state = state["tool_calls_by_output_index"].get(output_index)
                    if call_state is None:
                        continue
                    func = call_state.get("function")
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
                    call_state = None
                    if isinstance(item_id, str):
                        call_state = state["tool_calls_by_item_id"].get(item_id)
                    if call_state is None and isinstance(output_index, int):
                        call_state = state["tool_calls_by_output_index"].get(output_index)
                    if call_state is None:
                        continue
                    func = call_state.get("function")
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
        body = {
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
        return JSONResponse(status_code=upstream.status_code, content=body, headers=response_headers)

    def _build_candidate_targets(self, route_decision: RouteDecision) -> list[BackendTarget]:
        if route_decision.source == "request":
            model_chain = [route_decision.selected_model]
        else:
            model_chain = _dedupe_preserving_order(
                [route_decision.selected_model, *route_decision.fallback_models]
            )
        targets: list[BackendTarget] = []
        for model in model_chain:
            for account in self.accounts:
                if account.enabled and account.supports_model(model):
                    targets.append(
                        BackendTarget(
                            account=account,
                            account_name=account.name,
                            provider=account.provider,
                            base_url=account.base_url,
                            model=model,
                            upstream_model=self._resolve_upstream_model(account, model),
                            auth_mode=account.auth_mode,
                            organization=account.organization,
                            project=account.project,
                        )
                    )
        return targets

    def _is_temporarily_rate_limited(self, account_name: str) -> bool:
        until = self._rate_limited_until.get(account_name)
        if not until:
            return False
        now = time.time()
        if now >= until:
            self._rate_limited_until.pop(account_name, None)
            return False
        return True

    def _mark_rate_limited(self, account_name: str, headers: httpx.Headers) -> None:
        retry_after_seconds = _parse_retry_after_seconds(headers)
        until = time.time() + retry_after_seconds
        current = self._rate_limited_until.get(account_name, 0.0)
        if until > current:
            self._rate_limited_until[account_name] = until
        logger.info(
            "proxy_rate_limited account=%s retry_after_seconds=%.1f",
            account_name,
            retry_after_seconds,
        )
        self._audit(
            "proxy_rate_limited",
            account=account_name,
            retry_after_seconds=retry_after_seconds,
            until_epoch=until,
        )

    async def _resolve_bearer_token(self, account: BackendAccount) -> str | None:
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
            refreshed = await self._refresh_oauth_state(account, fallback_state=state)
            if refreshed and refreshed.access_token:
                return refreshed.access_token

        return state.access_token or None

    async def _refresh_oauth_state(
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
            if current and current.access_token and not _is_token_expiring(current.expires_at):
                return current

            refresh_token = (
                (current.refresh_token if current else None)
                or account.resolved_oauth_refresh_token()
            )
            if not refresh_token:
                logger.warning(
                    "oauth_refresh_skipped account=%s reason=missing_refresh_token",
                    account.name,
                )
                return current

            logger.info("oauth_refresh_start account=%s token_url=%s", account.name, token_url)
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
                response = await self.client.post(
                    token_url,
                    data=payload,
                    headers={"Accept": "application/json"},
                )
            except httpx.RequestError:
                logger.warning("oauth_refresh_error account=%s reason=request_error", account.name)
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
                logger.warning("oauth_refresh_error account=%s reason=invalid_json", account.name)
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
            logger.info(
                "oauth_refresh_success account=%s expires_at=%s",
                account.name,
                state.expires_at,
            )
            return state

    def _resolve_oauth_account_id(self, account: BackendAccount) -> str | None:
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


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _is_token_expiring(expires_at: int | None, skew_seconds: int = 60) -> bool:
    if expires_at is None:
        return False
    return expires_at <= int(time.time()) + skew_seconds


def _extract_expires_at(token_response: dict) -> int | None:
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


def _parse_retry_after_seconds(headers: httpx.Headers, default_seconds: float = 30.0) -> float:
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
            return delta
    except Exception:
        pass

    return default_seconds
