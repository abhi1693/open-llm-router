from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from open_llm_router.audit import JsonlAuditLogger
from open_llm_router.auth import AuthConfigurationError, Authenticator
from open_llm_router.config import RoutingConfig, load_routing_config
from open_llm_router.proxy import BackendProxy
from open_llm_router.router_engine import InvalidModelError, SmartModelRouter
from open_llm_router.settings import get_settings

app = FastAPI(
    title="Open-LLM Router",
    description="OpenAI-compatible API router with model auto-selection and fallback.",
    version="0.1.0",
)

logger = logging.getLogger("uvicorn.error")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not request.url.path.startswith("/v1"):
        return await call_next(request)

    authenticator: Authenticator | None = getattr(app.state, "authenticator", None)
    if authenticator is not None:
        auth_error = await authenticator.authenticate_request(request)
        if auth_error is not None:
            return auth_error

    return await call_next(request)


def _build_models_response(config: RoutingConfig) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "auto",
                "object": "model",
                "created": 0,
                "owned_by": "open-llm-router",
            }
        ],
    }


def _build_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    tools = payload.get("tools")
    functions = payload.get("functions")
    input_value = payload.get("input")

    summary: dict[str, Any] = {
        "stream": bool(payload.get("stream")),
        "has_tools": bool(tools) or bool(functions),
        "tool_choice_type": type(payload.get("tool_choice")).__name__
        if "tool_choice" in payload
        else None,
        "reasoning_effort": (
            payload.get("reasoning_effort")
            or (payload.get("reasoning", {}) or {}).get("effort")
            if isinstance(payload.get("reasoning"), dict)
            else payload.get("reasoning_effort")
        ),
    }

    if isinstance(messages, list):
        summary["messages_count"] = len(messages)
    if isinstance(input_value, list):
        summary["input_count"] = len(input_value)
    if isinstance(input_value, str):
        summary["input_chars"] = len(input_value)
    if isinstance(payload.get("max_tokens"), int):
        summary["max_tokens"] = payload.get("max_tokens")
    if isinstance(payload.get("max_output_tokens"), int):
        summary["max_output_tokens"] = payload.get("max_output_tokens")
    return summary


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    routing_config = load_routing_config(settings.routing_config_path)
    authenticator = Authenticator(settings)
    app.state.settings = settings
    app.state.authenticator = authenticator
    app.state.routing_config = routing_config
    app.state.smart_router = SmartModelRouter(routing_config)
    audit_logger = JsonlAuditLogger(
        path=settings.router_audit_log_path,
        enabled=settings.router_audit_log_enabled,
    )
    app.state.audit_logger = audit_logger
    app.state.backend_proxy = BackendProxy(
        base_url=settings.backend_base_url,
        timeout_seconds=settings.backend_timeout_seconds,
        backend_api_key=settings.backend_api_key,
        retry_statuses=routing_config.retry_statuses,
        accounts=routing_config.accounts,
        audit_hook=audit_logger.log,
    )
    logger.info(
        (
            "startup complete routing_config_path=%s accounts=%d models=%d default_model=%s "
            "audit_log_enabled=%s audit_log_path=%s"
        ),
        settings.routing_config_path,
        len(routing_config.accounts),
        len(routing_config.available_models()),
        routing_config.default_model,
        settings.router_audit_log_enabled,
        settings.router_audit_log_path,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    proxy: BackendProxy = app.state.backend_proxy
    await proxy.close()
    audit_logger: JsonlAuditLogger | None = getattr(app.state, "audit_logger", None)
    if audit_logger is not None:
        audit_logger.close()
    logger.info("shutdown complete")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    config: RoutingConfig = app.state.routing_config
    return _build_models_response(config)


async def _proxy_json_request(request: Request, path: str):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Expected JSON body: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object request body.")

    router: SmartModelRouter = app.state.smart_router
    proxy: BackendProxy = app.state.backend_proxy

    request_id = (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or uuid4().hex[:12]
    )
    try:
        route_decision = router.decide(payload=payload, endpoint=path)
    except InvalidModelError as exc:
        logger.info(
            "invalid_model request_id=%s requested_model=%s available_models=%d",
            request_id,
            exc.requested_model,
            len(exc.available_models),
        )
        raise HTTPException(
            status_code=400,
            detail={
                "type": "invalid_model",
                "message": str(exc),
                "requested_model": exc.requested_model,
                "available_models": sorted(exc.available_models),
            },
        ) from exc
    logger.info(
        (
            "route_decision request_id=%s path=%s source=%s task=%s complexity=%s "
            "requested_model=%s selected_model=%s fallback_count=%d"
        ),
        request_id,
        path,
        route_decision.source,
        route_decision.task,
        route_decision.complexity,
        route_decision.requested_model,
        route_decision.selected_model,
        len(route_decision.fallback_models),
    )
    if route_decision.candidate_scores:
        top = route_decision.candidate_scores[0]
        logger.info(
            "route_ranked request_id=%s top_model=%s top_utility=%s ranked_models=%s",
            request_id,
            top.get("model"),
            top.get("utility"),
            ",".join(route_decision.ranked_models[:5]),
        )

    audit_logger: JsonlAuditLogger | None = getattr(app.state, "audit_logger", None)
    if audit_logger is not None:
        audit_logger.log(
            {
                "event": "route_decision",
                "request_id": request_id,
                "path": path,
                "requested_model": route_decision.requested_model,
                "selected_model": route_decision.selected_model,
                "source": route_decision.source,
                "task": route_decision.task,
                "complexity": route_decision.complexity,
                "fallback_models": route_decision.fallback_models,
                "ranked_models": route_decision.ranked_models,
                "signals": route_decision.signals,
                "decision_trace": route_decision.decision_trace,
                "candidate_scores": route_decision.candidate_scores,
                "payload_summary": _build_payload_summary(payload),
            }
        )

    stream = bool(payload.get("stream"))
    return await proxy.forward_with_fallback(
        path=path,
        payload=payload,
        incoming_headers=request.headers,
        route_decision=route_decision,
        stream=stream,
        request_id=request_id,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy_json_request(request, "/v1/chat/completions")


@app.post("/v1/responses")
async def responses(request: Request):
    return await _proxy_json_request(request, "/v1/responses")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy_json_request(request, "/v1/completions")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _proxy_json_request(request, "/v1/embeddings")


@app.post("/v1/images/generations")
async def image_generations(request: Request):
    return await _proxy_json_request(request, "/v1/images/generations")


@app.post("/v1/{subpath:path}")
async def v1_passthrough(subpath: str, request: Request):
    # Catch additional OpenAI-compatible JSON endpoints while preserving routing behavior.
    return await _proxy_json_request(request, f"/v1/{subpath}")


@app.exception_handler(FileNotFoundError)
async def config_missing_handler(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.exception_handler(AuthConfigurationError)
async def auth_config_handler(_: Request, exc: AuthConfigurationError):
    return JSONResponse(status_code=500, content={"error": str(exc)})


def run() -> None:
    import uvicorn

    uvicorn.run("open_llm_router.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
