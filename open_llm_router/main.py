from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from open_llm_router.audit import JsonlAuditLogger
from open_llm_router.auth import AuthConfigurationError, Authenticator
from open_llm_router.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry
from open_llm_router.config import (
    RoutingConfig,
    load_routing_config_with_metadata,
)
from open_llm_router.idempotency import (
    IdempotencyBackend,
    IdempotencyConfig,
    build_idempotency_store,
    build_idempotency_cache_key,
)
from open_llm_router.live_metrics import (
    LiveMetricsCollector,
    build_live_metrics_store,
    snapshot_to_dict,
)
from open_llm_router.policy_updater import (
    RuntimePolicyUpdater,
    apply_runtime_overrides,
)
from open_llm_router.proxy import BackendProxy
from open_llm_router.router_engine import (
    InvalidModelError,
    RoutingConstraintError,
    SmartModelRouter,
)
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


def _runtime_policy_snapshot(config: RoutingConfig) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for model, profile in sorted(config.model_profiles.items()):
        output[model] = {
            "latency_ms": float(profile.latency_ms),
            "failure_rate": float(profile.failure_rate),
        }
    return output


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    routing_config, explain_metadata = load_routing_config_with_metadata(
        settings.routing_config_path
    )
    apply_runtime_overrides(
        path=settings.router_runtime_overrides_path,
        routing_config=routing_config,
        logger=logger,
    )
    authenticator = Authenticator(settings)
    app.state.settings = settings
    app.state.authenticator = authenticator
    app.state.routing_config = routing_config
    app.state.routing_config_explain = explain_metadata
    app.state.smart_router = SmartModelRouter(routing_config)
    audit_logger = JsonlAuditLogger(
        path=settings.router_audit_log_path,
        enabled=settings.router_audit_log_enabled,
    )
    app.state.audit_logger = audit_logger
    live_metrics_store = build_live_metrics_store(
        redis_url=settings.redis_url,
        logger=logger,
        alpha=settings.live_metrics_ewma_alpha,
    )
    live_metrics_collector = LiveMetricsCollector(
        store=live_metrics_store,
        logger=logger,
        enabled=settings.live_metrics_enabled,
    )
    await live_metrics_collector.start()
    app.state.live_metrics_store = live_metrics_store
    app.state.live_metrics_collector = live_metrics_collector

    def audit_event_hook(event: dict[str, Any]) -> None:
        audit_logger.log(event)
        live_metrics_collector.ingest(event)

    app.state.audit_event_hook = audit_event_hook
    app.state.circuit_breakers = CircuitBreakerRegistry(
        CircuitBreakerConfig(
            enabled=settings.circuit_breaker_enabled,
            failure_threshold=max(1, settings.circuit_breaker_failure_threshold),
            recovery_timeout_seconds=max(1.0, settings.circuit_breaker_recovery_timeout_seconds),
            half_open_max_requests=max(1, settings.circuit_breaker_half_open_max_requests),
        )
    )
    app.state.idempotency_store = build_idempotency_store(
        config=IdempotencyConfig(
            enabled=settings.idempotency_enabled,
            ttl_seconds=max(1, settings.idempotency_ttl_seconds),
            wait_timeout_seconds=max(0.1, settings.idempotency_wait_timeout_seconds),
        ),
        redis_url=settings.redis_url,
        logger=logger,
    )
    app.state.backend_proxy = BackendProxy(
        base_url=settings.backend_base_url,
        timeout_seconds=settings.backend_timeout_seconds,
        connect_timeout_seconds=settings.backend_connect_timeout_seconds,
        read_timeout_seconds=settings.backend_read_timeout_seconds,
        write_timeout_seconds=settings.backend_write_timeout_seconds,
        pool_timeout_seconds=settings.backend_pool_timeout_seconds,
        backend_api_key=settings.backend_api_key,
        retry_statuses=routing_config.retry_statuses,
        accounts=routing_config.accounts,
        model_registry=routing_config.models,
        audit_hook=audit_event_hook,
        circuit_breakers=app.state.circuit_breakers,
    )
    policy_updater = RuntimePolicyUpdater(
        routing_config=routing_config,
        metrics_store=live_metrics_store,
        logger=logger,
        enabled=settings.live_metrics_enabled,
        interval_seconds=settings.live_metrics_update_interval_seconds,
        min_samples=settings.live_metrics_min_samples,
        max_adjustment_ratio=settings.runtime_policy_max_adjustment_ratio,
        overrides_path=settings.router_runtime_overrides_path,
    )
    await policy_updater.start()
    app.state.policy_updater = policy_updater
    logger.info(
        (
            "startup complete routing_config_path=%s accounts=%d models=%d default_model=%s "
            "audit_log_enabled=%s audit_log_path=%s live_metrics_enabled=%s"
        ),
        settings.routing_config_path,
        len(routing_config.accounts),
        len(routing_config.available_models()),
        routing_config.default_model,
        settings.router_audit_log_enabled,
        settings.router_audit_log_path,
        settings.live_metrics_enabled,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    policy_updater: RuntimePolicyUpdater | None = getattr(app.state, "policy_updater", None)
    if policy_updater is not None:
        await policy_updater.stop()
    live_metrics_collector: LiveMetricsCollector | None = getattr(
        app.state, "live_metrics_collector", None
    )
    if live_metrics_collector is not None:
        await live_metrics_collector.close()
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


@app.get("/v1/router/live-metrics")
async def router_live_metrics() -> dict[str, Any]:
    collector: LiveMetricsCollector | None = getattr(app.state, "live_metrics_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="Live metrics collector is unavailable.")
    snapshot = await collector.snapshot()
    return {
        "object": "router.live_metrics",
        "dropped_events": collector.dropped_events,
        "models": snapshot_to_dict(snapshot),
    }


@app.get("/v1/router/policy")
async def router_policy() -> dict[str, Any]:
    policy_updater: RuntimePolicyUpdater | None = getattr(app.state, "policy_updater", None)
    config: RoutingConfig = app.state.routing_config
    status_payload: dict[str, Any] = {}
    if policy_updater is not None:
        status_obj = policy_updater.status
        status_payload = {
            "enabled": status_obj.enabled,
            "interval_seconds": status_obj.interval_seconds,
            "min_samples": status_obj.min_samples,
            "last_run_epoch": status_obj.last_run_epoch,
            "last_applied_models": status_obj.last_applied_models,
            "last_error": status_obj.last_error,
        }
    return {
        "object": "router.policy",
        "updater": status_payload,
        "model_profiles": _runtime_policy_snapshot(config),
    }


async def _proxy_json_request(request: Request, path: str):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Expected JSON body: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object request body.")

    router: SmartModelRouter = app.state.smart_router
    proxy: BackendProxy = app.state.backend_proxy
    idempotency_store: IdempotencyBackend = app.state.idempotency_store

    request_id = (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or uuid4().hex[:12]
    )
    is_stream = bool(payload.get("stream"))
    idempotency_key = request.headers.get("Idempotency-Key")
    idempotency_result = None
    cache_key: str | None = None
    is_leader = False
    if idempotency_key and not is_stream:
        tenant_id = (
            request.headers.get("OpenAI-Organization")
            or request.headers.get("X-Tenant-Id")
            or "default"
        )
        cache_key = build_idempotency_cache_key(
            idempotency_key=idempotency_key.strip(),
            tenant_id=tenant_id.strip(),
            path=path,
            payload=payload,
        )
        idempotency_result = await idempotency_store.begin(cache_key)
        if idempotency_result.mode == "replay" and idempotency_result.cached is not None:
            return idempotency_result.cached.to_fastapi_response()
        if idempotency_result.mode == "wait":
            cached = await idempotency_store.wait_for_existing(idempotency_result)
            if cached is not None:
                return cached.to_fastapi_response()
        is_leader = idempotency_result.mode == "leader"
    try:
        route_decision = router.decide(payload=payload, endpoint=path)
    except InvalidModelError as exc:
        if cache_key and is_leader:
            await idempotency_store.release_without_store(cache_key)
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
    except RoutingConstraintError as exc:
        if cache_key and is_leader:
            await idempotency_store.release_without_store(cache_key)
        logger.info(
            "routing_constraints_unsatisfied request_id=%s constraint=%s",
            request_id,
            exc.constraint,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "type": "routing_constraints_unsatisfied",
                "message": str(exc),
                "constraint": exc.constraint,
                "details": exc.details,
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

    audit_hook: Callable[[dict[str, Any]], None] | None = getattr(
        app.state, "audit_event_hook", None
    )
    if audit_hook is not None:
        audit_hook(
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

    try:
        response: Response = await proxy.forward_with_fallback(
            path=path,
            payload=payload,
            incoming_headers=request.headers,
            route_decision=route_decision,
            stream=is_stream,
            request_id=request_id,
        )
    except Exception:
        if cache_key and is_leader:
            # If upstream call crashes before we can cache a final response,
            # unblock waiters so they can retry.
            await idempotency_store.release_without_store(cache_key)
        raise

    if cache_key and is_leader:
        try:
            if isinstance(response, StreamingResponse):
                await idempotency_store.release_without_store(cache_key)
            else:
                response_body = bytes(getattr(response, "body", b""))
                cache_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() in {"content-type", "x-router-model", "x-router-provider"}
                }
                await idempotency_store.store(
                    key=cache_key,
                    status_code=response.status_code,
                    headers=cache_headers,
                    body=response_body,
                )
                response.headers["x-router-idempotency-status"] = "stored"
        except Exception:
            await idempotency_store.release_without_store(cache_key)
            raise

    return response


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
