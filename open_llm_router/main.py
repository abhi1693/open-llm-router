from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from open_llm_router.config import (
    RoutingConfig,
    load_routing_config_with_metadata,
)
from open_llm_router.config.settings import get_settings
from open_llm_router.gateway.audit import JsonlAuditLogger
from open_llm_router.gateway.auth import AuthConfigurationError, Authenticator
from open_llm_router.gateway.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from open_llm_router.gateway.idempotency import (
    IdempotencyBackend,
    IdempotencyConfig,
    build_idempotency_cache_key,
    build_idempotency_store,
)
from open_llm_router.gateway.proxy import BackendProxy
from open_llm_router.routing.classifier import _load_local_embedding_runtime
from open_llm_router.routing.router_engine import (
    InvalidModelError,
    RouteDecision,
    RoutingConstraintError,
    SmartModelRouter,
)
from open_llm_router.runtime.live_metrics import (
    LiveMetricsCollector,
    build_live_metrics_store,
    snapshot_to_dict,
)
from open_llm_router.runtime.policy_updater import (
    RuntimePolicyUpdater,
    apply_runtime_overrides,
)

app = FastAPI(
    title="Open-LLM Router",
    description="OpenAI-compatible API router with model auto-selection and fallback.",
    version="0.1.0",
)

logger = logging.getLogger("uvicorn.error")


@app.middleware("http")
async def auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if not request.url.path.startswith("/v1"):
        return await call_next(request)

    authenticator: Authenticator | None = getattr(app.state, "authenticator", None)
    if authenticator is not None:
        auth_error = await authenticator.authenticate_request(request)
        if auth_error is not None:
            return auth_error

    return await call_next(request)


def _build_models_response(config: RoutingConfig) -> dict[str, Any]:
    no_data_markers = {"n/a", "null", "none", "undefined"}

    def _prune_empty_fields(value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if normalized.lower() in no_data_markers:
                return None
            return normalized
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, raw in value.items():
                normalized = _prune_empty_fields(raw)
                if normalized is None:
                    continue
                if isinstance(normalized, (list, dict)) and not normalized:
                    continue
                cleaned[key] = normalized
            return cleaned
        if isinstance(value, list):
            cleaned_list: list[Any] = []
            for item in value:
                normalized = _prune_empty_fields(item)
                if normalized is None:
                    continue
                if isinstance(normalized, (list, dict)) and not normalized:
                    continue
                cleaned_list.append(normalized)
            return cleaned_list
        return value

    def _format_price(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "0"
        parsed = float(value)
        if parsed <= 0:
            return "0"
        if parsed >= 1:
            text = f"{parsed:.6f}"
        else:
            text = f"{parsed:.12f}"
        return text.rstrip("0").rstrip(".")

    def _provider_for_model(model_id: str, metadata: dict[str, Any]) -> str:
        provider = metadata.get("provider")
        if isinstance(provider, str) and provider.strip():
            return provider.strip()
        if "/" in model_id:
            prefix, _, _ = model_id.partition("/")
            if prefix.strip():
                return prefix.strip()
        return "open-llm-router"

    def _supported_parameters(metadata: dict[str, Any]) -> list[str]:
        raw = metadata.get("supported_parameters")
        if isinstance(raw, list):
            deduped: list[str] = []
            seen: set[str] = set()
            for item in raw:
                if not isinstance(item, str):
                    continue
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                deduped.append(normalized)
            if deduped:
                return deduped

        capabilities = metadata.get("capabilities")
        if not isinstance(capabilities, list):
            return []
        capability_set = {
            item.strip().lower()
            for item in capabilities
            if isinstance(item, str) and item.strip()
        }
        supported = {"max_tokens", "temperature", "top_p"}
        if "tool_use" in capability_set:
            supported.update({"tools", "tool_choice"})
        if "reasoning" in capability_set:
            supported.update({"reasoning", "include_reasoning"})
        if "json_mode" in capability_set:
            supported.update({"structured_outputs", "response_format"})
        return sorted(supported)

    def _architecture(
        model_id: str, metadata: dict[str, Any], supported_parameters: list[str]
    ) -> dict[str, Any]:
        raw = metadata.get("architecture")
        if isinstance(raw, dict):
            input_modalities = raw.get("input_modalities")
            output_modalities = raw.get("output_modalities")
            tokenizer = raw.get("tokenizer")
            instruct_type = raw.get("instruct_type")
            modality = raw.get("modality")
            if not isinstance(input_modalities, list):
                input_modalities = ["text"]
            if not isinstance(output_modalities, list):
                output_modalities = ["text"]
            if not isinstance(tokenizer, str) or not tokenizer.strip():
                tokenizer = "Other"
            if not isinstance(modality, str) or not modality.strip():
                modality = (
                    f"{'+'.join(input_modalities)}->{'+'.join(output_modalities)}"
                )
            if not isinstance(instruct_type, str):
                instruct_type = None
            return {
                "modality": modality,
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "tokenizer": tokenizer,
                "instruct_type": instruct_type,
            }

        capabilities = metadata.get("capabilities")
        capability_values = capabilities if isinstance(capabilities, list) else []
        capability_set = {
            item.strip().lower()
            for item in capability_values
            if isinstance(item, str) and item.strip()
        }
        input_modalities = ["text"]
        if "vision" in capability_set or "image" in capability_set:
            input_modalities.append("image")
        output_modalities = ["text"]
        return {
            "modality": f"{'+'.join(input_modalities)}->{'+'.join(output_modalities)}",
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "tokenizer": "Other",
            "instruct_type": "chat" if "tools" in supported_parameters else None,
        }

    def _pricing(metadata: dict[str, Any]) -> dict[str, str]:
        raw = metadata.get("pricing")
        if isinstance(raw, dict):
            output: dict[str, str] = {}
            for key, value in raw.items():
                if not isinstance(key, str):
                    continue
                output[key] = _format_price(value)
            if output:
                return output

        costs = metadata.get("costs")
        if not isinstance(costs, dict):
            return {"prompt": "0", "completion": "0"}
        prompt_per_1k = costs.get("input_per_1k")
        completion_per_1k = costs.get("output_per_1k")
        prompt = (
            float(prompt_per_1k) / 1000.0
            if isinstance(prompt_per_1k, (int, float))
            else 0.0
        )
        completion = (
            float(completion_per_1k) / 1000.0
            if isinstance(completion_per_1k, (int, float))
            else 0.0
        )
        return {
            "prompt": _format_price(prompt),
            "completion": _format_price(completion),
        }

    def _top_provider(context_length: int, metadata: dict[str, Any]) -> dict[str, Any]:
        raw = metadata.get("top_provider")
        if isinstance(raw, dict):
            value = dict(raw)
            value.setdefault("context_length", context_length)
            value.setdefault("max_completion_tokens", None)
            value.setdefault("is_moderated", False)
            return value

        limits = metadata.get("limits")
        max_completion_tokens: int | None = None
        if isinstance(limits, dict):
            raw_limit = limits.get("max_output_tokens")
            if isinstance(raw_limit, int):
                max_completion_tokens = raw_limit
        return {
            "context_length": context_length,
            "max_completion_tokens": max_completion_tokens,
            "is_moderated": False,
        }

    def _build_model_entry(model_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        provider = _provider_for_model(model_id, metadata)
        canonical_slug = metadata.get("canonical_slug")
        if not isinstance(canonical_slug, str) or not canonical_slug.strip():
            canonical_slug = model_id

        name = metadata.get("name")
        if not isinstance(name, str) or not name.strip():
            name = model_id

        created = metadata.get("created")
        if not isinstance(created, int):
            created = 0

        limits = metadata.get("limits")
        context_length = metadata.get("context_length")
        if not isinstance(context_length, int) and isinstance(limits, dict):
            maybe_context = limits.get("context_tokens")
            if isinstance(maybe_context, int):
                context_length = maybe_context
        if not isinstance(context_length, int):
            context_length = 0

        supported_parameters = _supported_parameters(metadata)
        entry = {
            "id": model_id,
            "object": "model",
            "owned_by": provider,
            "canonical_slug": canonical_slug,
            "name": name,
            "created": created,
            "description": str(metadata.get("description") or ""),
            "context_length": context_length,
            "architecture": _architecture(model_id, metadata, supported_parameters),
            "pricing": _pricing(metadata),
            "top_provider": _top_provider(context_length, metadata),
            "per_request_limits": metadata.get("per_request_limits"),
            "supported_parameters": supported_parameters,
            "default_parameters": (
                metadata.get("default_parameters")
                if isinstance(metadata.get("default_parameters"), dict)
                else {
                    "temperature": None,
                    "top_p": None,
                    "frequency_penalty": None,
                }
            ),
            "hugging_face_id": (
                metadata.get("hugging_face_id")
                if isinstance(metadata.get("hugging_face_id"), str)
                else ""
            ),
            "expiration_date": metadata.get("expiration_date"),
        }
        return cast(dict[str, Any], _prune_empty_fields(entry))

    auto_entry = cast(
        dict[str, Any],
        _prune_empty_fields(
            {
                "id": "auto",
                "object": "model",
                "owned_by": "open-llm-router",
                "canonical_slug": "auto",
                "name": "Auto Router",
                "created": 0,
                "description": "Automatically routes requests across configured models.",
                "context_length": 0,
                "architecture": {
                    "modality": "text->text",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "tokenizer": "Other",
                    "instruct_type": None,
                },
                "pricing": {"prompt": "0", "completion": "0"},
                "top_provider": {
                    "context_length": 0,
                    "max_completion_tokens": None,
                    "is_moderated": False,
                },
                "per_request_limits": None,
                "supported_parameters": [],
                "default_parameters": {
                    "temperature": None,
                    "top_p": None,
                    "frequency_penalty": None,
                },
                "hugging_face_id": "",
                "expiration_date": None,
            }
        ),
    )

    seen: set[str] = {"auto"}
    data: list[dict[str, Any]] = [auto_entry]
    for model_id in config.available_models():
        if model_id in seen:
            continue
        seen.add(model_id)
        metadata = config.models.get(model_id)
        if not isinstance(metadata, dict):
            metadata = {}
        data.append(_build_model_entry(model_id, metadata))

    return {
        "object": "list",
        "data": data,
    }


def _build_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    tools = payload.get("tools")
    functions = payload.get("functions")
    input_value = payload.get("input")

    summary: dict[str, Any] = {
        "stream": bool(payload.get("stream")),
        "has_tools": bool(tools) or bool(functions),
        "tool_choice_type": (
            type(payload.get("tool_choice")).__name__
            if "tool_choice" in payload
            else None
        ),
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


_AUDIT_REDACTED_KEYS = {
    "text_preview",
    "text_preview_total",
    "arguments_preview",
}


def _sanitize_audit_event(event: dict[str, Any]) -> dict[str, Any]:
    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                if key in _AUDIT_REDACTED_KEYS:
                    sanitized[key] = "[redacted]"
                else:
                    sanitized[key] = _sanitize_value(item)
            return sanitized
        if isinstance(value, list):
            return [_sanitize_value(item) for item in value]
        return value

    sanitized_event = _sanitize_value(event)
    if isinstance(sanitized_event, dict):
        return cast(dict[str, Any], sanitized_event)
    return dict(event)


def _extract_error_type_from_response(response: Response) -> str | None:
    if not isinstance(response, JSONResponse):
        return None
    body = bytes(getattr(response, "body", b""))
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    error_type = error.get("type")
    if not isinstance(error_type, str):
        return None
    normalized = error_type.strip()
    return normalized or None


def _emit_proxy_terminal_event(
    *,
    audit_hook: Callable[[dict[str, Any]], None] | None,
    request_id: str,
    path: str,
    stream: bool,
    status: int,
    outcome: str,
    route_decision: RouteDecision | None = None,
    error_type: str | None = None,
    note: str | None = None,
) -> None:
    if audit_hook is None:
        return
    event: dict[str, Any] = {
        "event": "proxy_terminal",
        "request_id": request_id,
        "path": path,
        "stream": stream,
        "status": int(status),
        "outcome": outcome,
    }
    if route_decision is not None:
        event.update(
            {
                "requested_model": route_decision.requested_model,
                "selected_model": route_decision.selected_model,
                "source": route_decision.source,
                "task": route_decision.task,
                "complexity": route_decision.complexity,
            }
        )
    if error_type:
        event["error_type"] = error_type
    if note:
        event["note"] = note
    try:
        audit_hook(event)
    except Exception:
        pass


def _runtime_policy_snapshot(config: RoutingConfig) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for model, profile in sorted(config.model_profiles.items()):
        output[model] = {
            "latency_ms": float(profile.latency_ms),
            "failure_rate": float(profile.failure_rate),
        }
    return output


def _prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _prometheus_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    rendered = ",".join(
        f'{key}="{_prometheus_escape(str(value))}"'
        for key, value in sorted(labels.items())
    )
    return "{" + rendered + "}"


def _append_prometheus_metric(
    lines: list[str],
    declared: set[str],
    *,
    name: str,
    metric_type: str,
    help_text: str,
    value: float | int,
    labels: dict[str, str] | None = None,
) -> None:
    if name not in declared:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
        declared.add(name)
    lines.append(f"{name}{_prometheus_labels(labels)} {float(value):.6f}")


def _render_prometheus_metrics(
    *,
    collector: LiveMetricsCollector | None,
    models_snapshot: dict[str, Any],
) -> str:
    lines: list[str] = []
    declared: set[str] = set()

    if collector is not None:
        _append_prometheus_metric(
            lines,
            declared,
            name="router_live_metrics_events_dropped_total",
            metric_type="counter",
            help_text="Number of live metrics events dropped due to queue pressure.",
            value=collector.dropped_events,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_live_metrics_queue_depth",
            metric_type="gauge",
            help_text="Current live metrics queue depth.",
            value=collector.queue_depth,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_live_metrics_queue_capacity",
            metric_type="gauge",
            help_text="Configured live metrics queue capacity.",
            value=collector.queue_capacity,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_proxy_retries_total",
            metric_type="counter",
            help_text="Total number of upstream proxy retries.",
            value=collector.proxy_retries_total,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_proxy_timeouts_total",
            metric_type="counter",
            help_text="Total number of upstream proxy timeouts.",
            value=collector.proxy_timeouts_total,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_proxy_rate_limited_total",
            metric_type="counter",
            help_text="Total number of upstream 429 rate-limited responses.",
            value=collector.proxy_rate_limited_total,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_proxy_connect_latency_slo_violations_total",
            metric_type="counter",
            help_text=(
                "Total number of connect latency SLO threshold crossings based on "
                "rolling p95 per target."
            ),
            value=collector.proxy_connect_latency_alerts_total,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_proxy_attempt_latency_ms_sum",
            metric_type="counter",
            help_text="Sum of upstream attempt latencies (ms) captured on proxy errors.",
            value=collector.proxy_attempt_latency_sum_ms,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_proxy_attempt_latency_ms_count",
            metric_type="counter",
            help_text="Count of upstream attempt latencies captured on proxy errors.",
            value=collector.proxy_attempt_latency_count,
        )
        for (provider, account), count in sorted(
            collector.proxy_retries_by_target.items()
        ):
            _append_prometheus_metric(
                lines,
                declared,
                name="router_proxy_retries_by_target_total",
                metric_type="counter",
                help_text="Proxy retries partitioned by provider/account target.",
                value=count,
                labels={"provider": provider, "account": account},
            )
        for (provider, account), count in sorted(
            collector.proxy_timeouts_by_target.items()
        ):
            _append_prometheus_metric(
                lines,
                declared,
                name="router_proxy_timeouts_by_target_total",
                metric_type="counter",
                help_text="Proxy timeouts partitioned by provider/account target.",
                value=count,
                labels={"provider": provider, "account": account},
            )
        for (provider, account, model), count in sorted(
            collector.proxy_connect_latency_alerts_by_target.items()
        ):
            _append_prometheus_metric(
                lines,
                declared,
                name="router_proxy_connect_latency_slo_violations_by_target_total",
                metric_type="counter",
                help_text=(
                    "Connect latency SLO threshold crossings by provider/account/model "
                    "(rolling p95)."
                ),
                value=count,
                labels={
                    "provider": provider,
                    "account": account,
                    "model": model,
                },
            )
        for (provider, account, model), values in sorted(
            collector.proxy_connect_latency_quantiles_by_target.items()
        ):
            count = int(values.get("count", 0))
            if count > 0:
                _append_prometheus_metric(
                    lines,
                    declared,
                    name="router_proxy_connect_latency_samples_by_target",
                    metric_type="gauge",
                    help_text="Rolling connect latency sample count by target.",
                    value=count,
                    labels={
                        "provider": provider,
                        "account": account,
                        "model": model,
                    },
                )
            for quantile in ("p50", "p95", "p99"):
                quantile_value = values.get(quantile)
                if not isinstance(quantile_value, (int, float)):
                    continue
                _append_prometheus_metric(
                    lines,
                    declared,
                    name="router_proxy_connect_latency_ms_quantile",
                    metric_type="gauge",
                    help_text=(
                        "Rolling connect latency quantiles by target "
                        "(collector window)."
                    ),
                    value=float(quantile_value),
                    labels={
                        "provider": provider,
                        "account": account,
                        "model": model,
                        "quantile": quantile,
                    },
                )
        for error_type, count in sorted(collector.proxy_errors_by_type.items()):
            _append_prometheus_metric(
                lines,
                declared,
                name="router_proxy_errors_by_type_total",
                metric_type="counter",
                help_text="Proxy request errors partitioned by error type.",
                value=count,
                labels={"error_type": error_type},
            )
        for status_class, count in sorted(
            collector.proxy_responses_by_status_class.items()
        ):
            _append_prometheus_metric(
                lines,
                declared,
                name="router_proxy_responses_by_status_class_total",
                metric_type="counter",
                help_text="Proxy responses partitioned by HTTP status class.",
                value=count,
                labels={"status_class": status_class},
            )

    for model, values in sorted(models_snapshot.items()):
        labels = {"model": model}
        _append_prometheus_metric(
            lines,
            declared,
            name="router_model_samples",
            metric_type="gauge",
            help_text="Live metrics sample count by model.",
            value=values.get("samples", 0),
            labels=labels,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_model_route_decisions_total",
            metric_type="counter",
            help_text="Route decisions observed by model.",
            value=values.get("route_decisions", 0),
            labels=labels,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_model_responses_total",
            metric_type="counter",
            help_text="Upstream responses observed by model.",
            value=values.get("responses", 0),
            labels=labels,
        )
        _append_prometheus_metric(
            lines,
            declared,
            name="router_model_errors_total",
            metric_type="counter",
            help_text="Upstream errors observed by model.",
            value=values.get("errors", 0),
            labels=labels,
        )
        for metric_name, metric_key in (
            ("router_model_ewma_connect_ms", "ewma_connect_ms"),
            ("router_model_ewma_request_latency_ms", "ewma_request_latency_ms"),
            ("router_model_ewma_failure_rate", "ewma_failure_rate"),
        ):
            value = values.get(metric_key)
            if isinstance(value, (int, float)):
                _append_prometheus_metric(
                    lines,
                    declared,
                    name=metric_name,
                    metric_type="gauge",
                    help_text=f"{metric_key} by model.",
                    value=value,
                    labels=labels,
                )

    return "\n".join(lines) + "\n"


def _setup_optional_tracing(
    *,
    app_obj: FastAPI,
    proxy: BackendProxy,
    settings: Any,
) -> None:
    if not bool(getattr(settings, "observability_tracing_enabled", False)):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:
        logger.warning("observability_tracing_unavailable reason=%s", str(exc))
        return

    service_name = str(
        getattr(settings, "observability_service_name", "open-llm-router")
    )
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = getattr(settings, "observability_otlp_endpoint", None)
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
        except Exception as exc:
            logger.warning(
                "observability_otlp_exporter_unavailable reason=%s", str(exc)
            )
    trace.set_tracer_provider(provider)

    try:
        FastAPIInstrumentor.instrument_app(app_obj)
    except Exception as exc:
        logger.warning(
            "observability_fastapi_instrumentation_failed reason=%s", str(exc)
        )
    try:
        HTTPXClientInstrumentor().instrument_client(proxy.client)
    except Exception as exc:
        logger.warning("observability_httpx_instrumentation_failed reason=%s", str(exc))


def _prefetch_local_semantic_classifier_model(routing_config: RoutingConfig) -> None:
    semantic_cfg = routing_config.semantic_classifier
    if not bool(semantic_cfg.enabled):
        return
    if semantic_cfg.backend != "local_embedding":
        return

    model_name = str(semantic_cfg.local_model_name or "").strip()
    if not model_name:
        return

    local_files_only = bool(semantic_cfg.local_files_only)
    logger.info(
        "semantic_classifier_prefetch_start model=%s local_files_only=%s",
        model_name,
        local_files_only,
    )
    runtime = _load_local_embedding_runtime(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    if runtime is None:
        logger.warning(
            "semantic_classifier_prefetch_failed model=%s local_files_only=%s",
            model_name,
            local_files_only,
        )
        return
    logger.info(
        "semantic_classifier_prefetch_complete model=%s local_files_only=%s",
        model_name,
        local_files_only,
    )


def _prefetch_local_route_reranker_model(routing_config: RoutingConfig) -> None:
    reranker_cfg = routing_config.route_reranker
    if not bool(reranker_cfg.enabled):
        return
    if reranker_cfg.backend != "local_embedding":
        return

    model_name = str(reranker_cfg.local_model_name or "").strip()
    if not model_name:
        return

    local_files_only = bool(reranker_cfg.local_files_only)
    logger.info(
        "route_reranker_prefetch_start model=%s local_files_only=%s",
        model_name,
        local_files_only,
    )
    runtime = _load_local_embedding_runtime(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    if runtime is None:
        logger.warning(
            "route_reranker_prefetch_failed model=%s local_files_only=%s",
            model_name,
            local_files_only,
        )
        return
    logger.info(
        "route_reranker_prefetch_complete model=%s local_files_only=%s",
        model_name,
        local_files_only,
    )


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
    await asyncio.to_thread(_prefetch_local_semantic_classifier_model, routing_config)
    await asyncio.to_thread(_prefetch_local_route_reranker_model, routing_config)
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
    app.state.audit_payload_summary_enabled = bool(audit_logger.enabled)
    app.state.audit_safe_logging_enabled = bool(
        settings.router_audit_safe_logging_enabled
    )
    live_metrics_store = build_live_metrics_store(
        redis_url=settings.redis_url,
        logger=logger,
        alpha=settings.live_metrics_ewma_alpha,
    )
    live_metrics_collector = LiveMetricsCollector(
        store=live_metrics_store,
        logger=logger,
        enabled=settings.live_metrics_enabled,
        connect_latency_window_size=settings.live_metrics_connect_latency_window_size,
        connect_latency_alert_threshold_ms=settings.live_metrics_connect_latency_alert_threshold_ms,
    )
    await live_metrics_collector.start()
    app.state.live_metrics_store = live_metrics_store
    app.state.live_metrics_collector = live_metrics_collector

    def audit_event_hook(event: dict[str, Any]) -> None:
        event_payload = event
        if bool(getattr(app.state, "audit_safe_logging_enabled", True)):
            event_payload = _sanitize_audit_event(event)
        audit_logger.log(event_payload)
        live_metrics_collector.ingest(event_payload)

    app.state.audit_event_hook = audit_event_hook
    app.state.circuit_breakers = CircuitBreakerRegistry(
        CircuitBreakerConfig(
            enabled=settings.circuit_breaker_enabled,
            failure_threshold=max(1, settings.circuit_breaker_failure_threshold),
            recovery_timeout_seconds=max(
                1.0, settings.circuit_breaker_recovery_timeout_seconds
            ),
            half_open_max_requests=max(
                1, settings.circuit_breaker_half_open_max_requests
            ),
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
        oauth_state_persistence_path=settings.routing_config_path,
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
        classifier_metrics_provider=live_metrics_collector,
    )
    await policy_updater.start()
    app.state.policy_updater = policy_updater
    _setup_optional_tracing(
        app_obj=app, proxy=app.state.backend_proxy, settings=settings
    )
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
    policy_updater: RuntimePolicyUpdater | None = getattr(
        app.state, "policy_updater", None
    )
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
    collector: LiveMetricsCollector | None = getattr(
        app.state, "live_metrics_collector", None
    )
    if collector is None:
        raise HTTPException(
            status_code=503, detail="Live metrics collector is unavailable."
        )
    snapshot = await collector.snapshot()
    connect_latency_quantiles = [
        {
            "provider": provider,
            "account": account,
            "model": model,
            "count": int(values.get("count", 0)),
            "p50_ms": float(values.get("p50", 0.0)),
            "p95_ms": float(values.get("p95", 0.0)),
            "p99_ms": float(values.get("p99", 0.0)),
        }
        for (provider, account, model), values in sorted(
            collector.proxy_connect_latency_quantiles_by_target.items()
        )
    ]
    connect_latency_alerts = [
        {
            "provider": provider,
            "account": account,
            "model": model,
            "alerts": int(count),
        }
        for (provider, account, model), count in sorted(
            collector.proxy_connect_latency_alerts_by_target.items()
        )
    ]
    return {
        "object": "router.live_metrics",
        "dropped_events": collector.dropped_events,
        "queue_depth": collector.queue_depth,
        "queue_capacity": collector.queue_capacity,
        "proxy_retries_total": collector.proxy_retries_total,
        "proxy_timeouts_total": collector.proxy_timeouts_total,
        "proxy_rate_limited_total": collector.proxy_rate_limited_total,
        "proxy_connect_latency_window_size": collector.proxy_connect_latency_window_size,
        "proxy_connect_latency_alert_threshold_ms": collector.proxy_connect_latency_alert_threshold_ms,
        "proxy_connect_latency_slo_violations_total": collector.proxy_connect_latency_alerts_total,
        "proxy_connect_latency_slo_violations_by_target": connect_latency_alerts,
        "proxy_connect_latency_quantiles_by_target": connect_latency_quantiles,
        "proxy_attempt_latency_ms_sum": collector.proxy_attempt_latency_sum_ms,
        "proxy_attempt_latency_ms_count": collector.proxy_attempt_latency_count,
        "proxy_errors_by_type": collector.proxy_errors_by_type,
        "proxy_responses_by_status_class": collector.proxy_responses_by_status_class,
        "models": snapshot_to_dict(snapshot),
    }


@app.get("/v1/router/policy")
async def router_policy() -> dict[str, Any]:
    policy_updater: RuntimePolicyUpdater | None = getattr(
        app.state, "policy_updater", None
    )
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
            "last_classifier_samples": status_obj.last_classifier_samples,
            "last_classifier_success_rate": status_obj.last_classifier_success_rate,
            "last_classifier_adjusted": status_obj.last_classifier_adjusted,
            "classifier_adjustment_history": status_obj.classifier_adjustment_history,
        }
    classifier_calibration = config.classifier_calibration
    return {
        "object": "router.policy",
        "updater": status_payload,
        "model_profiles": _runtime_policy_snapshot(config),
        "classifier_calibration": {
            "enabled": classifier_calibration.enabled,
            "min_samples": classifier_calibration.min_samples,
            "target_secondary_success_rate": (
                classifier_calibration.target_secondary_success_rate
            ),
            "secondary_low_confidence_min_confidence": (
                classifier_calibration.secondary_low_confidence_min_confidence
            ),
            "secondary_mixed_signal_min_confidence": (
                classifier_calibration.secondary_mixed_signal_min_confidence
            ),
            "adjustment_step": classifier_calibration.adjustment_step,
            "deadband": classifier_calibration.deadband,
            "min_threshold": classifier_calibration.min_threshold,
            "max_threshold": classifier_calibration.max_threshold,
        },
    }


async def _proxy_json_request(request: Request, path: str) -> Response:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Expected JSON body: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Expected a JSON object request body."
        )

    router: SmartModelRouter = app.state.smart_router
    proxy: BackendProxy = app.state.backend_proxy
    idempotency_store: IdempotencyBackend = app.state.idempotency_store

    request_id = (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or uuid4().hex[:12]
    )
    audit_hook: Callable[[dict[str, Any]], None] | None = getattr(
        app.state, "audit_event_hook", None
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
        if (
            idempotency_result.mode == "replay"
            and idempotency_result.cached is not None
        ):
            cached_response = idempotency_result.cached.to_fastapi_response()
            _emit_proxy_terminal_event(
                audit_hook=audit_hook,
                request_id=request_id,
                path=path,
                stream=is_stream,
                status=int(cached_response.status_code),
                outcome=(
                    "success" if int(cached_response.status_code) < 400 else "error"
                ),
                error_type=_extract_error_type_from_response(cached_response),
                note="idempotency_replay",
            )
            return cached_response
        if idempotency_result.mode == "wait":
            cached = await idempotency_store.wait_for_existing(idempotency_result)
            if cached is not None:
                cached_response = cached.to_fastapi_response()
                _emit_proxy_terminal_event(
                    audit_hook=audit_hook,
                    request_id=request_id,
                    path=path,
                    stream=is_stream,
                    status=int(cached_response.status_code),
                    outcome=(
                        "success" if int(cached_response.status_code) < 400 else "error"
                    ),
                    error_type=_extract_error_type_from_response(cached_response),
                    note="idempotency_wait_replay",
                )
                return cached_response
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
        _emit_proxy_terminal_event(
            audit_hook=audit_hook,
            request_id=request_id,
            path=path,
            stream=is_stream,
            status=400,
            outcome="error",
            error_type="invalid_model",
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
        _emit_proxy_terminal_event(
            audit_hook=audit_hook,
            request_id=request_id,
            path=path,
            stream=is_stream,
            status=400,
            outcome="error",
            error_type="routing_constraints_unsatisfied",
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
            "requested_model=%s selected_model=%s fallback_count=%d task_reason=%s "
            "latest_override=%s latest_factual=%s latest_text_len=%s "
            "routing_mode=%s selected_reason=%s"
        ),
        request_id,
        path,
        route_decision.source,
        route_decision.task,
        route_decision.complexity,
        route_decision.requested_model,
        route_decision.selected_model,
        len(route_decision.fallback_models),
        route_decision.signals.get("task_reason"),
        route_decision.signals.get("latest_turn_override_applied"),
        route_decision.signals.get("latest_user_factual_query"),
        route_decision.signals.get("latest_user_text_length"),
        route_decision.signals.get("routing_mode"),
        route_decision.decision_trace.get("selected_reason"),
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

    if audit_hook is not None:
        event: dict[str, Any] = {
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
        }
        if bool(getattr(app.state, "audit_payload_summary_enabled", False)):
            event["payload_summary"] = _build_payload_summary(payload)
        audit_hook(event)

    try:
        response: Response = await proxy.forward_with_fallback(
            path=path,
            payload=payload,
            incoming_headers=request.headers,
            route_decision=route_decision,
            stream=is_stream,
            request_id=request_id,
        )
    except asyncio.CancelledError:
        if cache_key and is_leader:
            await idempotency_store.release_without_store(cache_key)
        _emit_proxy_terminal_event(
            audit_hook=audit_hook,
            request_id=request_id,
            path=path,
            stream=is_stream,
            status=499,
            outcome="cancelled",
            route_decision=route_decision,
            error_type="request_cancelled",
        )
        raise
    except Exception as exc:
        if cache_key and is_leader:
            # If upstream call crashes before we can cache a final response,
            # unblock waiters so they can retry.
            await idempotency_store.release_without_store(cache_key)
        _emit_proxy_terminal_event(
            audit_hook=audit_hook,
            request_id=request_id,
            path=path,
            stream=is_stream,
            status=500,
            outcome="error",
            route_decision=route_decision,
            error_type=exc.__class__.__name__,
        )
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
                    if key.lower()
                    in {"content-type", "x-router-model", "x-router-provider"}
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

    terminal_error_type = _extract_error_type_from_response(response)
    terminal_outcome = "success"
    if terminal_error_type == "routing_exhausted":
        terminal_outcome = "exhausted"
    elif int(response.status_code) >= 400:
        terminal_outcome = "error"
    _emit_proxy_terminal_event(
        audit_hook=audit_hook,
        request_id=request_id,
        path=path,
        stream=is_stream,
        status=int(response.status_code),
        outcome=terminal_outcome,
        route_decision=route_decision,
        error_type=terminal_error_type,
    )

    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _proxy_json_request(request, "/v1/chat/completions")


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    return await _proxy_json_request(request, "/v1/responses")


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _proxy_json_request(request, "/v1/completions")


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    return await _proxy_json_request(request, "/v1/embeddings")


@app.post("/v1/images/generations")
async def image_generations(request: Request) -> Response:
    return await _proxy_json_request(request, "/v1/images/generations")


@app.post("/v1/{subpath:path}")
async def v1_passthrough(subpath: str, request: Request) -> Response:
    # Catch additional OpenAI-compatible JSON endpoints while preserving routing behavior.
    return await _proxy_json_request(request, f"/v1/{subpath}")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    settings = get_settings()
    if not settings.observability_metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics endpoint is disabled.")

    collector: LiveMetricsCollector | None = getattr(
        app.state, "live_metrics_collector", None
    )
    models_snapshot: dict[str, Any] = {}
    if collector is not None:
        models_snapshot = snapshot_to_dict(await collector.snapshot())
    payload = _render_prometheus_metrics(
        collector=collector,
        models_snapshot=models_snapshot,
    )
    return PlainTextResponse(
        content=payload,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.exception_handler(FileNotFoundError)
async def config_missing_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.exception_handler(AuthConfigurationError)
async def auth_config_handler(_: Request, exc: AuthConfigurationError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


def run() -> None:
    import uvicorn

    uvicorn.run("open_llm_router.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
