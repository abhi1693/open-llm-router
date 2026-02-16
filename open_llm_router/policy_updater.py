from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from open_llm_router.config import ModelProfile, RoutingConfig
from open_llm_router.live_metrics import (
    LiveMetricsStore,
    ModelMetricsSnapshot,
    is_target_metrics_key,
    snapshot_to_dict,
)


@dataclass(slots=True)
class RuntimePolicyUpdaterStatus:
    enabled: bool
    interval_seconds: float
    min_samples: int
    last_run_epoch: float | None = None
    last_applied_models: int = 0
    last_error: str | None = None


class RuntimePolicyUpdater:
    def __init__(
        self,
        *,
        routing_config: RoutingConfig,
        metrics_store: LiveMetricsStore,
        logger: logging.Logger | None = None,
        enabled: bool = True,
        interval_seconds: float = 30.0,
        min_samples: int = 30,
        max_adjustment_ratio: float = 0.15,
        overrides_path: str | None = None,
    ) -> None:
        self._routing_config = routing_config
        self._metrics_store = metrics_store
        self._logger = logger
        self._enabled = enabled
        self._interval_seconds = max(1.0, float(interval_seconds))
        self._min_samples = max(1, int(min_samples))
        self._max_adjustment_ratio = max(0.01, min(1.0, float(max_adjustment_ratio)))
        self._overrides_path = Path(overrides_path) if overrides_path else None

        self._task: asyncio.Task[None] | None = None
        self._status = RuntimePolicyUpdaterStatus(
            enabled=enabled,
            interval_seconds=self._interval_seconds,
            min_samples=self._min_samples,
        )

    @property
    def status(self) -> RuntimePolicyUpdaterStatus:
        return self._status

    async def start(self) -> None:
        if not self._enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="runtime-policy-updater")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def run_once(self) -> int:
        snapshot = await self._metrics_store.snapshot_all()
        applied = self._apply_snapshot(snapshot)
        if self._overrides_path:
            _write_runtime_overrides(
                path=self._overrides_path,
                routing_config=self._routing_config,
                snapshot=snapshot,
            )
        self._status.last_run_epoch = time.time()
        self._status.last_applied_models = applied
        self._status.last_error = None
        return applied

    async def _run(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                self._status.last_run_epoch = time.time()
                self._status.last_error = str(exc)
                if self._logger is not None:
                    self._logger.warning("runtime_policy_update_failed error=%s", str(exc))
            await asyncio.sleep(self._interval_seconds)

    def _apply_snapshot(self, snapshot: dict[str, ModelMetricsSnapshot]) -> int:
        applied = 0
        for model, metrics in snapshot.items():
            if is_target_metrics_key(model):
                continue
            if metrics.samples < self._min_samples:
                continue

            profile = self._routing_config.model_profiles.get(model)
            if profile is None:
                profile = ModelProfile()

            changed = False
            observed_latency = metrics.ewma_request_latency_ms or metrics.ewma_connect_ms
            if observed_latency is not None and observed_latency > 0:
                baseline = float(profile.latency_ms) if profile.latency_ms > 0 else float(observed_latency)
                updated_latency = _bounded_adjust(
                    current=baseline,
                    target=float(observed_latency),
                    max_ratio=self._max_adjustment_ratio,
                )
                if abs(updated_latency - baseline) >= 1e-6:
                    profile.latency_ms = updated_latency
                    changed = True

            if metrics.ewma_failure_rate is not None:
                bounded_target = min(max(float(metrics.ewma_failure_rate), 0.0), 1.0)
                baseline_failure = float(profile.failure_rate)
                if baseline_failure <= 0:
                    updated_failure = bounded_target
                else:
                    updated_failure = _bounded_adjust(
                        current=baseline_failure,
                        target=bounded_target,
                        max_ratio=self._max_adjustment_ratio,
                    )
                updated_failure = min(max(updated_failure, 0.0), 1.0)
                if abs(updated_failure - baseline_failure) >= 1e-9:
                    profile.failure_rate = updated_failure
                    changed = True

            if changed:
                self._routing_config.model_profiles[model] = profile
                applied += 1

        if applied and self._logger is not None:
            self._logger.info("runtime_policy_update_applied models=%d", applied)
        return applied


def _bounded_adjust(current: float, target: float, max_ratio: float) -> float:
    if current <= 0:
        return target
    upper = current * (1.0 + max_ratio)
    lower = max(0.0, current * (1.0 - max_ratio))
    return min(max(target, lower), upper)


def apply_runtime_overrides(
    *,
    path: str | None,
    routing_config: RoutingConfig,
    logger: logging.Logger | None = None,
) -> int:
    if not path:
        return 0
    file_path = Path(path)
    if not file_path.exists():
        return 0

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception as exc:
        if logger is not None:
            logger.warning("runtime_overrides_load_failed path=%s error=%s", file_path, str(exc))
        return 0

    overrides = raw.get("model_profiles")
    if not isinstance(overrides, dict):
        return 0

    applied = 0
    for model, fields in overrides.items():
        if not isinstance(model, str) or not isinstance(fields, dict):
            continue
        profile = routing_config.model_profiles.get(model) or ModelProfile()
        changed = False
        latency_ms = fields.get("latency_ms")
        failure_rate = fields.get("failure_rate")
        if isinstance(latency_ms, (int, float)) and float(latency_ms) > 0:
            profile.latency_ms = float(latency_ms)
            changed = True
        if isinstance(failure_rate, (int, float)):
            profile.failure_rate = min(max(float(failure_rate), 0.0), 1.0)
            changed = True
        if changed:
            routing_config.model_profiles[model] = profile
            applied += 1

    if applied and logger is not None:
        logger.info("runtime_overrides_applied path=%s models=%d", file_path, applied)
    return applied


def _write_runtime_overrides(
    *,
    path: Path,
    routing_config: RoutingConfig,
    snapshot: dict[str, ModelMetricsSnapshot],
) -> None:
    payload: dict[str, Any] = {
        "generated_at_epoch": time.time(),
        "model_profiles": {},
        "metrics_snapshot": snapshot_to_dict(snapshot),
    }

    for model, profile in sorted(routing_config.model_profiles.items()):
        payload["model_profiles"][model] = {
            "latency_ms": float(profile.latency_ms),
            "failure_rate": float(profile.failure_rate),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
