from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_llm_router.config import ModelProfile, RoutingConfig
from open_llm_router.live_metrics import (
    LiveMetricsStore,
    ModelMetricsSnapshot,
    is_target_metrics_key,
    snapshot_to_dict,
)
from open_llm_router.route_decision_tracker import ClassifierCalibrationSnapshot
from open_llm_router.utils.yaml_utils import load_yaml_dict, write_yaml_dict


@dataclass(slots=True)
class RuntimePolicyUpdaterStatus:
    enabled: bool
    interval_seconds: float
    min_samples: int
    last_run_epoch: float | None = None
    last_applied_models: int = 0
    last_error: str | None = None
    last_classifier_samples: int = 0
    last_classifier_success_rate: float | None = None
    last_classifier_adjusted: bool = False
    last_feature_weight_adjusted_count: int = 0
    last_feature_weight_scale: float | None = None
    classifier_adjustment_history: list[dict[str, Any]] = field(default_factory=list)


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
        classifier_metrics_provider: Any | None = None,
    ) -> None:
        self._routing_config = routing_config
        self._metrics_store = metrics_store
        self._logger = logger
        self._enabled = enabled
        self._interval_seconds = max(1.0, float(interval_seconds))
        self._min_samples = max(1, int(min_samples))
        self._max_adjustment_ratio = max(0.01, min(1.0, float(max_adjustment_ratio)))
        self._overrides_path = Path(overrides_path) if overrides_path else None
        self._classifier_metrics_provider = classifier_metrics_provider
        self._last_classifier_totals = {
            "secondary_total": 0,
            "secondary_success": 0,
            "secondary_non_success": 0,
        }
        self._classifier_adjustment_history: list[dict[str, Any]] = []
        self._classifier_adjustment_history_limit = 50
        self._baseline_model_profiles = {
            model: profile.model_copy(deep=True)
            for model, profile in routing_config.model_profiles.items()
        }
        self._baseline_feature_weights = {
            name: float(value)
            for name, value in routing_config.learned_routing.feature_weights.items()
            if isinstance(value, (int, float))
        }

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
        (
            feature_weight_adjusted_count,
            feature_weight_scale,
        ) = self._apply_learned_feature_weight_adaptation(snapshot)
        classifier_adjusted = self._apply_classifier_calibration()
        if self._overrides_path:
            _write_runtime_overrides(
                path=self._overrides_path,
                routing_config=self._routing_config,
                snapshot=snapshot,
                classifier_calibration=self._classifier_calibration_overrides_payload(),
            )
        self._status.last_run_epoch = time.time()
        self._status.last_applied_models = applied
        self._status.last_feature_weight_adjusted_count = feature_weight_adjusted_count
        self._status.last_feature_weight_scale = feature_weight_scale
        self._status.last_classifier_adjusted = classifier_adjusted
        self._status.classifier_adjustment_history = list(
            self._classifier_adjustment_history
        )
        self._status.last_error = None
        return applied + feature_weight_adjusted_count

    async def _run(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                self._status.last_run_epoch = time.time()
                self._status.last_error = str(exc)
                if self._logger is not None:
                    self._logger.warning(
                        "runtime_policy_update_failed error=%s", str(exc)
                    )
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
            observed_latency = (
                metrics.ewma_request_latency_ms or metrics.ewma_connect_ms
            )
            if observed_latency is not None and observed_latency > 0:
                baseline = (
                    float(profile.latency_ms)
                    if profile.latency_ms > 0
                    else float(observed_latency)
                )
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

    def _apply_learned_feature_weight_adaptation(
        self,
        snapshot: dict[str, ModelMetricsSnapshot],
    ) -> tuple[int, float | None]:
        learned_cfg = self._routing_config.learned_routing
        if not bool(learned_cfg.enabled):
            return 0, None
        if not isinstance(learned_cfg.feature_weights, dict):
            return 0, None

        weighted_failures = 0.0
        weighted_latency_ratios = 0.0
        total_weight = 0.0

        for model, metrics in snapshot.items():
            if is_target_metrics_key(model):
                continue
            if metrics.samples < self._min_samples:
                continue

            weight = float(max(metrics.samples, metrics.route_decisions, 1))
            observed_failure = (
                float(metrics.ewma_failure_rate)
                if metrics.ewma_failure_rate is not None
                else float(
                    self._routing_config.model_profiles.get(
                        model, ModelProfile()
                    ).failure_rate
                )
            )
            observed_failure = min(max(observed_failure, 0.0), 1.0)

            observed_latency = (
                metrics.ewma_request_latency_ms or metrics.ewma_connect_ms
            )
            baseline_profile = self._baseline_model_profiles.get(model)
            baseline_latency = (
                float(baseline_profile.latency_ms)
                if baseline_profile is not None and baseline_profile.latency_ms > 0
                else float(
                    self._routing_config.model_profiles.get(
                        model, ModelProfile()
                    ).latency_ms
                )
            )
            latency_ratio = 1.0
            if (
                observed_latency is not None
                and observed_latency > 0
                and baseline_latency > 0
            ):
                latency_ratio = float(observed_latency) / baseline_latency
                latency_ratio = min(max(latency_ratio, 0.1), 5.0)

            weighted_failures += observed_failure * weight
            weighted_latency_ratios += latency_ratio * weight
            total_weight += weight

        if total_weight <= 0:
            return 0, None

        avg_failure = weighted_failures / total_weight
        avg_latency_ratio = weighted_latency_ratios / total_weight
        feature_scale = _feature_weight_scale_for_stability(
            avg_failure=avg_failure,
            avg_latency_ratio=avg_latency_ratio,
        )

        adjusted = 0
        for name, raw_current in list(learned_cfg.feature_weights.items()):
            if not isinstance(raw_current, (int, float)):
                continue
            if name.startswith("task_"):
                continue

            current = float(raw_current)
            baseline = self._baseline_feature_weights.get(name, current)
            target = baseline * feature_scale
            updated = _bounded_adjust(
                current=current,
                target=target,
                max_ratio=self._max_adjustment_ratio,
            )
            if abs(updated - current) < 1e-9:
                continue
            learned_cfg.feature_weights[name] = updated
            adjusted += 1

        if adjusted and self._logger is not None:
            self._logger.info(
                (
                    "runtime_feature_weight_update_applied adjusted=%d "
                    "scale=%.4f avg_failure=%.4f avg_latency_ratio=%.4f"
                ),
                adjusted,
                feature_scale,
                avg_failure,
                avg_latency_ratio,
            )
        return adjusted, feature_scale

    def _apply_classifier_calibration(self) -> bool:
        cfg = self._routing_config.classifier_calibration
        if not bool(cfg.enabled):
            return False
        provider = self._classifier_metrics_provider
        if provider is None:
            return False

        snapshot = getattr(provider, "classifier_calibration_snapshot", None)
        if not isinstance(snapshot, ClassifierCalibrationSnapshot):
            return False

        current_totals = {
            "secondary_total": int(snapshot.secondary_total),
            "secondary_success": int(snapshot.secondary_success),
            "secondary_non_success": int(snapshot.secondary_non_success),
        }
        delta_total = (
            current_totals["secondary_total"]
            - self._last_classifier_totals["secondary_total"]
        )
        delta_success = (
            current_totals["secondary_success"]
            - self._last_classifier_totals["secondary_success"]
        )
        delta_non_success = (
            current_totals["secondary_non_success"]
            - self._last_classifier_totals["secondary_non_success"]
        )

        if delta_total <= 0:
            return False
        if delta_success < 0 or delta_non_success < 0:
            self._last_classifier_totals = current_totals
            return False

        if delta_total < int(cfg.min_samples):
            self._status.last_classifier_samples = delta_total
            return False

        success_rate = float(delta_success) / float(max(1, delta_total))
        self._status.last_classifier_samples = delta_total
        self._status.last_classifier_success_rate = success_rate

        target = float(cfg.target_secondary_success_rate)
        deadband = max(0.0, float(cfg.deadband))
        step = max(0.0, float(cfg.adjustment_step))
        min_threshold = float(cfg.min_threshold)
        max_threshold = float(cfg.max_threshold)

        low_before = float(cfg.secondary_low_confidence_min_confidence)
        mixed_before = float(cfg.secondary_mixed_signal_min_confidence)
        low_after = low_before
        mixed_after = mixed_before

        if success_rate < (target - deadband):
            low_after += step
            mixed_after += step
        elif success_rate > (target + deadband):
            low_after -= step
            mixed_after -= step
        else:
            self._last_classifier_totals = current_totals
            return False

        low_after = min(max(low_after, min_threshold), max_threshold)
        mixed_after = min(max(mixed_after, min_threshold), max_threshold)
        mixed_after = max(mixed_after, low_after)

        changed = (
            abs(low_after - low_before) >= 1e-9
            or abs(mixed_after - mixed_before) >= 1e-9
        )
        if changed:
            cfg.secondary_low_confidence_min_confidence = low_after
            cfg.secondary_mixed_signal_min_confidence = mixed_after
            self._record_classifier_adjustment(
                success_rate=success_rate,
                samples=delta_total,
                low_before=low_before,
                low_after=low_after,
                mixed_before=mixed_before,
                mixed_after=mixed_after,
            )
            if self._logger is not None:
                self._logger.info(
                    (
                        "classifier_calibration_update success_rate=%.4f samples=%d "
                        "threshold_low=%.4f threshold_mixed=%.4f"
                    ),
                    success_rate,
                    delta_total,
                    low_after,
                    mixed_after,
                )

        self._last_classifier_totals = current_totals
        return changed

    def _record_classifier_adjustment(
        self,
        *,
        success_rate: float,
        samples: int,
        low_before: float,
        low_after: float,
        mixed_before: float,
        mixed_after: float,
    ) -> None:
        event = {
            "ts": time.time(),
            "secondary_success_rate": float(success_rate),
            "secondary_samples": int(samples),
            "threshold_low_before": float(low_before),
            "threshold_low_after": float(low_after),
            "threshold_mixed_before": float(mixed_before),
            "threshold_mixed_after": float(mixed_after),
        }
        self._classifier_adjustment_history.append(event)
        if (
            len(self._classifier_adjustment_history)
            > self._classifier_adjustment_history_limit
        ):
            self._classifier_adjustment_history = self._classifier_adjustment_history[
                -self._classifier_adjustment_history_limit :
            ]

    def _classifier_calibration_overrides_payload(self) -> dict[str, Any] | None:
        cfg = self._routing_config.classifier_calibration
        provider = self._classifier_metrics_provider
        snapshot = (
            getattr(provider, "classifier_calibration_snapshot", None)
            if provider is not None
            else None
        )
        if snapshot is not None and not isinstance(
            snapshot, ClassifierCalibrationSnapshot
        ):
            snapshot = None

        payload: dict[str, Any] = {
            "enabled": bool(cfg.enabled),
            "target_secondary_success_rate": float(cfg.target_secondary_success_rate),
            "secondary_low_confidence_min_confidence": float(
                cfg.secondary_low_confidence_min_confidence
            ),
            "secondary_mixed_signal_min_confidence": float(
                cfg.secondary_mixed_signal_min_confidence
            ),
            "min_samples": int(cfg.min_samples),
            "adjustment_step": float(cfg.adjustment_step),
            "deadband": float(cfg.deadband),
            "min_threshold": float(cfg.min_threshold),
            "max_threshold": float(cfg.max_threshold),
            "history": list(self._classifier_adjustment_history),
        }
        if isinstance(snapshot, ClassifierCalibrationSnapshot):
            payload["secondary_total"] = int(snapshot.secondary_total)
            payload["secondary_success"] = int(snapshot.secondary_success)
            payload["secondary_non_success"] = int(snapshot.secondary_non_success)
            payload["secondary_success_rate"] = (
                None
                if snapshot.secondary_success_rate is None
                else float(snapshot.secondary_success_rate)
            )
        return payload


def _bounded_adjust(current: float, target: float, max_ratio: float) -> float:
    if current <= 0:
        return target
    upper = current * (1.0 + max_ratio)
    lower = max(0.0, current * (1.0 - max_ratio))
    return min(max(target, lower), upper)


def _feature_weight_scale_for_stability(
    *,
    avg_failure: float,
    avg_latency_ratio: float,
) -> float:
    failure_pressure = max(0.0, (avg_failure - 0.03) / 0.25)
    latency_pressure = max(0.0, (avg_latency_ratio - 1.1) / 0.9)
    stability_pressure = min(1.5, (0.7 * failure_pressure) + (0.3 * latency_pressure))
    scale = 1.0 - (0.35 * stability_pressure)
    return min(max(scale, 0.55), 1.0)


class RuntimeOverridesManager:
    def __init__(
        self,
        *,
        routing_config: RoutingConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._routing_config = routing_config
        self._logger = logger

    def apply_from_path(self, *, path: str | None) -> int:
        if not path:
            return 0
        file_path = Path(path)
        if not file_path.exists():
            return 0

        try:
            raw = load_yaml_dict(
                file_path,
                error_message=f"Expected YAML object in '{file_path}'.",
            )
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "runtime_overrides_load_failed path=%s error=%s",
                    file_path,
                    str(exc),
                )
            return 0
        return self.apply_from_raw(raw=raw, source_path=file_path)

    def apply_from_raw(
        self,
        *,
        raw: dict[str, Any],
        source_path: Path | None = None,
    ) -> int:
        overrides = raw.get("model_profiles")
        if not isinstance(overrides, dict):
            overrides = {}

        applied = 0
        for model, fields in overrides.items():
            if not isinstance(model, str) or not isinstance(fields, dict):
                continue
            profile = self._routing_config.model_profiles.get(model) or ModelProfile()
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
                self._routing_config.model_profiles[model] = profile
                applied += 1

        self.apply_classifier_calibration_overrides(raw=raw)
        self.apply_learned_feature_weight_overrides(raw=raw)

        if applied and self._logger is not None and source_path is not None:
            self._logger.info(
                "runtime_overrides_applied path=%s models=%d", source_path, applied
            )
        return applied

    def write(
        self,
        *,
        path: Path,
        snapshot: dict[str, ModelMetricsSnapshot],
        classifier_calibration: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "generated_at_epoch": time.time(),
            "model_profiles": {},
            "metrics_snapshot": snapshot_to_dict(snapshot),
        }

        for model, profile in sorted(self._routing_config.model_profiles.items()):
            payload["model_profiles"][model] = {
                "latency_ms": float(profile.latency_ms),
                "failure_rate": float(profile.failure_rate),
            }

        if isinstance(classifier_calibration, dict):
            payload["classifier_calibration"] = classifier_calibration
        payload["learned_routing"] = {
            "feature_weights": {
                name: float(value)
                for name, value in sorted(
                    self._routing_config.learned_routing.feature_weights.items()
                )
                if isinstance(value, (int, float))
            }
        }

        write_yaml_dict(path, payload)

    def apply_classifier_calibration_overrides(self, *, raw: dict[str, Any]) -> None:
        calibration_raw = raw.get("classifier_calibration")
        if not isinstance(calibration_raw, dict):
            return

        cfg = self._routing_config.classifier_calibration
        low_raw = calibration_raw.get("secondary_low_confidence_min_confidence")
        mixed_raw = calibration_raw.get("secondary_mixed_signal_min_confidence")
        low = (
            float(low_raw)
            if isinstance(low_raw, (int, float))
            else float(cfg.secondary_low_confidence_min_confidence)
        )
        mixed = (
            float(mixed_raw)
            if isinstance(mixed_raw, (int, float))
            else float(cfg.secondary_mixed_signal_min_confidence)
        )
        min_threshold = float(cfg.min_threshold)
        max_threshold = float(cfg.max_threshold)
        low = min(max(low, min_threshold), max_threshold)
        mixed = min(max(mixed, min_threshold), max_threshold)
        mixed = max(mixed, low)

        cfg.secondary_low_confidence_min_confidence = low
        cfg.secondary_mixed_signal_min_confidence = mixed

    def apply_learned_feature_weight_overrides(self, *, raw: dict[str, Any]) -> None:
        learned_raw = raw.get("learned_routing")
        if not isinstance(learned_raw, dict):
            return
        feature_weights_raw = learned_raw.get("feature_weights")
        if not isinstance(feature_weights_raw, dict):
            return

        for name, value in feature_weights_raw.items():
            if not isinstance(name, str):
                continue
            if not isinstance(value, (int, float)):
                continue
            self._routing_config.learned_routing.feature_weights[name] = float(value)


def apply_runtime_overrides(
    *,
    path: str | None,
    routing_config: RoutingConfig,
    logger: logging.Logger | None = None,
) -> int:
    return RuntimeOverridesManager(
        routing_config=routing_config,
        logger=logger,
    ).apply_from_path(path=path)


def _write_runtime_overrides(
    *,
    path: Path,
    routing_config: RoutingConfig,
    snapshot: dict[str, ModelMetricsSnapshot],
    classifier_calibration: dict[str, Any] | None = None,
) -> None:
    RuntimeOverridesManager(routing_config=routing_config).write(
        path=path,
        snapshot=snapshot,
        classifier_calibration=classifier_calibration,
    )


def _apply_classifier_calibration_overrides(
    *,
    raw: dict[str, Any],
    routing_config: RoutingConfig,
) -> None:
    RuntimeOverridesManager(
        routing_config=routing_config
    ).apply_classifier_calibration_overrides(raw=raw)


def _apply_learned_feature_weight_overrides(
    *,
    raw: dict[str, Any],
    routing_config: RoutingConfig,
) -> None:
    RuntimeOverridesManager(
        routing_config=routing_config
    ).apply_learned_feature_weight_overrides(raw=raw)
