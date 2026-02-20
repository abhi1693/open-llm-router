from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from open_llm_router.config import RoutingConfig
from open_llm_router.live_metrics import (
    InMemoryLiveMetricsStore,
    LiveMetricsCollector,
    build_target_metrics_key,
)
from open_llm_router.policy_updater import RuntimePolicyUpdater, apply_runtime_overrides
from tests.yaml_test_utils import save_yaml_file


def test_live_metrics_collector_records_events() -> None:
    async def _run() -> None:
        store = InMemoryLiveMetricsStore(alpha=0.5)
        collector = LiveMetricsCollector(store=store, enabled=True)
        await collector.start()

        collector.ingest(
            {
                "event": "route_decision",
                "selected_model": "m1",
                "task": "coding",
                "complexity": "high",
            }
        )
        collector.ingest(
            {
                "event": "proxy_upstream_connected",
                "model": "m1",
                "connect_ms": 1000.0,
            }
        )
        collector.ingest(
            {
                "event": "proxy_response",
                "model": "m1",
                "status": 200,
                "request_latency_ms": 1200.0,
            }
        )
        collector.ingest(
            {
                "event": "proxy_retry",
                "provider": "openai",
                "account": "acct-a",
                "model": "m1",
                "status": 503,
            }
        )
        collector.ingest(
            {
                "event": "proxy_request_error",
                "model": "m1",
                "error_type": "ConnectTimeout",
                "is_timeout": True,
                "provider": "openai",
                "account": "acct-a",
                "attempt_latency_ms": 15.5,
            }
        )

        await asyncio.sleep(0)
        await collector.close()

        snapshot = await store.snapshot_all()
        assert "m1" in snapshot
        m1 = snapshot["m1"]
        assert m1.route_decisions == 1
        assert m1.responses == 1
        assert m1.errors == 1
        assert m1.ewma_connect_ms is not None
        assert m1.ewma_request_latency_ms is not None
        assert m1.ewma_failure_rate is not None
        assert collector.proxy_retries_total == 1
        assert collector.proxy_timeouts_total == 1
        assert collector.proxy_attempt_latency_count == 1
        assert collector.proxy_errors_by_type.get("ConnectTimeout") == 1
        assert collector.proxy_responses_by_status_class.get("2xx") == 1

    asyncio.run(_run())


def test_runtime_policy_updater_applies_bounded_adjustments() -> None:
    async def _run() -> None:
        config = RoutingConfig.model_validate(
            {
                "default_model": "m1",
                "task_routes": {"general": {"default": "m1"}},
                "model_profiles": {
                    "m1": {
                        "latency_ms": 1000.0,
                        "failure_rate": 0.10,
                    }
                },
            }
        )
        store = InMemoryLiveMetricsStore(alpha=1.0)

        # Directly feed enough samples to exceed the minimum threshold.
        for _ in range(12):
            await store.record_response("m1", status=200, request_latency_ms=2000.0)

        updater = RuntimePolicyUpdater(
            routing_config=config,
            metrics_store=store,
            enabled=True,
            interval_seconds=60.0,
            min_samples=10,
            max_adjustment_ratio=0.1,
            overrides_path=None,
        )

        applied = await updater.run_once()
        assert applied >= 1
        profile = config.model_profiles["m1"]
        # bounded by +10% from 1000 -> 1100
        assert profile.latency_ms == 1100.0
        # bounded by -10% from 0.10 towards observed 0.0 -> 0.09
        assert profile.failure_rate == pytest.approx(0.09)

    asyncio.run(_run())


def test_runtime_policy_updater_adapts_feature_weights_for_unstable_runtime() -> None:
    async def _run() -> None:
        config = RoutingConfig.model_validate(
            {
                "default_model": "m1",
                "task_routes": {"general": {"default": "m1"}},
                "learned_routing": {
                    "enabled": True,
                    "feature_weights": {
                        "complexity_score": 1.2,
                        "reasoning_effort_high": 1.0,
                        "task_coding": 0.8,
                    },
                },
                "model_profiles": {
                    "m1": {
                        "latency_ms": 1000.0,
                        "failure_rate": 0.02,
                    }
                },
            }
        )
        store = InMemoryLiveMetricsStore(alpha=1.0)

        for _ in range(12):
            await store.record_response("m1", status=500, request_latency_ms=2500.0)

        updater = RuntimePolicyUpdater(
            routing_config=config,
            metrics_store=store,
            enabled=True,
            interval_seconds=60.0,
            min_samples=10,
            max_adjustment_ratio=0.2,
            overrides_path=None,
        )

        adjusted = await updater.run_once()
        assert adjusted >= 1
        weights = config.learned_routing.feature_weights
        assert weights["complexity_score"] < 1.2
        assert weights["reasoning_effort_high"] < 1.0
        # Task indicators remain stable; adaptation targets cross-cutting features.
        assert weights["task_coding"] == pytest.approx(0.8)
        assert updater.status.last_feature_weight_adjusted_count >= 1
        assert updater.status.last_feature_weight_scale is not None
        assert updater.status.last_feature_weight_scale < 1.0

    asyncio.run(_run())


def test_runtime_policy_updater_restores_feature_weights_toward_baseline() -> None:
    async def _run() -> None:
        config = RoutingConfig.model_validate(
            {
                "default_model": "m1",
                "task_routes": {"general": {"default": "m1"}},
                "learned_routing": {
                    "enabled": True,
                    "feature_weights": {
                        "complexity_score": 1.2,
                    },
                },
                "model_profiles": {
                    "m1": {
                        "latency_ms": 1000.0,
                        "failure_rate": 0.02,
                    }
                },
            }
        )
        store = InMemoryLiveMetricsStore(alpha=1.0)

        for _ in range(12):
            await store.record_response("m1", status=500, request_latency_ms=2500.0)

        updater = RuntimePolicyUpdater(
            routing_config=config,
            metrics_store=store,
            enabled=True,
            interval_seconds=60.0,
            min_samples=10,
            max_adjustment_ratio=0.2,
            overrides_path=None,
        )

        await updater.run_once()
        after_unstable = config.learned_routing.feature_weights["complexity_score"]
        assert after_unstable < 1.2

        for _ in range(12):
            await store.record_response("m1", status=200, request_latency_ms=700.0)

        await updater.run_once()
        after_recovery = config.learned_routing.feature_weights["complexity_score"]
        assert after_recovery > after_unstable
        assert after_recovery <= 1.2
        assert updater.status.last_feature_weight_scale == pytest.approx(1.0)

    asyncio.run(_run())


def test_live_metrics_collector_tracks_target_dimension_metrics() -> None:
    async def _run() -> None:
        store = InMemoryLiveMetricsStore(alpha=0.5)
        collector = LiveMetricsCollector(store=store, enabled=True)
        await collector.start()

        collector.ingest(
            {
                "event": "proxy_response",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "status": 200,
                "request_latency_ms": 220.0,
            }
        )

        await asyncio.sleep(0)
        await collector.close()

        snapshot = await store.snapshot_all()
        target_key = build_target_metrics_key(
            model="m1",
            provider="openai",
            account="acct-a",
        )
        assert target_key is not None
        assert "m1" in snapshot
        assert target_key in snapshot
        assert snapshot["m1"].responses == 1
        assert snapshot[target_key].responses == 1

    asyncio.run(_run())


def test_live_metrics_collector_tracks_connect_latency_quantiles_and_alerts() -> None:
    async def _run() -> None:
        store = InMemoryLiveMetricsStore(alpha=0.5)
        collector = LiveMetricsCollector(
            store=store,
            enabled=True,
            connect_latency_window_size=16,
            connect_latency_alert_threshold_ms=100.0,
        )
        await collector.start()

        collector.ingest(
            {
                "event": "proxy_upstream_connected",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "connect_ms": 50.0,
            }
        )
        collector.ingest(
            {
                "event": "proxy_upstream_connected",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "connect_ms": 120.0,
            }
        )
        collector.ingest(
            {
                "event": "proxy_upstream_connected",
                "model": "m1",
                "provider": "openai",
                "account": "acct-a",
                "connect_ms": 130.0,
            }
        )

        await asyncio.sleep(0)
        await collector.close()

        quantiles = collector.proxy_connect_latency_quantiles_by_target
        key = ("openai", "acct-a", "m1")
        assert key in quantiles
        assert quantiles[key]["count"] == 3
        assert quantiles[key]["p95"] == pytest.approx(130.0)
        assert quantiles[key]["p99"] == pytest.approx(130.0)
        assert collector.proxy_connect_latency_alerts_total == 1
        assert collector.proxy_connect_latency_alerts_by_target[key] == 1

    asyncio.run(_run())


def test_live_metrics_collector_tracks_secondary_classifier_outcomes() -> None:
    async def _run() -> None:
        store = InMemoryLiveMetricsStore(alpha=0.5)
        collector = LiveMetricsCollector(store=store, enabled=True)
        await collector.start()

        for request_id, outcome in (
            ("req-1", "success"),
            ("req-2", "error"),
            ("req-3", "exhausted"),
        ):
            collector.ingest(
                {
                    "event": "route_decision",
                    "request_id": request_id,
                    "selected_model": "m1",
                    "task": "coding",
                    "complexity": "medium",
                    "signals": {
                        "secondary_classifier_used": True,
                    },
                }
            )
            collector.ingest(
                {
                    "event": "proxy_terminal",
                    "request_id": request_id,
                    "outcome": outcome,
                    "status": 200 if outcome == "success" else 500,
                }
            )

        await asyncio.sleep(0)
        await collector.close()

        snapshot = collector.classifier_calibration_snapshot
        assert snapshot.secondary_total == 3
        assert snapshot.secondary_success == 1
        assert snapshot.secondary_non_success == 2
        assert snapshot.secondary_success_rate == pytest.approx(1.0 / 3.0)

    asyncio.run(_run())


def test_runtime_policy_updater_ignores_target_dimension_keys() -> None:
    async def _run() -> None:
        config = RoutingConfig.model_validate(
            {
                "default_model": "m1",
                "task_routes": {"general": {"default": "m1"}},
                "model_profiles": {
                    "m1": {
                        "latency_ms": 1000.0,
                        "failure_rate": 0.10,
                    }
                },
            }
        )
        store = InMemoryLiveMetricsStore(alpha=1.0)

        for _ in range(12):
            await store.record_response(
                "m1",
                status=200,
                request_latency_ms=250.0,
                provider="openai",
                account="acct-a",
            )

        updater = RuntimePolicyUpdater(
            routing_config=config,
            metrics_store=store,
            enabled=True,
            interval_seconds=60.0,
            min_samples=10,
            max_adjustment_ratio=0.1,
            overrides_path=None,
        )
        await updater.run_once()

        assert "m1" in config.model_profiles
        assert not any(key.startswith("target::") for key in config.model_profiles)

    asyncio.run(_run())


def test_runtime_policy_updater_adjusts_classifier_thresholds_from_feedback() -> None:
    async def _run() -> None:
        config = RoutingConfig.model_validate(
            {
                "default_model": "m1",
                "task_routes": {"general": {"default": "m1"}},
                "classifier_calibration": {
                    "enabled": True,
                    "min_samples": 4,
                    "target_secondary_success_rate": 0.75,
                    "secondary_low_confidence_min_confidence": 0.2,
                    "secondary_mixed_signal_min_confidence": 0.4,
                    "adjustment_step": 0.05,
                    "deadband": 0.0,
                    "min_threshold": 0.05,
                    "max_threshold": 0.95,
                },
            }
        )
        store = InMemoryLiveMetricsStore(alpha=0.5)
        collector = LiveMetricsCollector(store=store, enabled=True)
        await collector.start()

        # 1/5 success -> below target -> thresholds should become more conservative.
        outcomes = ["success", "error", "error", "error", "error"]
        for idx, outcome in enumerate(outcomes):
            request_id = f"req-{idx}"
            collector.ingest(
                {
                    "event": "route_decision",
                    "request_id": request_id,
                    "selected_model": "m1",
                    "task": "general",
                    "complexity": "low",
                    "signals": {"secondary_classifier_used": True},
                }
            )
            collector.ingest(
                {
                    "event": "proxy_terminal",
                    "request_id": request_id,
                    "status": 200 if outcome == "success" else 500,
                    "outcome": outcome,
                }
            )

        await asyncio.sleep(0)
        await collector.close()

        updater = RuntimePolicyUpdater(
            routing_config=config,
            metrics_store=store,
            enabled=True,
            interval_seconds=60.0,
            min_samples=1,
            max_adjustment_ratio=0.1,
            overrides_path=None,
            classifier_metrics_provider=collector,
        )

        await updater.run_once()
        assert (
            config.classifier_calibration.secondary_low_confidence_min_confidence
            == 0.25
        )
        assert (
            config.classifier_calibration.secondary_mixed_signal_min_confidence == 0.45
        )
        assert updater.status.last_classifier_adjusted is True
        assert updater.status.last_classifier_samples == 5
        assert updater.status.last_classifier_success_rate == pytest.approx(0.2)
        assert len(updater.status.classifier_adjustment_history) == 1
        assert updater.status.classifier_adjustment_history[0][
            "threshold_low_after"
        ] == pytest.approx(0.25)

    asyncio.run(_run())


def test_apply_runtime_overrides_updates_classifier_calibration_thresholds(
    tmp_path: Path,
) -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "m1",
            "task_routes": {"general": {"default": "m1"}},
            "classifier_calibration": {
                "enabled": True,
                "secondary_low_confidence_min_confidence": 0.2,
                "secondary_mixed_signal_min_confidence": 0.4,
                "min_threshold": 0.05,
                "max_threshold": 0.9,
            },
        }
    )

    overrides_path = tmp_path / "router.runtime.overrides.yaml"
    payload = {
        "classifier_calibration": {
            "secondary_low_confidence_min_confidence": 0.31,
            "secondary_mixed_signal_min_confidence": 0.52,
        }
    }
    save_yaml_file(overrides_path, payload)

    applied = apply_runtime_overrides(
        path=str(overrides_path),
        routing_config=config,
    )

    assert applied == 0
    assert (
        config.classifier_calibration.secondary_low_confidence_min_confidence
        == pytest.approx(0.31)
    )
    assert (
        config.classifier_calibration.secondary_mixed_signal_min_confidence
        == pytest.approx(0.52)
    )


def test_apply_runtime_overrides_updates_model_profiles(tmp_path: Path) -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "m1",
            "task_routes": {"general": {"default": "m1"}},
            "model_profiles": {
                "m1": {
                    "latency_ms": 1000.0,
                    "failure_rate": 0.10,
                }
            },
        }
    )

    overrides_path = tmp_path / "router.runtime.overrides.yaml"
    payload = {
        "model_profiles": {
            "m1": {"latency_ms": 850.0, "failure_rate": 0.03},
            "m2": {"latency_ms": 400.0, "failure_rate": 0.02},
        }
    }
    save_yaml_file(overrides_path, payload)

    applied = apply_runtime_overrides(
        path=str(overrides_path),
        routing_config=config,
    )

    assert applied == 2
    assert config.model_profiles["m1"].latency_ms == 850.0
    assert config.model_profiles["m1"].failure_rate == 0.03
    assert config.model_profiles["m2"].latency_ms == 400.0
    assert config.model_profiles["m2"].failure_rate == 0.02


def test_apply_runtime_overrides_updates_learned_feature_weights(
    tmp_path: Path,
) -> None:
    config = RoutingConfig.model_validate(
        {
            "default_model": "m1",
            "task_routes": {"general": {"default": "m1"}},
            "learned_routing": {
                "enabled": True,
                "feature_weights": {
                    "complexity_score": 1.2,
                    "reasoning_effort_high": 1.0,
                },
            },
        }
    )

    overrides_path = tmp_path / "router.runtime.overrides.yaml"
    payload = {
        "learned_routing": {
            "feature_weights": {
                "complexity_score": 0.95,
                "reasoning_effort_high": 0.72,
            }
        }
    }
    save_yaml_file(overrides_path, payload)

    applied = apply_runtime_overrides(
        path=str(overrides_path),
        routing_config=config,
    )

    assert applied == 0
    assert config.learned_routing.feature_weights["complexity_score"] == pytest.approx(
        0.95
    )
    assert config.learned_routing.feature_weights[
        "reasoning_effort_high"
    ] == pytest.approx(0.72)


def test_live_metrics_collector_bounds_high_cardinality_maps() -> None:
    async def _run() -> None:
        store = InMemoryLiveMetricsStore(alpha=0.5)
        collector = LiveMetricsCollector(
            store=store,
            enabled=True,
            target_dimension_max_keys=2,
            error_type_max_keys=2,
        )
        await collector.start()

        for account, error_type in (
            ("acct-a", "ConnectTimeout"),
            ("acct-b", "ReadTimeout"),
            ("acct-c", "ProtocolError"),
        ):
            collector.ingest(
                {
                    "event": "proxy_retry",
                    "provider": "openai",
                    "account": account,
                    "model": "m1",
                }
            )
            collector.ingest(
                {
                    "event": "proxy_request_error",
                    "provider": "openai",
                    "account": account,
                    "model": "m1",
                    "error_type": error_type,
                    "is_timeout": True,
                }
            )

        await asyncio.sleep(0)
        await collector.close()

        assert collector.proxy_retries_by_target == {
            ("openai", "acct-b"): 1,
            ("openai", "acct-c"): 1,
        }
        assert collector.proxy_timeouts_by_target == {
            ("openai", "acct-b"): 1,
            ("openai", "acct-c"): 1,
        }
        assert collector.proxy_errors_by_type == {
            "ReadTimeout": 1,
            "ProtocolError": 1,
        }

    asyncio.run(_run())
