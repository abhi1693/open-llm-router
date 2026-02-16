from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from open_llm_router.config import RoutingConfig
from open_llm_router.live_metrics import (
    InMemoryLiveMetricsStore,
    LiveMetricsCollector,
    build_target_metrics_key,
)
from open_llm_router.policy_updater import RuntimePolicyUpdater, apply_runtime_overrides


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
                "event": "proxy_request_error",
                "model": "m1",
                "error_type": "ConnectTimeout",
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
    with overrides_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    applied = apply_runtime_overrides(
        path=str(overrides_path),
        routing_config=config,
    )

    assert applied == 2
    assert config.model_profiles["m1"].latency_ms == 850.0
    assert config.model_profiles["m1"].failure_rate == 0.03
    assert config.model_profiles["m2"].latency_ms == 400.0
    assert config.model_profiles["m2"].failure_rate == 0.02
