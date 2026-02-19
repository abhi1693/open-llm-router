from __future__ import annotations

import pytest

from open_llm_router.proxy_metrics import ProxyMetricsAccumulator


def test_proxy_metrics_accumulator_records_core_metrics() -> None:
    metrics = ProxyMetricsAccumulator(
        connect_latency_window_size=16,
        connect_latency_alert_threshold_ms=100.0,
        target_dimension_max_keys=2,
        error_type_max_keys=2,
    )

    metrics.record_connect(("openai", "acct-a", "m1"), 50.0)
    metrics.record_connect(("openai", "acct-a", "m1"), 150.0)
    metrics.record_connect(("openai", "acct-a", "m1"), 160.0)

    assert metrics.proxy_connect_latency_alerts_total == 1
    assert metrics.proxy_connect_latency_alerts_by_target == {
        ("openai", "acct-a", "m1"): 1
    }
    assert metrics.proxy_connect_latency_quantiles_by_target[("openai", "acct-a", "m1")][
        "p95"
    ] == pytest.approx(160.0)

    metrics.record_response(200)
    metrics.record_response(429)
    assert metrics.proxy_responses_by_status_class == {"2xx": 1, "4xx": 1}
    assert metrics.proxy_rate_limited_total == 1

    metrics.record_retry(("openai", "acct-a"))
    metrics.record_retry(("openai", "acct-b"))
    metrics.record_retry(("openai", "acct-c"))
    assert metrics.proxy_retries_total == 3
    assert metrics.proxy_retries_by_target == {
        ("openai", "acct-b"): 1,
        ("openai", "acct-c"): 1,
    }

    metrics.record_error(
        target_key=("openai", "acct-a"),
        error_type="ConnectTimeout",
        is_timeout=True,
        attempt_latency_ms=10.5,
    )
    metrics.record_error(
        target_key=("openai", "acct-b"),
        error_type="ReadTimeout",
        is_timeout=True,
        attempt_latency_ms=None,
    )
    metrics.record_error(
        target_key=("openai", "acct-c"),
        error_type="ProtocolError",
        is_timeout=False,
        attempt_latency_ms=5.0,
    )

    assert metrics.proxy_errors_by_type == {"ReadTimeout": 1, "ProtocolError": 1}
    assert metrics.proxy_timeouts_total == 2
    assert metrics.proxy_timeouts_by_target == {
        ("openai", "acct-a"): 1,
        ("openai", "acct-b"): 1,
    }
    assert metrics.proxy_attempt_latency_count == 2
    assert metrics.proxy_attempt_latency_sum_ms == pytest.approx(15.5)
