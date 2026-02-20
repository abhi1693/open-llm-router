from __future__ import annotations

from open_llm_router.bounded_maps import (
    BoundedCounterMap,
    BoundedDequeMap,
    BoundedValueMap,
)
from open_llm_router.stats_utils import percentile


class ProxyMetricsAccumulator:
    def __init__(
        self,
        *,
        connect_latency_window_size: int = 256,
        connect_latency_alert_threshold_ms: float = 8000.0,
        target_dimension_max_keys: int = 4096,
        error_type_max_keys: int = 256,
    ) -> None:
        self._connect_latency_window_size = max(10, int(connect_latency_window_size))
        self._connect_latency_alert_threshold_ms = max(
            0.0, float(connect_latency_alert_threshold_ms)
        )
        max_target_keys = max(1, int(target_dimension_max_keys))
        max_error_type_keys = max(1, int(error_type_max_keys))

        self._proxy_retries_total = 0
        self._proxy_timeouts_total = 0
        self._proxy_rate_limited_total = 0
        self._proxy_connect_latency_alerts_total = 0
        self._proxy_attempt_latency_sum_ms = 0.0
        self._proxy_attempt_latency_count = 0

        self._proxy_retries_by_target: BoundedCounterMap[tuple[str, str]] = (
            BoundedCounterMap(max_keys=max_target_keys)
        )
        self._proxy_timeouts_by_target: BoundedCounterMap[tuple[str, str]] = (
            BoundedCounterMap(max_keys=max_target_keys)
        )
        self._proxy_connect_latency_alerts_by_target: BoundedCounterMap[
            tuple[str, str, str]
        ] = BoundedCounterMap(max_keys=max_target_keys)
        self._proxy_connect_latency_alert_active_by_target: BoundedValueMap[
            tuple[str, str, str], bool
        ] = BoundedValueMap(max_keys=max_target_keys)
        self._proxy_connect_latency_samples_by_target: BoundedDequeMap[
            tuple[str, str, str], float
        ] = BoundedDequeMap(
            max_keys=max_target_keys,
            window_size=self._connect_latency_window_size,
        )
        self._proxy_errors_by_type: BoundedCounterMap[str] = BoundedCounterMap(
            max_keys=max_error_type_keys
        )
        self._proxy_responses_by_status_class: BoundedCounterMap[str] = (
            BoundedCounterMap(max_keys=16)
        )

    @property
    def connect_latency_window_size(self) -> int:
        return self._connect_latency_window_size

    @property
    def connect_latency_alert_threshold_ms(self) -> float:
        return self._connect_latency_alert_threshold_ms

    @property
    def proxy_retries_total(self) -> int:
        return self._proxy_retries_total

    @property
    def proxy_timeouts_total(self) -> int:
        return self._proxy_timeouts_total

    @property
    def proxy_rate_limited_total(self) -> int:
        return self._proxy_rate_limited_total

    @property
    def proxy_connect_latency_alerts_total(self) -> int:
        return self._proxy_connect_latency_alerts_total

    @property
    def proxy_attempt_latency_sum_ms(self) -> float:
        return self._proxy_attempt_latency_sum_ms

    @property
    def proxy_attempt_latency_count(self) -> int:
        return self._proxy_attempt_latency_count

    @property
    def proxy_retries_by_target(self) -> dict[tuple[str, str], int]:
        return self._proxy_retries_by_target.to_dict()

    @property
    def proxy_timeouts_by_target(self) -> dict[tuple[str, str], int]:
        return self._proxy_timeouts_by_target.to_dict()

    @property
    def proxy_connect_latency_alerts_by_target(self) -> dict[tuple[str, str, str], int]:
        return self._proxy_connect_latency_alerts_by_target.to_dict()

    @property
    def proxy_errors_by_type(self) -> dict[str, int]:
        return self._proxy_errors_by_type.to_dict()

    @property
    def proxy_responses_by_status_class(self) -> dict[str, int]:
        return self._proxy_responses_by_status_class.to_dict()

    @property
    def proxy_connect_latency_quantiles_by_target(
        self,
    ) -> dict[tuple[str, str, str], dict[str, float | int]]:
        output: dict[tuple[str, str, str], dict[str, float | int]] = {}
        for key, samples in self._proxy_connect_latency_samples_by_target.items():
            values = list(samples)
            if not values:
                continue
            output[key] = {
                "count": len(values),
                "p50": percentile(values, 0.50) or 0.0,
                "p95": percentile(values, 0.95) or 0.0,
                "p99": percentile(values, 0.99) or 0.0,
            }
        return output

    def record_connect(
        self, target_key: tuple[str, str, str], connect_ms: float
    ) -> None:
        samples = self._proxy_connect_latency_samples_by_target.append(
            target_key, connect_ms
        )
        if self._connect_latency_alert_threshold_ms <= 0.0:
            return

        p95_value = percentile(list(samples), 0.95)
        is_alerting = (
            p95_value is not None
            and p95_value > self._connect_latency_alert_threshold_ms
        )
        was_alerting = self._proxy_connect_latency_alert_active_by_target.get(
            target_key, False
        )
        if is_alerting and not was_alerting:
            self._proxy_connect_latency_alerts_total += 1
            self._proxy_connect_latency_alerts_by_target.increment(target_key)
        self._proxy_connect_latency_alert_active_by_target.set(target_key, is_alerting)

    def record_response(self, status: int) -> None:
        self._proxy_responses_by_status_class.increment(f"{max(0, status) // 100}xx")
        if status == 429:
            self._proxy_rate_limited_total += 1

    def record_retry(self, target_key: tuple[str, str]) -> None:
        self._proxy_retries_total += 1
        self._proxy_retries_by_target.increment(target_key)

    def record_error(
        self,
        *,
        target_key: tuple[str, str],
        error_type: str,
        is_timeout: bool,
        attempt_latency_ms: float | None,
    ) -> None:
        self._proxy_errors_by_type.increment(error_type)
        if is_timeout:
            self._proxy_timeouts_total += 1
            self._proxy_timeouts_by_target.increment(target_key)
        if attempt_latency_ms is not None:
            self._proxy_attempt_latency_sum_ms += max(0.0, attempt_latency_ms)
            self._proxy_attempt_latency_count += 1
