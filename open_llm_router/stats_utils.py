from __future__ import annotations

from math import ceil


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = max(1, ceil(quantile * len(sorted_values)))
    idx = min(len(sorted_values) - 1, rank - 1)
    return float(sorted_values[idx])
