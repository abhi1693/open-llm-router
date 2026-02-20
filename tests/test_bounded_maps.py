from __future__ import annotations

from open_llm_router.runtime.bounded_maps import (
    BoundedCounterMap,
    BoundedDequeMap,
    BoundedValueMap,
)


def test_bounded_counter_map_evicts_oldest_key() -> None:
    counters = BoundedCounterMap[str](max_keys=2)
    counters.increment("a")
    counters.increment("b")
    counters.increment("c")

    assert counters.to_dict() == {"b": 1, "c": 1}


def test_bounded_value_map_evicts_oldest_key() -> None:
    values = BoundedValueMap[str, bool](max_keys=2)
    values.set("a", True)
    values.set("b", False)
    values.set("c", True)

    assert values.to_dict() == {"b": False, "c": True}


def test_bounded_deque_map_bounds_key_and_window_size() -> None:
    windows = BoundedDequeMap[str, int](max_keys=2, window_size=2)
    windows.append("a", 1)
    windows.append("a", 2)
    windows.append("a", 3)
    windows.append("b", 10)
    windows.append("c", 20)

    payload = {key: list(values) for key, values in windows.items()}
    assert payload == {"b": [10], "c": [20]}
