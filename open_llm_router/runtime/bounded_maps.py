from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import ItemsView
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class _BoundedMap[K, V]:
    def __init__(self, max_keys: int):
        self._max_keys = max(1, int(max_keys))
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None, *, touch: bool = False) -> V | None:
        if key not in self._data:
            return default
        value = self._data[key]
        if touch:
            self._data.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        is_new = key not in self._data
        self._data[key] = value
        self._data.move_to_end(key)
        if is_new and len(self._data) > self._max_keys:
            self._data.popitem(last=False)

    def items(self) -> ItemsView[K, V]:
        return self._data.items()

    def to_dict(self) -> dict[K, V]:
        return dict(self._data)


class _BoundedMapViewMixin[K, V]:
    _map: _BoundedMap[K, V]

    def items(self) -> ItemsView[K, V]:
        return self._map.items()

    def to_dict(self) -> dict[K, V]:
        return self._map.to_dict()


class BoundedCounterMap[K](_BoundedMapViewMixin[K, int]):
    def __init__(self, max_keys: int):
        self._map: _BoundedMap[K, int] = _BoundedMap(max_keys=max_keys)

    def increment(self, key: K, amount: int = 1) -> int:
        current = self._map.get(key, 0)
        next_value = int(current or 0) + int(amount)
        self._map.set(key, next_value)
        return next_value

    def get(self, key: K, default: int = 0) -> int:
        value = self._map.get(key, default)
        return int(value or 0)


class BoundedValueMap[K, V](_BoundedMapViewMixin[K, V]):
    def __init__(self, max_keys: int):
        self._map: _BoundedMap[K, V] = _BoundedMap(max_keys=max_keys)

    def set(self, key: K, value: V) -> None:
        self._map.set(key, value)

    def get(self, key: K, default: V) -> V:
        value = self._map.get(key, default)
        return value if value is not None else default


class BoundedDequeMap[K, T](_BoundedMapViewMixin[K, deque[T]]):
    def __init__(self, *, max_keys: int, window_size: int):
        self._window_size = max(1, int(window_size))
        self._map: _BoundedMap[K, deque[T]] = _BoundedMap(max_keys=max_keys)

    def append(self, key: K, value: T) -> deque[T]:
        bucket = self._map.get(key, None, touch=False)
        if bucket is None:
            bucket = deque(maxlen=self._window_size)
        bucket.append(value)
        self._map.set(key, bucket)
        return bucket
