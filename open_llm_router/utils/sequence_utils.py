from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def dedupe_preserving_order[T: Hashable](values: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    output: list[T] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
