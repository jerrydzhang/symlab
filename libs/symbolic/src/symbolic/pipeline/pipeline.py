from __future__ import annotations

from typing import Any, Generic, Iterator, Protocol, TypeVar, Callable

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class Stage(Protocol, Generic[T1, T2]):
    """A single pipeline step: maps an input of type ``T1`` to ``T2``."""

    def __call__(self, input: T1) -> T2: ...


class Pipeline(Generic[T2]):
    """Composable, type-safe pipeline over a type progression."""

    _stages: list[Any]

    def __init__(self, first_stage: Stage[None, T2]) -> None:
        self._stages = [first_stage]

    def then(self, next_stage: Stage[T2, T3]) -> "Pipeline[T3]":
        """Append ``next_stage``; its input type must equal the current output."""
        p: Pipeline[T3] = Pipeline.__new__(Pipeline)
        p._stages = [*self._stages, next_stage]
        return p

    def filter(self, predicate: Callable[[T2], bool]) -> "Pipeline[T2]":
        """Drop entries for which ``predicate`` returns ``False``."""
        p: Pipeline[T2] = Pipeline.__new__(Pipeline)
        p._stages = [*self._stages, ("filter", predicate)]
        return p

    def iter(self, n: int) -> Iterator[T2]:
        """Generate up to ``n`` valid entries (lazy, pulls on demand)."""
        produced = 0
        while produced < n:
            value: Any = None
            for stage in self._stages:
                if isinstance(stage, tuple) and stage[0] == "filter":
                    if not stage[1](value):
                        value = None
                        break
                else:
                    value = stage(value)
                    if value is None:
                        break
            if value is not None:
                produced += 1
                yield value
