"""Model protocol: the interface every symbolic-regression model implements."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from symbolic import Expression, OperatorSet


class SRModel(Protocol):
    """A symbolic regression model that solves a batch of problems.

    Each problem is a ``(X, y)`` tuple. The model returns one result per
    problem: an :class:`~symbolic.Expression` if it produced a valid equation,
    or ``None`` if the output was structurally invalid (e.g. a decoder
    produced an impossible token sequence).
    """

    def fit(
        self,
        problems: list[tuple[np.ndarray, np.ndarray]],
        opset: OperatorSet,
    ) -> list[Expression | None]: ...
