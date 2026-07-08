"""Warm-layer SR metrics computed natively on ``Expression``.

``score`` is a thin delegate to :meth:`Expression.score`; the metric logic
lives on ``Expression`` itself. ``ScoreResult`` is defined in
:mod:`symbolic.expression` and re-imported here so
``from symbolic.metrics import ScoreResult`` keeps working.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .expression import Expression, ScoreResult


def score(
    expr: Expression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> ScoreResult:
    """Compute standard SR metrics for an Expression on data.

    Thin delegate for :meth:`Expression.score`; prefer calling the method
    directly: ``expr.score(X, y)``.
    """
    return expr.score(X, y)
