"""Standalone scorer functions.

A scorer is a plain callable ``(expression, X, y) -> float`` that measures some
quality of an expression against data. Scorers are free functions — they are
not methods on :class:`~symbolic.Expression`, because scoring is an extrinsic
measurement, not an intrinsic property of the expression.
"""

from __future__ import annotations

import numpy as np

from symbolic import Expression


def r2(expression: Expression, X: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination: ``1 - SS_res / SS_tot``.

    ``SS_res`` is the mean squared residual; ``SS_tot`` is the population
    variance of ``y`` (``ddof=0``), guarded to ``1e-9`` so constant targets do
    not divide by zero. A perfect prediction scores ``1.0``; a predictor worse
    than the mean scores below zero.
    """
    y_pred = expression.evaluate(X)
    ss_res = np.mean((y - y_pred) ** 2)
    ss_tot = np.var(y)
    return float(1.0 - ss_res / max(ss_tot, 1e-9))


def complexity(expression: Expression, X: np.ndarray, y: np.ndarray) -> float:
    """Native DAG node count: commands + constants + inputs.

    Shares the ``(expression, X, y)`` signature for scorer-call uniformity but
    ignores ``X`` and ``y`` — complexity is a structural property.
    """
    _, _ = X, y
    n = len(expression.commands) + len(expression.constants) + expression.num_inputs
    return float(n)
