"""Warm-layer SR metrics computed natively on ``Expression``.

The cold layer (SRBench) scores recovered models through sklearn/sympy; this
module is the warm-layer equivalent that operates directly on the flat
command-DAG representation. numpy only, no sklearn, no file I/O.

R² uses the population variance of ``y`` as its denominator, guarded to
``1e-9`` when ``y`` is constant (the same guard SRBench applies). Complexity
is the total node count of the DAG: inputs + constants + command nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .expression import Expression


@dataclass(frozen=True)
class ScoreResult:
    """Standard symbolic-regression metrics for an ``Expression`` on data."""

    r2: float
    mse: float
    mae: float
    complexity: int


def score(
    expr: Expression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> ScoreResult:
    """Compute standard SR metrics for an Expression on data.

    ``X`` is ``(num_inputs, num_samples)`` matching ``Expression.evaluate``;
    ``y`` is ``(num_samples,)`` ground truth.
    """
    y_pred = expr.evaluate(X)
    residual = y - y_pred

    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))

    var_y = float(np.var(y))
    denom = var_y if var_y != 0.0 else 1e-9
    r2 = float(1.0 - mse / denom)

    complexity = len(expr.commands) + len(expr.constants) + expr.num_inputs
    return ScoreResult(r2=r2, mse=mse, mae=mae, complexity=complexity)
