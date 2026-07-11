"""Data types that flow through the generation pipeline.

The pipeline progresses ``Skeleton -> Populated -> Evaluated``. Each stage
accepts the previous type and returns the next; the :class:`~pipeline.pipeline.Pipeline`
builder enforces this progression statically via ``ty``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symbolic import Expression, OperatorSet


@dataclass
class Skeleton:
    """Tree structure with placeholder constant slots — no real values yet.

    ``expression`` carries the command DAG with all constants set to ``0.0``;
    downstream stages fill those slots with sampled magnitudes.
    """

    opset: OperatorSet
    num_inputs: int
    num_constants: int
    expression: Expression


@dataclass
class Populated:
    """Tree with real numeric constants assigned to every slot."""

    opset: OperatorSet
    num_inputs: int
    expression: Expression


@dataclass
class Evaluated:
    """Expression plus sampled ``(X, y)`` data — fully ready for a model."""

    expression: Expression
    X: np.ndarray
    y: np.ndarray
