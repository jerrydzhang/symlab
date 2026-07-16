from dataclasses import dataclass

import numpy as np

from symbolic import Expression, OperatorSet


@dataclass
class Skeleton:
    """Tree structure with placeholder (``0.0``) constants awaiting a fill stage."""

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

    opset: OperatorSet
    expression: Expression
    X: np.ndarray
    y: np.ndarray
