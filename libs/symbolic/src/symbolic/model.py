from typing import Protocol

import numpy as np

from symbolic import Expression, OperatorSet


class SRModel(Protocol):
    """Symbolic-regression model: fit a batch of ``(X, y)`` problems.

    Returns one :class:`~symbolic.Expression` per problem, or ``None`` if the
    result is structurally invalid.
    """

    def fit(
        self,
        problems: list[tuple[np.ndarray, np.ndarray]],
        opset: OperatorSet,
    ) -> list[Expression | None]: ...
