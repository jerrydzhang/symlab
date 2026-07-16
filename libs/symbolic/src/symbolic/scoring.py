import numpy as np

from symbolic import Expression


def r2(expression: Expression, X: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination: ``1 - SS_res / SS_tot``.

    ``SS_tot`` is guarded to ``1e-9`` so constant targets don't divide by zero.
    """
    y_pred = expression.evaluate(X)
    ss_res = np.mean((y - y_pred) ** 2)
    ss_tot = np.var(y)
    return float(1.0 - ss_res / max(ss_tot, 1e-9))


def complexity(expression: Expression, X: np.ndarray, y: np.ndarray) -> float:
    """DAG node count (commands + constants + inputs); ignores ``X`` and ``y``."""
    _, _ = X, y
    n = len(expression.commands) + len(expression.constants) + expression.num_inputs
    return float(n)
