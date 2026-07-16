import numpy as np

from symbolic import Expression


def r2(
    expressions: list[Expression], Xs: list[np.ndarray], ys: list[np.ndarray]
) -> list[float]:
    """Coefficient of determination for each (expression, X, y) triple.

    ``SS_tot`` is guarded to ``1e-9`` so constant targets don't divide by zero.
    """
    scores: list[float] = []
    for expression, X, y in zip(expressions, Xs, ys, strict=True):
        y_pred = expression.evaluate(X)
        ss_res = np.mean((y - y_pred) ** 2)
        ss_tot = np.var(y)
        scores.append(float(1.0 - ss_res / max(ss_tot, 1e-9)))
    return scores


def complexity(
    expressions: list[Expression], Xs: list[np.ndarray], ys: list[np.ndarray]
) -> list[float]:
    """DAG node count for each expression; ignores X and y."""
    scores: list[float] = []
    for expression, X, y in zip(expressions, Xs, ys, strict=True):
        n = len(expression.commands) + len(expression.constants) + expression.num_inputs
        scores.append(float(n))
    return scores
