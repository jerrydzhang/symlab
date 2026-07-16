import numpy as np

from symbolic import (
    Expression,
    OperatorSet,
    RandomBinaryTree,
    MantissaExponentConstants,
    r2,
)


class RandomBFGSModel:
    """Random tree search + least-squares constant fitting.

    Parameters
    ----------
    max_ops:
        Upper bound on operator nodes per random tree.
    n_tries:
        Number of random trees tried per problem.
    rng:
        Seeded generator for reproducibility.
    """

    def __init__(
        self,
        max_ops: int = 5,
        n_tries: int = 10,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_ops = max_ops
        self.n_tries = n_tries
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(
        self,
        problems: list[tuple[np.ndarray, np.ndarray]],
        opset: OperatorSet,
    ) -> list[Expression | None]:
        results: list[Expression | None] = []
        for X, y in problems:
            n_inputs = X.shape[1]
            best_expr: Expression | None = None
            best_r2 = -np.inf
            last_expr: Expression | None = None
            for _ in range(self.n_tries):
                tree_gen = RandomBinaryTree(
                    opset,
                    max_ops=self.max_ops,
                    num_vars=(n_inputs, n_inputs),
                    rng=self.rng,
                )
                skeleton = tree_gen(None)

                const_gen = MantissaExponentConstants(rng=self.rng)
                populated = const_gen(skeleton)
                expr = populated.expression
                last_expr = expr

                try:
                    candidate = expr.fit(X, y)
                # if expression fitting results in an exception use the unfit expression
                except (ValueError, FloatingPointError, ArithmeticError):
                    candidate = expr

                # if scoring fails score becomes -inf
                with np.errstate(all="ignore"):
                    score = r2(candidate, X, y)
                if not np.isfinite(score):
                    score = -np.inf

                if score > best_r2:
                    best_r2 = score
                    best_expr = candidate

            # if all tries failed, return the last expression (unfit) instead of None
            results.append(best_expr if best_expr is not None else last_expr)
        return results
