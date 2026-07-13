"""Random skeleton + BFGS constant fitting — the simplest SR baseline."""

from __future__ import annotations

import numpy as np

from symbolic import (
    Expression,
    OperatorSet,
    RandomBinaryTree,
    MantissaExponentConstants,
    r2,
)


class RandomBFGSModel:
    """Dumbest SR: random search over tree structure + least-squares over constants.

    For each problem, generates ``n_tries`` random expression skeletons, fills
    their constants with random magnitudes, then fits constants to the data via
    Levenberg-Marquardt least-squares. Returns the best-fitting expression.

    Skeletons are structurally valid by construction, so the output is always a
    real (if poor) expression. A try is only discarded when its random initial
    constants make the least-squares residual non-finite (e.g. ``exp`` of a huge
    constant); such numerically invalid tries are skipped rather than crashing.
    With any reasonable ``n_tries`` at least one try succeeds, so the result is
    effectively never ``None`` in practice — ``None`` is returned only if every
    try fails numerically.

    Parameters
    ----------
    max_ops:
        Upper bound on operator nodes per random tree.
    n_tries:
        Number of random trees to try per problem.
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
            # Skeleton arity must match the data's input count: Expression.evaluate
            # sizes its working memory from X.shape[1], so a mismatched skeleton
            # would index out of bounds. Pin num_vars to the problem dimension.
            n_inputs = X.shape[1]
            best_expr: Expression | None = None
            best_r2 = -np.inf
            for _ in range(self.n_tries):
                # Extreme sampled constants can overflow during optimization
                # (e.g. exp of a large value); that is expected noise from random
                # search, so suppress it and skip any try that fails outright.
                with np.errstate(all="ignore"):
                    try:
                        tree_gen = RandomBinaryTree(
                            opset,
                            max_ops=self.max_ops,
                            num_vars=(n_inputs, n_inputs),
                            rng=self.rng,
                        )
                        skeleton = tree_gen(None)

                        const_gen = MantissaExponentConstants(rng=self.rng)
                        populated = const_gen(skeleton)

                        fitted = populated.expression.fit(X, y)
                        score = r2(fitted, X, y)
                    except (ValueError, FloatingPointError, ArithmeticError):
                        continue
                if score > best_r2:
                    best_r2 = score
                    best_expr = fitted

            results.append(best_expr)
        return results
