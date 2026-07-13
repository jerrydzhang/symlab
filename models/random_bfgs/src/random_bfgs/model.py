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
    real (if poor) expression. A try is *never* discarded: when the random
    initial constants overflow during least-squares (e.g. ``exp`` of a huge
    value) the fit raises, and the unfitted expression is scored instead — its
    terrible ``r2`` is the signal that this try was bad, not a reason to drop
    it. A non-finite score (``nan``/``inf`` from evaluation overflow) is mapped
    to ``-inf`` so it never wins but still competes. The result is effectively
    never ``None``; ``None`` is returned only if no try can construct an
    expression at all, which cannot happen for structurally valid skeletons.

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

                # Fit constants via least-squares. When the random initial
                # constants overflow (e.g. exp of a huge value), least_squares
                # raises because the residuals are non-finite. That is a
                # numerically bad but structurally valid try: score the
                # unfitted expression instead of discarding it, so its terrible
                # r2 competes with the other tries rather than vanishing.
                try:
                    candidate = expr.fit(X, y)
                except (ValueError, FloatingPointError, ArithmeticError):
                    candidate = expr

                # Overflow during evaluation can make r2 nan/inf; suppress the
                # warning and coerce non-finite scores to -inf so the try never
                # wins over a finite-scored one but is still counted.
                with np.errstate(all="ignore"):
                    score = r2(candidate, X, y)
                if not np.isfinite(score):
                    score = -np.inf

                if score > best_r2:
                    best_r2 = score
                    best_expr = candidate

            # Skeletons are always structurally valid, so last_expr is set
            # whenever any try ran. Fall back to it only if every try scored
            # -inf (all overflowed) and none beat the initial -inf.
            results.append(best_expr if best_expr is not None else last_expr)
        return results
