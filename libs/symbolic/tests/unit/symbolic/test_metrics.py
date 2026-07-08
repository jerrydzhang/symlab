import numpy as np
import pytest

from symbolic import OperatorSet, ScoreResult, score
from symbolic.expression import ExpressionBuilder


class TestR2:
    def test_r2_is_one_when_predictions_match_truth(self):
        # expr = x0 + x1 reproduces y exactly under the same numpy ops.
        b = ExpressionBuilder(OperatorSet.default(), 2)
        expr = b.build(b.apply("add", b.input(0), b.input(1)))

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        y = X[:, 0] + X[:, 1]

        m = expr.score(X, y)
        assert m.r2 == pytest.approx(1.0)

    def test_r2_is_negative_when_worse_than_mean(self):
        # A constant predictor pinned far from the data is worse than the mean.
        b = ExpressionBuilder(OperatorSet.default(), 1)
        expr = b.build(b.constant(5.0))

        X = np.zeros((4, 1))
        y = np.array([1.0, 2.0, 3.0, 4.0])

        m = expr.score(X, y)
        assert m.r2 < 0.0


class TestErrors:
    def test_mse_and_mae_match_hand_computed_values(self):
        # Constant predictor 2.0 vs y = [1, 2, 4]: residuals [-1, 0, 2].
        b = ExpressionBuilder(OperatorSet.default(), 1)
        expr = b.build(b.constant(2.0))

        X = np.zeros((3, 1))
        y = np.array([1.0, 2.0, 4.0])

        m = expr.score(X, y)
        assert m.mse == pytest.approx(5.0 / 3.0)  # (1 + 0 + 4) / 3
        assert m.mae == pytest.approx(1.0)        # (1 + 0 + 2) / 3
        # var(y) = 14/9 -> r2 = 1 - (5/3)/(14/9) = -1/14
        assert m.r2 == pytest.approx(-1.0 / 14.0)


class TestComplexity:
    def test_counts_inputs_constants_and_commands(self):
        # 2 inputs + 1 constant + 2 commands -> 5
        b = ExpressionBuilder(OperatorSet.default(), 2)
        c = b.constant(3.0)
        prod = b.apply("mul", b.input(0), b.input(1))
        expr = b.build(b.apply("add", prod, c))

        X = np.zeros((4, 2))
        y = np.arange(4, dtype=np.float64)

        m = expr.score(X, y)
        assert m.complexity == 5

    def test_trivial_expression_complexity_equals_num_inputs(self):
        # Just an input: no commands, no constants.
        b = ExpressionBuilder(OperatorSet.default(), 3)
        expr = b.build(b.input(0))

        X = np.zeros((4, 3))
        y = np.arange(4, dtype=np.float64)

        m = expr.score(X, y)
        assert m.complexity == expr.num_inputs == 3


class TestEdgeCases:
    def test_score_works_with_zero_constants(self):
        # add(x0, x1): no constants at all.
        b = ExpressionBuilder(OperatorSet.default(), 2)
        expr = b.build(b.apply("add", b.input(0), b.input(1)))

        assert len(expr.constants) == 0

        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 2))
        y = X[:, 0] + X[:, 1]

        m = expr.score(X, y)
        assert isinstance(m, ScoreResult)
        assert m.r2 == pytest.approx(1.0)
        assert m.complexity == 3  # 1 command + 0 constants + 2 inputs


class TestDelegate:
    def test_free_function_score_matches_method(self):
        # metrics.score is a thin delegate for Expression.score.
        b = ExpressionBuilder(OperatorSet.default(), 2)
        expr = b.build(b.apply("add", b.input(0), b.input(1)))

        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 2))
        y = X[:, 0] + X[:, 1]

        assert score(expr, X, y) == expr.score(X, y)
