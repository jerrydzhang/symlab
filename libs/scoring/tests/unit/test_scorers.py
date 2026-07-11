"""Tests for the standalone scorers."""

from __future__ import annotations

import numpy as np
import pytest
from typing import Callable

from scoring import complexity, r2
from symbolic import Expression, OperatorSet
from symbolic.expression import ExpressionBuilder


def _build(fn: Callable[[ExpressionBuilder, OperatorSet], Expression]) -> Expression:
    """Run an ``ExpressionBuilder`` lambda and return the built expression."""
    opset = OperatorSet.default()
    b = ExpressionBuilder(opset, 2)
    return fn(b, opset)


# --------------------------------------------------------------------------- #
# r2
# --------------------------------------------------------------------------- #


class TestR2:
    def test_is_one_when_predictions_match(self):
        # add(x0, x1) evaluated on random data is the ground-truth y.
        def fn(b, _opset):
            x0, x1 = b.input(0), b.input(1)
            return b.build(b.apply("add", x0, x1))

        expr = _build(fn)
        rng = np.random.default_rng(0)
        X = rng.uniform(-10, 10, size=(64, 2))
        y = expr.evaluate(X)
        assert r2(expr, X, y) == pytest.approx(1.0)

    def test_is_negative_when_worse_than_mean(self):
        # a constant 1e6 predictor is far below any reasonable mean baseline.
        def fn(b, _opset):
            c = b.constant(1e6)
            return b.build(c)

        expr = _build(fn)
        rng = np.random.default_rng(1)
        X = rng.uniform(-1, 1, size=(32, 2))
        y = rng.uniform(-1, 1, size=32)
        assert r2(expr, X, y) < 0.0

    def test_matches_hand_computed_value(self):
        # constant 2.0 vs y=[1,2,4]: mse=5/3, var(y)=14/9 -> r2 = -1/14.
        def fn(b, _opset):
            c = b.constant(2.0)
            return b.build(c)

        expr = _build(fn)
        y = np.array([1.0, 2.0, 4.0])
        X = np.zeros((3, 2))  # X unused by a constant expression
        assert r2(expr, X, y) == pytest.approx(-1.0 / 14.0)

    def test_constant_target_does_not_divide_by_zero(self):
        # var(y) == 0 is guarded -> returns a finite float, not nan/inf.
        def fn(b, _opset):
            c = b.constant(3.0)
            return b.build(c)

        expr = _build(fn)
        y = np.full(10, 5.0)
        X = np.zeros((10, 2))
        result = r2(expr, X, y)
        assert np.isfinite(result)


# --------------------------------------------------------------------------- #
# complexity
# --------------------------------------------------------------------------- #


class TestComplexity:
    def test_counts_commands_constants_and_inputs(self):
        # 2 inputs + 1 constant + 2 commands (mul, add) -> 5.0
        def fn(b, _opset):
            x0, x1 = b.input(0), b.input(1)
            c = b.constant(1.0)
            out = b.apply("add", b.apply("mul", x0, c), x1)
            return b.build(out)

        expr = _build(fn)
        assert complexity(expr, np.zeros((1, 2)), np.zeros(1)) == 5.0

    def test_trivial_input_only_expression(self):
        # no commands, no constants -> complexity == num_inputs
        def fn(b, _opset):
            return b.build(b.input(0))

        expr = _build(fn)
        assert complexity(expr, np.zeros((1, 2)), np.zeros(1)) == 2.0

    def test_ignores_X_and_y(self):
        # same expression, wildly different data -> identical complexity.
        def fn(b, _opset):
            x0, x1 = b.input(0), b.input(1)
            return b.build(b.apply("add", x0, x1))

        expr = _build(fn)
        a = complexity(expr, np.zeros((1, 2)), np.zeros(1))
        b = complexity(expr, np.ones((999, 2)), np.full(999, 1e9))
        assert a == b


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


class TestReturnType:
    def test_both_return_python_float(self):
        def fn_add(b, _opset):
            x0, x1 = b.input(0), b.input(1)
            return b.build(b.apply("add", x0, x1))

        expr = _build(fn_add)
        rng = np.random.default_rng(2)
        X = rng.uniform(-1, 1, size=(16, 2))
        y = rng.uniform(-1, 1, size=16)

        assert isinstance(r2(expr, X, y), float)
        assert isinstance(complexity(expr, X, y), float)
