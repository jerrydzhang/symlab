"""Tests for :class:`random_bfgs.RandomBFGSModel`."""

import numpy as np
import pytest

from random_bfgs import RandomBFGSModel
from symbolic import (
    Expression,
    MantissaExponentConstants,
    OperatorSet,
    Pipeline,
    RandomBinaryTree,
    UniformSamplePoints,
    is_valid,
)
from symbolic.expression import ExpressionBuilder

# Random constant search routinely overflows intermediate values (e.g. exp of a
# large sampled constant). The model scores those tries badly rather than
# discarding them, so overflow is expected behavior here, not a fault.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _opset() -> OperatorSet:
    return OperatorSet.default()


def _pipeline_problems(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate ``(X, y)`` problems via the data-generation pipeline."""
    rng = np.random.default_rng(seed)
    p: Pipeline = (
        Pipeline(RandomBinaryTree(_opset(), max_ops=3, rng=rng))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(lo=-3, hi=3, n=40, rng=rng))
        .filter(is_valid())
    )
    return [(ev.X, ev.y) for ev in p.iter(n)]


class TestFitShape:
    def test_returns_one_expression_per_problem(self):
        problems = _pipeline_problems(n=5, seed=7)
        model = RandomBFGSModel(n_tries=8, rng=np.random.default_rng(0))
        results = model.fit(problems, _opset())
        assert len(results) == len(problems)

    def test_never_returns_none_on_reasonable_data(self):
        problems = _pipeline_problems(n=4, seed=11)
        model = RandomBFGSModel(n_tries=20, rng=np.random.default_rng(3))
        results = model.fit(problems, _opset())
        assert len(results) == len(problems)
        for r in results:
            assert r is not None


class TestFitQuality:
    def _target_problem(self) -> tuple[np.ndarray, np.ndarray]:
        # y = x0 * 2.0 + 1.0
        b = ExpressionBuilder(_opset(), 1)
        expr = b.build(b.apply("add", b.apply("mul", b.input(0), b.constant(2.0)), b.constant(1.0)))
        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 5, size=(100, 1))
        return X, expr.evaluate(X)

    def test_finds_good_fit_for_linear_with_offset(self):
        from symbolic import r2

        X, y = self._target_problem()
        model = RandomBFGSModel(max_ops=4, n_tries=200, rng=np.random.default_rng(0))
        (result,) = model.fit([(X, y)], _opset())
        assert result is not None
        assert r2(result, X, y) > 0.95


class TestResultsAreValid:
    def test_results_are_expressions_and_evaluable(self):
        problems = _pipeline_problems(n=3, seed=23)
        model = RandomBFGSModel(n_tries=12, rng=np.random.default_rng(5))
        results = model.fit(problems, _opset())
        for (X, _y), r in zip(problems, results, strict=True):
            assert isinstance(r, Expression)
            # every result evaluates without error on its problem's X
            out = r.evaluate(X)
            assert out.shape == (X.shape[0],)

    def test_single_problem_batch(self):
        problems = _pipeline_problems(n=1, seed=29)
        model = RandomBFGSModel(n_tries=10, rng=np.random.default_rng(9))
        results = model.fit(problems, _opset())
        assert len(results) == 1
        assert isinstance(results[0], Expression)

    def test_empty_batch_returns_empty_list(self):
        model = RandomBFGSModel(n_tries=4, rng=np.random.default_rng(1))
        assert model.fit([], _opset()) == []


class TestNumericallyBadTriesAreScored:
    """Overflowing tries are scored (terribly), never skipped.

    A structurally valid skeleton whose ``exp(constant)`` overflows is a bad
    try, not an invalid one: its unfitted expression is scored so the terrible
    r2 competes with the other tries. The model must still return a real
    expression in this regime, never ``None``.
    """

    def test_returns_expression_with_overflow_prone_opset(self):
        # max_ops=5 with exp in the default opset routinely produces trees
        # whose exp of a large sampled constant overflows; the model must
        # still return evaluable expressions for every problem.
        problems = _pipeline_problems(n=4, seed=31)
        model = RandomBFGSModel(max_ops=5, n_tries=50, rng=np.random.default_rng(2))
        results = model.fit(problems, _opset())
        assert len(results) == len(problems)
        for (X, _y), r in zip(problems, results, strict=True):
            assert isinstance(r, Expression)
            out = r.evaluate(X)
            assert out.shape == (X.shape[0],)

    def test_fit_failure_is_scored_not_skipped(self, monkeypatch):
        # Force every fit() to raise, simulating universal overflow, and
        # confirm r2 is still invoked for every try (scored, not skipped)
        # and that the result is the unfitted expression, never None.
        import random_bfgs.model as model_mod

        problems = _pipeline_problems(n=2, seed=37)
        n_tries = 6
        calls = {"r2": 0}
        real_r2 = model_mod.r2

        def fake_fit(self, X, y):
            raise ValueError("Residuals are not finite")

        def counting_r2(expression, X, y):
            calls["r2"] += 1
            return real_r2(expression, X, y)

        monkeypatch.setattr(model_mod.Expression, "fit", fake_fit)
        monkeypatch.setattr(model_mod, "r2", counting_r2)

        model = RandomBFGSModel(max_ops=4, n_tries=n_tries, rng=np.random.default_rng(0))
        results = model.fit(problems, _opset())

        # Every try was scored (not skipped): r2 called once per try per problem.
        assert calls["r2"] == len(problems) * n_tries
        # The fallback is the unfitted expression: a real Expression, never None.
        for r in results:
            assert isinstance(r, Expression)
