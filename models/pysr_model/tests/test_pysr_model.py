import numpy as np
import pytest

from pysr_model import PySRModel
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

# PySR emits numpy warnings on overflowing intermediate candidates.
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


class TestConstruction:
    def test_default_opset_builds_operator_lists(self):
        model = PySRModel(opset=_opset())
        assert set(model._binary) == {"+", "-", "*", "/"}
        assert set(model._unary) == {"sin", "exp"}


class TestConstructionUnmappedOperator:
    def test_unknown_operator_raises_value_error(self):
        opset = OperatorSet(operators={"cos": (1, np.cos)})
        with pytest.raises(ValueError, match="cos"):
            PySRModel(opset=opset)


class TestFitShape:
    def test_returns_one_result_per_problem(self):
        problems = _pipeline_problems(n=3, seed=7)
        model = PySRModel(opset=_opset(), niterations=5, rng=np.random.default_rng(0))
        results = model.fit(problems)
        assert len(results) == len(problems)


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
        model = PySRModel(opset=_opset(), niterations=40, maxsize=20, rng=np.random.default_rng(0))
        (result,) = model.fit([(X, y)])
        assert result is not None
        assert r2([result], [X], [y])[0] > 0.95


class TestResultsAreValid:
    def test_results_are_expressions_and_evaluable(self):
        problems = _pipeline_problems(n=3, seed=23)
        model = PySRModel(opset=_opset(), niterations=5, rng=np.random.default_rng(5))
        results = model.fit(problems)
        for (X, _y), r in zip(problems, results, strict=True):
            assert isinstance(r, Expression)
            out = r.evaluate(X)
            assert out.shape == (X.shape[0],)


class TestEmptyBatch:
    def test_empty_batch_returns_empty_list(self):
        model = PySRModel(opset=_opset(), niterations=5, rng=np.random.default_rng(1))
        assert model.fit([]) == []
