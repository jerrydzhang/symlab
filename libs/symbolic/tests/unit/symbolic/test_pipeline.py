"""Tests for the data generation pipeline.

Covers: skeleton generation, constant filling, sample shapes, the full
chain, the ``is_valid`` filter, and the type progression enforced by ``ty``.
"""

import numpy as np
import pytest

from symbolic.pipeline import (
    Evaluated,
    MantissaExponentConstants,
    Pipeline,
    Populated,
    RandomBinaryTree,
    Simplify,
    Skeleton,
    UniformSamplePoints,
    is_valid,
)
from symbolic import OperatorSet
from symbolic.expression import ExpressionBuilder


def _opset() -> OperatorSet:
    return OperatorSet.default()


class TestRandomBinaryTree:
    def test_produces_valid_skeletons(self):
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(1))
        for _ in range(100):
            skel = gen(None)
            assert isinstance(skel, Skeleton)
            assert skel.opset is _opset() or skel.opset == _opset()
            assert 1 <= skel.num_inputs <= 2
            assert skel.num_constants > 0
            # num_constants must match the actual constant array length
            assert skel.num_constants == len(skel.expression.constants)
            # the placeholder expression evaluates without error
            X = np.random.default_rng(7).uniform(-1, 1, size=(8, skel.num_inputs))
            y = skel.expression.evaluate(X)
            assert y.shape == (8,)

    def test_respects_num_vars_range(self):
        gen = RandomBinaryTree(
            _opset(), num_vars=(3, 5), max_ops=4, rng=np.random.default_rng(2)
        )
        for _ in range(50):
            skel = gen(None)
            assert 3 <= skel.num_inputs <= 5

    def test_op_count_within_budget_and_at_least_one(self):
        gen = RandomBinaryTree(_opset(), max_ops=3, rng=np.random.default_rng(3))
        counts = []
        for _ in range(50):
            skel = gen(None)
            n = len(skel.expression.commands)
            assert 1 <= n <= 3
            counts.append(n)
        # a uniform sample over [1, 3] should hit more than one value
        assert len(set(counts)) > 1

    def test_placeholder_constants_are_all_zero(self):
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(4))
        skel = gen(None)
        assert np.all(skel.expression.constants == 0.0)


class TestMantissaExponentConstants:
    def test_fills_all_placeholders_nonzero(self):
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(5))
        fill = MantissaExponentConstants(rng=np.random.default_rng(6))
        for _ in range(50):
            skel = gen(None)
            pop = fill(skel)
            assert isinstance(pop, Populated)
            assert len(pop.expression.constants) == len(skel.expression.constants)
            # mantissa >= 0.1 => sampled values are essentially never zero
            assert np.all(pop.expression.constants != 0.0)
            # structure is preserved verbatim
            np.testing.assert_array_equal(
                pop.expression.commands, skel.expression.commands
            )
            assert pop.expression.output_index == skel.expression.output_index
            assert pop.num_inputs == skel.num_inputs

    def test_value_distribution_respects_mantissa_exponent(self):
        # pin mantissa=1.0, exponent=3 => |value| == 1000 exactly, sign varies
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(7))
        fill = MantissaExponentConstants(
            mantissa=(1.0, 1.0), exponent=(3, 3), rng=np.random.default_rng(8)
        )
        pop = fill(gen(None))
        assert np.all(np.abs(pop.expression.constants) == pytest.approx(1000.0))
        assert np.all(np.isin(pop.expression.constants, [-1000.0, 1000.0]))

    def test_exponent_range_bounds_magnitude(self):
        # exponent in [-1, 1], mantissa in (0.1, 1.0] => |value| <= 10
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(9))
        fill = MantissaExponentConstants(
            mantissa=(0.1, 1.0), exponent=(-1, 1), rng=np.random.default_rng(10)
        )
        pop = fill(gen(None))
        assert np.all(np.abs(pop.expression.constants) <= 10.0 + 1e-9)
        assert np.all(np.abs(pop.expression.constants) >= 0.01 - 1e-12)


class TestUniformSamplePoints:
    def test_correct_shapes(self):
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(11))
        pop = MantissaExponentConstants(rng=np.random.default_rng(12))(gen(None))
        sampler = UniformSamplePoints(lo=-5, hi=5, n=50, rng=np.random.default_rng(13))
        ev = sampler(pop)
        assert isinstance(ev, Evaluated)
        assert ev.X.shape == (50, pop.num_inputs)
        assert ev.y.shape == (50,)

    def test_y_matches_direct_evaluation(self):
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(14))
        pop = MantissaExponentConstants(rng=np.random.default_rng(15))(gen(None))
        sampler = UniformSamplePoints(lo=-3, hi=3, n=20, rng=np.random.default_rng(16))
        ev = sampler(pop)
        np.testing.assert_allclose(ev.y, pop.expression.evaluate(ev.X))

    def test_X_stays_within_range(self):
        gen = RandomBinaryTree(_opset(), rng=np.random.default_rng(17))
        pop = MantissaExponentConstants(rng=np.random.default_rng(18))(gen(None))
        ev = UniformSamplePoints(lo=-2, hi=2, n=40, rng=np.random.default_rng(19))(pop)
        assert np.all(ev.X >= -2.0) and np.all(ev.X <= 2.0)


class TestFullPipeline:
    def test_produces_valid_evaluated_entries(self):
        rng = np.random.default_rng(100)
        p: Pipeline = (
            Pipeline(RandomBinaryTree(_opset(), max_ops=4, rng=rng))
            .then(MantissaExponentConstants(rng=rng))
            .then(UniformSamplePoints(lo=-5, hi=5, n=50, rng=rng))
            .filter(is_valid())
        )
        results = list(p.iter(50))
        assert len(results) == 50
        for ev in results:
            assert isinstance(ev, Evaluated)
            assert ev.X.shape == (50, ev.expression.num_inputs)
            assert ev.y.shape == (50,)
            assert not np.any(np.isnan(ev.y))
            assert np.max(np.abs(ev.y)) <= 5e4

    def test_iter_is_lazy(self):
        # a generator is returned, not a materialized list
        rng = np.random.default_rng(101)
        p: Pipeline = (
            Pipeline(RandomBinaryTree(_opset(), max_ops=3, rng=rng))
            .then(MantissaExponentConstants(rng=rng))
            .then(UniformSamplePoints(n=20, rng=rng))
            .filter(is_valid())
        )
        it = p.iter(10)
        assert hasattr(it, "__next__")
        first = next(it)
        assert isinstance(first, Evaluated)


def _evaluated_with(y: np.ndarray) -> Evaluated:
    """Build a throwaway Evaluated carrying the given ``y`` for filter tests."""
    b = ExpressionBuilder(_opset(), 1)
    expr = b.build(b.apply("sin", b.input(0)))
    X = np.zeros((len(y), 1))
    return Evaluated(_opset(), expr, X, np.asarray(y, dtype=np.float64))


class TestIsValid:
    def test_rejects_nan(self):
        assert is_valid()(_evaluated_with(np.array([1.0, np.nan, 2.0]))) is False

    def test_rejects_overflow(self):
        assert (
            is_valid(overflow_threshold=5e4)(_evaluated_with(np.array([1e6]))) is False
        )

    def test_rejects_constant_output(self):
        assert is_valid()(_evaluated_with(np.full(8, 3.0))) is False

    def test_accepts_normal_output(self):
        assert is_valid()(_evaluated_with(np.arange(1.0, 9.0))) is True

    def test_threshold_is_configurable(self):
        # max|y| = 1000: below the default 5e4 but above a tightened threshold.
        # y must be non-constant so the variance check does not fire.
        y = np.array([500.0, 1000.0, 750.0])
        assert is_valid()(_evaluated_with(y)) is True
        assert is_valid(overflow_threshold=1e2)(_evaluated_with(y)) is False


class TestTypeProgression:
    def test_stage_output_types(self):
        rng = np.random.default_rng(200)
        skel = RandomBinaryTree(_opset(), rng=rng)(None)
        assert isinstance(skel, Skeleton)

        pop = MantissaExponentConstants(rng=rng)(skel)
        assert isinstance(pop, Populated)

        ev = UniformSamplePoints(rng=rng)(pop)
        assert isinstance(ev, Evaluated)


class TestSimplify:
    def test_simplify_collapses_constant_arithmetic(self):
        # add(1, 2) -> 3 — constants fold
        b = ExpressionBuilder(OperatorSet.default(), 1)
        expr = b.build(b.apply("add", b.constant(1.0), b.constant(2.0)))
        pop = Populated(opset=OperatorSet.default(), num_inputs=1, expression=expr)
        result = Simplify()(pop)
        # Should be a single constant node
        assert len(result.expression.commands) == 0
        assert result.expression.constants[0] == pytest.approx(3.0)

    def test_simplify_preserves_semantics(self):
        # mul(x0, add(x0, 0)) should simplify to mul(x0, x0) or similar
        # but the simplified expression must evaluate the same as original
        b = ExpressionBuilder(OperatorSet.default(), 1)
        x = b.input(0)
        expr = b.build(b.apply("mul", x, b.apply("add", x, b.constant(0.0))))
        pop = Populated(opset=OperatorSet.default(), num_inputs=1, expression=expr)

        rng = np.random.default_rng(0)
        X = rng.uniform(-5, 5, size=(100, 1))

        result = Simplify()(pop)
        np.testing.assert_allclose(
            result.expression.evaluate(X),
            pop.expression.evaluate(X),
            atol=1e-10,
        )

    def test_simplify_in_pipeline_between_population_and_sampling(self):
        # Verify it slots into the type chain correctly
        rng = np.random.default_rng(42)
        gen = (
            Pipeline(RandomBinaryTree(opset=OperatorSet.default(), max_ops=5, rng=rng))
            .then(MantissaExponentConstants(rng=rng))
            .then(Simplify())
            .then(UniformSamplePoints(n=50, rng=rng))
            .filter(is_valid())
        )
        entries = list(gen.iter(5))
        assert len(entries) >= 1
