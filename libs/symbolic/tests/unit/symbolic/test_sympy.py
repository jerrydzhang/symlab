"""Tests for sympy interop on ``Expression`` (``to_sympy`` / ``from_sympy``).

``from_sympy`` is the only path SRBench model strings take into our
``Expression`` DAG, and it does non-trivial folding of sympy's normalized
n-ary ``Add``/``Mul`` and integer ``Pow``. These tests cover:

* round-trips through the public ``Expression.from_sympy`` / ``to_sympy``
  methods,
* the exact command-DAG shape that folding produces (structure, not just
  value),
* operator/symbol rejection outside the opset,
* numeric evaluate-equivalence on random inputs (the strongest check:
  equal values imply an equivalent DAG),
* edge cases (single var, constants, zero-constant folding, shared DAG
  nodes with reuse).

All structural claims below were verified against the current folding
implementation; numpy is the only numeric dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from symbolic.expression import Expression, ExpressionBuilder, OperatorSet

FEATURES = ["a", "b", "c"]
OP = OperatorSet.default()

# Random inputs are drawn from a safe window: bounded away from zero (so no
# division blows up) and small enough that exp() never overflows. Every case
# in this file is well-defined on this window.
_RNG = np.random.default_rng(20240517)
_N_SAMPLES = 256
_X = _RNG.uniform(0.5, 2.0, size=(_N_SAMPLES, len(FEATURES)))


def _symbols() -> dict[str, sp.Symbol]:
    return {n: sp.Symbol(n) for n in FEATURES}


def _numeric(sympy_expr: sp.Expr, X: np.ndarray = _X) -> np.ndarray:
    """Evaluate a sympy expression columnwise on ``X`` with numpy semantics."""
    f = sp.lambdify(FEATURES, sympy_expr, modules="numpy")
    return np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=np.float64)


def _from(src: str) -> Expression:
    return Expression.from_sympy(src, FEATURES, OP)


# Expressions used by multiple parametrized tests. Each entry exercises both
# directions of the round-trip and the numeric path.
ROUND_TRIP_CASES = [
    pytest.param("a + b", id="add"),
    pytest.param("a * b", id="mul"),
    pytest.param("a * b + c", id="mixed"),
    pytest.param("2.0 * a + b", id="with-constant"),
    pytest.param("sin(exp(a))", id="nested-unary"),
    pytest.param("a / b", id="div"),
    pytest.param("a - b", id="sub"),
]


class TestRoundTrip:
    """``from_sympy`` then ``to_sympy`` returns a mathematically equal tree."""

    @pytest.mark.parametrize("src", ROUND_TRIP_CASES)
    def test_to_sympy_inverts_from_sympy(self, src):
        original = sp.sympify(src, locals=_symbols())  # ty: ignore
        back = _from(src).to_sympy(FEATURES)
        # Compare symbolically: ordering/normalization may differ in spelling
        # (e.g. "a/b/c" -> "a/(b*c)") but simplify(difference) == 0 is exact.
        assert sp.simplify(back - original) == 0


class TestEvaluateEquivalence:
    """The real test: ``expr.evaluate(X)`` matches a numeric sympy eval.

    Equal values on random inputs imply a behaviorally-equivalent DAG, which
    subsumes structural round-trip correctness.
    """

    @pytest.mark.parametrize("src", ROUND_TRIP_CASES)
    def test_evaluate_matches_sympy_numeric(self, src):
        original = sp.sympify(src, locals=_symbols())  # ty: ignore
        expr = _from(src)
        got = expr.evaluate(_X)
        assert got.shape == (_N_SAMPLES,)
        np.testing.assert_allclose(got, _numeric(original), rtol=1e-12, atol=1e-12)


class TestFoldingStructure:
    """Folding produces a specific command-DAG shape, verified structurally."""

    def _shape(self, src: str):
        """Return (render, opcode-name list, constants) for a parsed source."""
        expr = _from(src)
        names = [OP.code_to_name(int(row[0])) for row in expr.commands]
        return expr._render(FEATURES), names, list(expr.constants)

    def test_pow_integer_positive_expands_to_mul_chain(self):
        # a**2 -> mul(a, a): exactly one binary command, no pow anywhere.
        render, ops, consts = self._shape("a**2")
        assert render == "mul(a, a)"
        assert ops == ["mul"]
        assert "pow" not in ops
        assert consts == []

    def test_pow_integer_positive_three_factors(self):
        # a**3 -> mul(mul(a, a), a): two commands, no pow.
        render, ops, _ = self._shape("a**3")
        assert render == "mul(mul(a, a), a)"
        assert ops == ["mul", "mul"]
        assert "pow" not in ops

    def test_pow_negative_one_becomes_div_by_one(self):
        # a**-1 -> div(1.0, a): a synthesized 1.0 constant, single div.
        render, ops, consts = self._shape("a**-1")
        assert render == "div(1.0, a)"
        assert ops == ["div"]
        assert consts == [1.0]

    def test_coeff_times_vars_folds_left_with_constant_first(self):
        # sympy normalizes 2*a*b to Mul(2, a, b); folds to mul(mul(2.0, a), b).
        render, ops, consts = self._shape("2*a*b")
        assert render == "mul(mul(2.0, a), b)"
        assert ops == ["mul", "mul"]
        assert consts == [2.0]

    def test_subtraction_routes_through_sub_not_add_neg(self):
        # sympy normalizes a - b to Add(a, Mul(-1, b)); must become a single sub.
        render, ops, consts = self._shape("a - b")
        assert render == "sub(a, b)"
        assert ops == ["sub"]
        assert "add" not in ops
        assert consts == []

    def test_chained_division_folds_left(self):
        # a/b/c -> div(div(a, b), c): two div commands, left-folded.
        render, ops, consts = self._shape("a/b/c")
        assert render == "div(div(a, b), c)"
        assert ops == ["div", "div"]
        assert consts == []

    def test_all_negative_terms_insert_zero_constant(self):
        # -a - b has no positive term -> _walk_add seeds with constant(0.0),
        # then subtracts each. Exercises the "zero constants in the result" path.
        render, ops, consts = self._shape("-a - b")
        assert render == "sub(sub(0.0, a), b)"
        assert ops == ["sub", "sub"]
        assert consts == [0.0]
        # And it still evaluates as -a - b.
        expr = _from("-a - b")
        np.testing.assert_allclose(expr.evaluate(_X), -_X[:, 0] - _X[:, 1])


class TestFoldingEvaluateEquivalence:
    """Folded DAGs must still evaluate identically to the source sympy."""

    @pytest.mark.parametrize(
        "src",
        [
            pytest.param("a**2", id="pow2"),
            pytest.param("a**3", id="pow3"),
            pytest.param("a**-1", id="pow-neg1"),
            pytest.param("2*a*b", id="coeff-mul"),
            pytest.param("a - b", id="sub"),
            pytest.param("a/b/c", id="chained-div"),
            pytest.param("-a - b", id="all-negative"),
        ],
    )
    def test_folded_evaluates_like_source(self, src):
        original = sp.sympify(src, locals=_symbols())  # ty: ignore
        np.testing.assert_allclose(
            _from(src).evaluate(_X), _numeric(original), rtol=1e-12, atol=1e-12
        )


class TestErrors:
    """Operators/symbols outside the opset raise ValueError, never silently."""

    @pytest.mark.parametrize(
        "src",
        [
            pytest.param("sqrt(a)", id="sqrt"),
            pytest.param("log(a)", id="log"),
            pytest.param("a**0.5", id="fractional-pow"),
            pytest.param("a + z", id="unknown-symbol"),
            pytest.param("a**b", id="symbolic-exponent"),
        ],
    )
    def test_unsupported_raises_value_error(self, src):
        with pytest.raises(ValueError):
            _from(src)


class TestEdgeCases:
    def test_single_variable_has_no_commands_and_evaluates_as_column(self):
        expr = _from("a")
        assert len(expr.commands) == 0
        assert len(expr.constants) == 0
        np.testing.assert_allclose(expr.evaluate(_X), _X[:, 0])

    def test_constant_only_has_no_commands_and_broadcasts(self):
        expr = _from("3.0")
        assert len(expr.commands) == 0
        assert list(expr.constants) == [3.0]
        out = expr.evaluate(_X)
        assert out.shape == (_N_SAMPLES,)
        np.testing.assert_allclose(out, np.full(_N_SAMPLES, 3.0))
        # And it round-trips back to a value equal to 3.
        assert sp.simplify(expr.to_sympy(FEATURES) - 3) == 0

    def test_pure_variable_expression_has_zero_constants(self):
        # add(a, b) introduces no constants -> exercises the empty-constants
        # slice in evaluate() and confirms folding doesn't synthesize any.
        expr = _from("a + b")
        assert len(expr.constants) == 0
        np.testing.assert_allclose(expr.evaluate(_X), _X[:, 0] + _X[:, 1])

    def test_shared_subexpression_dag_round_trips_and_evaluates(self):
        # Build a DAG that genuinely *reuses* one node: add(m, m) where
        # m = mul(a, a). to_sympy is memoized over indices, so a shared node
        # must not raise and must render once. sympy collapses a*a + a*a to
        # 2*a**2, so we compare by value, not by structure, on the way back.
        b = ExpressionBuilder(OP, len(FEATURES))
        aa = b.apply("mul", b.input(0), b.input(0))
        shared = b.build(b.apply("add", aa, aa))
        assert len(shared.commands) == 2  # one mul, one add — mul is reused

        rendered = shared.to_sympy(FEATURES)
        assert sp.simplify(rendered - (sp.Symbol("a") ** 2 + sp.Symbol("a") ** 2)) == 0

        # Round-trip the rendered form back through from_sympy and check values.
        back = Expression.from_sympy(rendered, FEATURES, OP)
        expected = _X[:, 0] ** 2 + _X[:, 0] ** 2
        np.testing.assert_allclose(back.evaluate(_X), expected, rtol=1e-12, atol=1e-12)


class TestFeatureNameDefaults:
    """Optional/default ``feature_names`` handling on to_sympy/from_sympy.

    These exercise *our* code paths (default-name inference and slicing), not
    sympy's: equality is checked only via ``sp.simplify(a - b) == 0`` and the
    rest is structural (``num_inputs``, exception types/messages).
    """

    def test_to_sympy_none_applies_default_x_names(self):
        # A 2-input expression rendered with no feature_names uses x0, x1.
        b = ExpressionBuilder(OP, 2)
        expr = b.build(b.apply("add", b.input(0), b.input(1)))

        rendered = str(expr.to_sympy())
        assert "x0" in rendered
        assert "x1" in rendered
        # The default must agree with explicitly passing the same names.
        assert sp.simplify(expr.to_sympy() - expr.to_sympy(["x0", "x1"])) == 0

    def test_to_sympy_extra_feature_names_are_sliced_away(self):
        # Only the first num_inputs names are referenced; extras are ignored.
        b = ExpressionBuilder(OP, 2)
        expr = b.build(b.apply("mul", b.input(0), b.input(1)))

        two = expr.to_sympy(["a", "b"])
        three = expr.to_sympy(["a", "b", "c"])
        assert sp.simplify(two - three) == 0

    def test_to_sympy_too_few_feature_names_raises(self):
        b = ExpressionBuilder(OP, 2)
        expr = b.build(b.apply("add", b.input(0), b.input(1)))
        with pytest.raises(ValueError):
            expr.to_sympy(["a"])

    def test_from_sympy_none_infers_contiguous_range_with_gap(self):
        # x0 + x1 -> num_inputs == 2.
        assert Expression.from_sympy("x0 + x1").num_inputs == 2
        # A gap is preserved: x1 is unused, but the range extends to x2.
        assert Expression.from_sympy("x0*x2").num_inputs == 3

    def test_from_sympy_none_rejects_non_x_int_symbols(self):
        with pytest.raises(ValueError, match=r"x\{int\}"):
            Expression.from_sympy("foo + bar")

    def test_from_sympy_explicit_names_accept_subset(self):
        # Declares 3 features; the expression references only one of them.
        expr = Expression.from_sympy("a + a", ["a", "b", "c"])
        assert expr.num_inputs == 3

    def test_default_to_sympy_and_from_sympy_compose(self):
        # Both defaults together: infer names on the way in, default names on
        # the way out, and the expression survives unchanged.
        e = Expression.from_sympy("x0*x1 + sin(x0)")
        expected = sp.sympify("x0*x1 + sin(x0)")
        assert sp.simplify(e.to_sympy() - expected) == 0


class TestEulerE:
    def test_from_sympy_handles_euler_e(self):
        """sp.E (Euler's number) should become a constant, not raise."""
        expr = Expression.from_sympy("E*x0", ["x0"])
        assert len(expr.constants) == 1
        assert expr.constants[0] == pytest.approx(np.e)
