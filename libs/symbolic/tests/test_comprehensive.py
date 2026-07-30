"""Tests for the comprehensive ``OperatorSet.comprehensive()`` opset.

Covers the three contracts the expanded opset must satisfy:

* the operator roster (22 ops, exact names + arities + insertion order),
* evaluation: each op's ``Expression.evaluate`` matches its numpy reference on
  inputs drawn from a per-op safe domain (``asin``/``acos``/``atanh`` need
  ``|x| < 1`` while ``acosh`` needs ``x >= 1`` -- no single window fits all, so
  ranges are per op),
* round-trip: ``ExpressionBuilder`` -> ``to_sympy`` -> ``from_sympy`` yields a
  numerically equal DAG (and, where sympy re-spells the operator, the original
  spelling is recovered).
"""

import numpy as np
import pytest

from symbolic import Expression, OperatorSet
from symbolic.expression import ExpressionBuilder

OP = OperatorSet.comprehensive()

# Independent numpy references (do NOT read them back out of the opset -- that
# would make the evaluation test circular). Arity 1.
UNARY = [
    ("sin", np.sin),
    ("cos", np.cos),
    ("tan", np.tan),
    ("asin", np.arcsin),
    ("acos", np.arccos),
    ("atan", np.arctan),
    ("sinh", np.sinh),
    ("cosh", np.cosh),
    ("tanh", np.tanh),
    ("asinh", np.arcsinh),
    ("acosh", np.arccosh),
    ("atanh", np.arctanh),
    ("exp", np.exp),
    ("log", np.log),
    ("sqrt", np.sqrt),
    ("square", np.square),
    ("cube", lambda x: np.power(x, 3)),
    ("abs", np.abs),
]

# Arity 2.
BINARY = [
    ("add", np.add),
    ("sub", np.subtract),
    ("mul", np.multiply),
    ("div", np.divide),
]

# Per-op safe input window. Domain-restricted ops are bounded strictly inside
# their domain so every reference value is finite.
SAFE_RANGE = {
    "sin": (-3.0, 3.0),
    "cos": (-3.0, 3.0),
    "tan": (-1.2, 1.2),  # keep |x| well below pi/2
    "asin": (-0.95, 0.95),
    "acos": (-0.95, 0.95),
    "atan": (-3.0, 3.0),
    "sinh": (-2.0, 2.0),
    "cosh": (-2.0, 2.0),
    "tanh": (-2.0, 2.0),
    "asinh": (-3.0, 3.0),
    "acosh": (1.0, 4.0),
    "atanh": (-0.95, 0.95),
    "exp": (-2.0, 2.0),
    "log": (0.1, 4.0),
    "sqrt": (0.1, 4.0),
    "square": (-2.0, 2.0),
    "cube": (-2.0, 2.0),
    "abs": (-2.0, 2.0),
}

EXPECTED_NAMES = [
    "add", "sub", "mul", "div",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "log", "sqrt", "square", "cube", "abs",
]
BINARY_NAMES = {"add", "sub", "mul", "div"}

_N_SAMPLES = 128
_RNG = np.random.default_rng(20260729)


def _column(lo: float, hi: float) -> np.ndarray:
    return _RNG.uniform(lo, hi, size=_N_SAMPLES)


class TestRoster:
    def test_has_exactly_22_operators(self):
        assert len(OP.operators) == 22

    def test_names_and_insertion_order_match_spec(self):
        assert list(OP.operators.keys()) == EXPECTED_NAMES

    def test_arities_are_correct(self):
        for name in BINARY_NAMES:
            assert OP[name][0] == 2, f"{name} must be arity 2"
        for name in EXPECTED_NAMES:
            if name not in BINARY_NAMES:
                assert OP[name][0] == 1, f"{name} must be arity 1"

    def test_codes_are_dense_and_match_insertion_order(self):
        for code, name in enumerate(EXPECTED_NAMES):
            assert OP.name_to_code(name) == code
            assert OP.code_to_name(code) == name

    def test_default_opset_is_unchanged(self):
        default = OperatorSet.default()
        assert list(default.operators.keys()) == [
            "add", "sub", "mul", "div", "sin", "exp",
        ]


class TestEvaluate:
    @pytest.mark.parametrize(
        "name,ref", UNARY, ids=[c[0] for c in UNARY]
    )
    def test_unary_matches_numpy_reference(self, name, ref):
        lo, hi = SAFE_RANGE[name]
        x = _column(lo, hi)
        X = x[:, np.newaxis]

        b = ExpressionBuilder(OP, 1)
        expr = b.build(b.apply(name, b.input(0)))
        got = expr.evaluate(X)

        assert got.shape == (_N_SAMPLES,)
        np.testing.assert_allclose(got, ref(x), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "name,ref", BINARY, ids=[c[0] for c in BINARY]
    )
    def test_binary_matches_numpy_reference(self, name, ref):
        # Nonzero columns so div stays finite.
        x0 = _column(0.5, 2.0)
        x1 = _column(0.5, 2.0)
        X = np.column_stack([x0, x1])

        b = ExpressionBuilder(OP, 2)
        expr = b.build(b.apply(name, b.input(0), b.input(1)))
        got = expr.evaluate(X)

        assert got.shape == (_N_SAMPLES,)
        np.testing.assert_allclose(got, ref(x0, x1), rtol=1e-12, atol=1e-12)


class TestRoundTrip:
    """build -> to_sympy -> from_sympy must preserve values (and spelling)."""

    @pytest.mark.parametrize("name", EXPECTED_NAMES)
    def test_round_trip_preserves_values(self, name):
        if name in BINARY_NAMES:
            x0 = _column(0.5, 2.0)
            x1 = _column(0.5, 2.0)
            X = np.column_stack([x0, x1])
            names = ["a", "b"]
            b = ExpressionBuilder(OP, 2)
            orig = b.build(b.apply(name, b.input(0), b.input(1)))
        else:
            lo, hi = SAFE_RANGE[name]
            X = _column(lo, hi)[:, np.newaxis]
            names = ["a"]
            b = ExpressionBuilder(OP, 1)
            orig = b.build(b.apply(name, b.input(0)))

        rendered = orig.to_sympy(names)
        back = Expression.from_sympy(rendered, names, OP)

        np.testing.assert_allclose(
            back.evaluate(X), orig.evaluate(X), rtol=1e-12, atol=1e-12
        )

    @pytest.mark.parametrize(
        "name,rendered",
        [
            # sympy re-spells these; from_sympy must recover the original op.
            ("sqrt", "sqrt(a)"),
            ("square", "a**2"),
            ("cube", "a**3"),
            ("abs", "Abs(a)"),
        ],
    )
    def test_re_spelled_op_is_recovered(self, name, rendered):
        b = ExpressionBuilder(OP, 1)
        orig = b.build(b.apply(name, b.input(0)))
        back = Expression.from_sympy(rendered, ["a"], OP)
        assert str(back) == f"{name}(x0)"
        X = _column(*SAFE_RANGE[name])[:, np.newaxis]
        np.testing.assert_allclose(
            back.evaluate(X), orig.evaluate(X), rtol=1e-12, atol=1e-12
        )

    def test_nested_expression_round_trips(self):
        b = ExpressionBuilder(OP, 2)
        nested = b.build(
            b.apply(
                "add",
                b.apply("cos", b.apply("sin", b.input(0))),
                b.apply("tanh", b.apply("square", b.input(1))),
            )
        )
        rendered = nested.to_sympy(["a", "b"])
        back = Expression.from_sympy(rendered, ["a", "b"], OP)

        assert str(back) == "add(cos(sin(x0)), tanh(square(x1)))"
        X = np.column_stack([_column(-1.0, 1.0), _column(0.5, 2.0)])
        np.testing.assert_allclose(
            back.evaluate(X), nested.evaluate(X), rtol=1e-12, atol=1e-12
        )
