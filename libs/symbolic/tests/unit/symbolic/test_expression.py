import numpy as np
import pytest

from symbolic.expression import ExpressionBuilder, Kind, OperatorSet, Ref

OP_CASES = [
    ("add", 2, np.add),
    ("sub", 2, np.subtract),
    ("mul", 2, np.multiply),
    ("div", 2, np.divide),
    ("sin", 1, np.sin),
    ("exp", 1, np.exp),
]


def _default():
    return OperatorSet.default()


class TestOperatorSet:
    def test_default_has_expected_ops_arities_and_insertion_order(self):
        op = _default()
        # Insertion order is guaranteed by dict ordering; assert exact ordering + arities.
        assert list(op.operators.keys()) == ["add", "sub", "mul", "div", "sin", "exp"]
        for name in ("add", "sub", "mul", "div"):
            assert op[name][0] == 2, f"{name} must be binary"
        for name in ("sin", "exp"):
            assert op[name][0] == 1, f"{name} must be unary"

    def test_codes_are_dense_and_in_insertion_order(self):
        op = _default()
        expected_codes = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sin": 4, "exp": 5}
        for name, code in expected_codes.items():
            assert op.name_to_code(name) == code
        # Dense round-trip: every code in 0..len-1 maps back to the right name.
        inv = {c: n for n, c in expected_codes.items()}
        for code in range(len(op.operators)):
            assert op.code_to_name(code) == inv[code]

    def test_name_to_code_and_code_to_name_are_inverses(self):
        op = _default()
        for name in op.operators:
            assert op.code_to_name(op.name_to_code(name)) == name
        for code in range(len(op.operators)):
            assert op.name_to_code(op.code_to_name(code)) == code

    def test_by_index_matches_getitem_for_every_op(self):
        op = _default()
        for code in range(len(op.operators)):
            name = op.code_to_name(code)
            assert op.by_index(code) == op[name]

    def test_getitem_unknown_opname_raises_keyerror(self):
        op = _default()
        with pytest.raises(KeyError):
            _ = op["nope"]

    def test_code_to_name_out_of_range_raises_keyerror(self):
        op = _default()
        with pytest.raises(KeyError):
            _ = op.code_to_name(len(op.operators))


class TestExpression:
    @pytest.mark.parametrize(
        "opname,arity,ufunc",
        OP_CASES,
        ids=[c[0] for c in OP_CASES],
    )
    def test_evaluate_single_op_matches_numpy_ufunc(self, opname, arity, ufunc):
        rng = np.random.default_rng(42)
        num_inputs = 2
        n_samples = 64
        # Use a nonzero range so div stays finite and deterministic.
        X = rng.uniform(0.5, 2.0, size=(n_samples, num_inputs))

        b = ExpressionBuilder(OperatorSet.default(), num_inputs)
        x0, x1 = b.input(0), b.input(1)
        out = b.apply(opname, x0, x1) if arity == 2 else b.apply(opname, x0)
        expr = b.build(out)

        result = expr.evaluate(X)
        assert result.shape == (n_samples,)
        expected = ufunc(X[:, 0], X[:, 1]) if arity == 2 else ufunc(X[:, 0])
        np.testing.assert_allclose(result, expected)

    def test_evaluate_output_shape_always_num_samples(self):
        rng = np.random.default_rng(1)
        for n_samples in (1, 5, 50):
            X = rng.standard_normal((n_samples, 2))
            b = ExpressionBuilder(OperatorSet.default(), 2)
            out = b.apply("add", b.input(0), b.input(1))
            result = b.build(out).evaluate(X)
            assert result.shape == (n_samples,)

    def test_evaluate_constant_broadcasts_across_samples(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(-1, 1, size=(32, 2))
        b = ExpressionBuilder(OperatorSet.default(), 2)
        out = b.apply("add", b.input(0), b.constant(3.0))
        result = b.build(out).evaluate(X)
        np.testing.assert_allclose(result, X[:, 0] + 3.0)

    def test_evaluate_single_sample_shape_and_value(self):
        X = np.array([[2.0, 5.0]])
        b = ExpressionBuilder(OperatorSet.default(), 2)
        out = b.apply("mul", b.input(0), b.input(1))
        result = b.build(out).evaluate(X)
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [10.0])

    def test_evaluate_div_elementwise(self):
        rng = np.random.default_rng(11)
        X = rng.uniform(0.5, 3.0, size=(40, 2))
        b = ExpressionBuilder(OperatorSet.default(), 2)
        out = b.apply("div", b.input(0), b.input(1))
        result = b.build(out).evaluate(X)
        np.testing.assert_allclose(result, np.divide(X[:, 0], X[:, 1]))

    def test_evaluate_identity_output_zero_commands(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((25, 3))
        b = ExpressionBuilder(OperatorSet.default(), 3)
        expr = b.build(b.input(0))
        assert len(expr.commands) == 0
        result = expr.evaluate(X)
        np.testing.assert_allclose(result, X[:, 0])

    def test_evaluate_constant_output_zero_commands(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((20, 2))
        b = ExpressionBuilder(OperatorSet.default(), 2)
        c = b.constant(-2.5)
        expr = b.build(c)
        assert len(expr.commands) == 0
        assert len(expr.constants) == 1
        result = expr.evaluate(X)
        assert result.shape == (20,)
        np.testing.assert_allclose(result, np.full(20, -2.5))

    def test_evaluate_zero_constants_expression(self):
        # No constants at all -> exercises constants[:, np.newaxis] on empty array.
        rng = np.random.default_rng(5)
        X = rng.standard_normal((18, 2))
        b = ExpressionBuilder(OperatorSet.default(), 2)
        out = b.apply("add", b.input(0), b.input(1))
        expr = b.build(out)
        assert len(expr.constants) == 0
        result = expr.evaluate(X)
        np.testing.assert_allclose(result, X[:, 0] + X[:, 1])

    def test_repr_and_str_reconstruct_symbolic_form(self):
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        expr = b.build(b.apply("add", x0, x1))
        assert repr(expr) == "Expression(add(x0, x1))"
        assert str(expr) == "add(x0, x1)"

    def test_repr_and_str_distinguish_inputs_from_constants(self):
        # Requires num_inputs: slot 2 is a constant (3.0), not x2.
        b = ExpressionBuilder(OperatorSet.default(), 2)
        expr = b.build(b.apply("add", b.input(0), b.constant(3.0)))
        assert repr(expr) == "Expression(add(x0, 3.0))"
        assert str(expr) == "add(x0, 3.0)"

    def test_repr_and_str_handle_chained_unary_and_constant_output(self):
        b = ExpressionBuilder(OperatorSet.default(), 1)
        e = b.apply("exp", b.input(0))
        expr = b.build(b.apply("sin", e))
        assert repr(expr) == "Expression(sin(exp(x0)))"
        assert str(expr) == "sin(exp(x0))"
        b2 = ExpressionBuilder(OperatorSet.default(), 1)
        expr2 = b2.build(b2.constant(-2.5))
        assert repr(expr2) == "Expression(-2.5)"
        assert str(expr2) == "-2.5"

    def test_repr_and_str_render_shared_subexpression_without_error(self):
        # DAG: add(m, m) — must not raise and must stay memoized.
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        m = b.apply("mul", x0, x1)
        expr = b.build(b.apply("add", m, m))
        assert repr(expr) == "Expression(add(mul(x0, x1), mul(x0, x1)))"
        assert str(expr) == "add(mul(x0, x1), mul(x0, x1))"


class TestExpressionBuilder:
    def test_input_returns_correct_ref(self):
        b = ExpressionBuilder(OperatorSet.default(), 3)
        assert b.input(0) == Ref(Kind.input, 0)
        assert b.input(2) == Ref(Kind.input, 2)

    def test_input_out_of_range_raises_indexerror(self):
        b = ExpressionBuilder(OperatorSet.default(), 3)
        for bad in (-1, 3, 99):
            with pytest.raises(IndexError):
                b.input(bad)

    def test_constant_returns_increasing_seq(self):
        b = ExpressionBuilder(OperatorSet.default(), 1)
        refs = [b.constant(v) for v in (1.0, 2.0, 3.0)]
        assert refs == [Ref(Kind.const, 0), Ref(Kind.const, 1), Ref(Kind.const, 2)]

    def test_apply_arity_mismatch_raises_valueerror(self):
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        with pytest.raises(ValueError):
            b.apply("add", x0)  # needs 2, got 1
        with pytest.raises(ValueError):
            b.apply("add", x0, x0, x1)  # needs 2, got 3
        with pytest.raises(ValueError):
            b.apply("sin", x0, x1)  # needs 1, got 2

    def test_apply_unknown_opname_raises_keyerror(self):
        b = ExpressionBuilder(OperatorSet.default(), 2)
        with pytest.raises(KeyError):
            b.apply("nope", b.input(0), b.input(1))

    def test_apply_returns_increasing_cmd_refs(self):
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        r0 = b.apply("add", x0, x1)
        r1 = b.apply("mul", x0, x1)
        assert r0 == Ref(Kind.cmd, 0)
        assert r1 == Ref(Kind.cmd, 1)

    def test_build_command_chaining(self):
        # sin(exp(x0)) - a command referencing an earlier command's result.
        rng = np.random.default_rng(2)
        X = rng.uniform(-0.5, 0.5, size=(27, 1))
        b = ExpressionBuilder(OperatorSet.default(), 1)
        x0 = b.input(0)
        e = b.apply("exp", x0)
        out = b.apply("sin", e)
        result = b.build(out).evaluate(X)
        np.testing.assert_allclose(result, np.sin(np.exp(X[:, 0])))

    def test_build_arrays_have_correct_dtypes_and_shapes(self):
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        b.constant(1.0)
        b.constant(2.0)
        out = b.apply("add", x0, x1)
        expr = b.build(out)
        assert expr.commands.dtype == np.uint16
        assert expr.commands.shape == (1, 3)
        assert expr.constants.dtype == np.float64
        assert expr.constants.ndim == 1
        assert isinstance(expr.output_index, int)

    def test_build_constants_form_contiguous_prefix_block(self):
        # Constants occupy a contiguous prefix before command results, regardless
        # of interleaving with apply() calls.
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        c0 = b.constant(10.0)
        m = b.apply("mul", x0, x1)
        b.constant(20.0)  # declared AFTER an apply -> still in const block
        out = b.apply("add", m, c0)  # cmd 1
        expr = b.build(out)

        # inputs: slots 0,1 ; constants: slots 2,3 ; cmds: slots 4,5
        # output is `add` = cmd 1 -> slot 5.
        assert expr.output_index == 5
        # command rows resolve to flat slots: mul uses inputs 0,1; add uses cmd0(slot4) and const0(slot2).
        assert tuple(expr.commands[0][1:]) == (0, 1)
        assert tuple(expr.commands[1][1:]) == (4, 2)


class TestFit:
    """``Expression.fit`` optimizes constants via Levenberg-Marquardt."""

    def test_fit_linear_recovers_both_constants(self):
        # add(mul(c0, x), c1) with wrong constants -> fit to 3*x + 2.
        # Built with ExpressionBuilder because from_sympy normalizes
        # coefficients (1.0*x + 0.0 collapses to a single constant).
        b = ExpressionBuilder(OperatorSet.default(), 1)
        c0, c1 = b.constant(1.0), b.constant(0.0)
        expr = b.build(b.apply("add", b.apply("mul", c0, b.input(0)), c1))

        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, size=(200, 1))
        y = 3.0 * X[:, 0] + 2.0

        fitted = expr.fit(X, y)
        np.testing.assert_allclose(fitted.constants, [3.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(fitted.evaluate(X), y, atol=1e-6)

    def test_fit_nonlinear_recovers_constant(self):
        # sin(c0*x) with c0 near target -> fit to sin(2.5*x). LM needs a
        # close start for periodic objectives.
        b = ExpressionBuilder(OperatorSet.default(), 1)
        expr = b.build(b.apply("sin", b.apply("mul", b.constant(2.4), b.input(0))))

        rng = np.random.default_rng(1)
        X = rng.uniform(-2, 2, size=(200, 1))
        y = np.sin(2.5 * X[:, 0])

        fitted = expr.fit(X, y)
        assert fitted.constants[0] == pytest.approx(2.5, abs=1e-4)
        np.testing.assert_allclose(fitted.evaluate(X), y, atol=1e-6)

    def test_fit_does_not_modify_original(self):
        b = ExpressionBuilder(OperatorSet.default(), 1)
        c0, c1 = b.constant(1.0), b.constant(0.0)
        expr = b.build(b.apply("add", b.apply("mul", c0, b.input(0)), c1))
        original = expr.constants.copy()

        rng = np.random.default_rng(2)
        X = rng.uniform(-3, 3, size=(100, 1))
        y = 5.0 * X[:, 0] - 1.0

        fitted = expr.fit(X, y)
        np.testing.assert_array_equal(expr.constants, original)
        # fit actually moved the constants away from the wrong start
        assert not np.allclose(fitted.constants, original)

    def test_fit_zero_constants_returns_copy(self):
        # add(a, a) built directly has zero constants (from_sympy would
        # collapse a + a to 2*a, introducing one).
        b = ExpressionBuilder(OperatorSet.default(), 1)
        expr = b.build(b.apply("add", b.input(0), b.input(0)))
        assert len(expr.constants) == 0

        rng = np.random.default_rng(3)
        X = rng.uniform(-3, 3, size=(50, 1))
        y = rng.standard_normal(50)  # arbitrary target; nothing to fit

        fitted = expr.fit(X, y)
        assert len(fitted.constants) == 0
        assert fitted is not expr
        np.testing.assert_array_equal(fitted.evaluate(X), expr.evaluate(X))

    def test_fit_returns_expression_evaluating_close_to_target(self):
        # exp(c0*x) fit to exp(0.7*x): the returned expression's evaluate
        # must match y to high precision (constant opt actually worked).
        b = ExpressionBuilder(OperatorSet.default(), 1)
        expr = b.build(b.apply("exp", b.apply("mul", b.constant(0.5), b.input(0))))

        rng = np.random.default_rng(4)
        X = rng.uniform(-1, 1, size=(300, 1))
        y = np.exp(0.7 * X[:, 0])

        fitted = expr.fit(X, y)
        np.testing.assert_allclose(fitted.constants, [0.7], atol=1e-6)
        np.testing.assert_allclose(fitted.evaluate(X), y, atol=1e-8)
