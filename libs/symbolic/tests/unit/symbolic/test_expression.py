import dataclasses

import numpy as np
import pytest

from symbolic.expression import ExpressionBuilder, Kind, OperatorSet, Ref

# opname -> (arity, numpy reference ufunc)
OP_CASES = [
    ("add", 2, np.add),
    ("sub", 2, np.subtract),
    ("mul", 2, np.multiply),
    ("div", 2, np.divide),
    ("sin", 1, np.sin),
    ("exp", 1, np.exp),
]


# --------------------------------------------------------------------------- #
# OperatorSet
# --------------------------------------------------------------------------- #


def _default():
    return OperatorSet.default()


def test_default_has_expected_ops_arities_and_insertion_order():
    op = _default()
    # Insertion order is guaranteed by dict ordering; assert exact ordering + arities.
    assert list(op.operators.keys()) == ["add", "sub", "mul", "div", "sin", "exp"]
    for name in ("add", "sub", "mul", "div"):
        assert op[name][0] == 2, f"{name} must be binary"
    for name in ("sin", "exp"):
        assert op[name][0] == 1, f"{name} must be unary"


def test_codes_are_dense_and_in_insertion_order():
    op = _default()
    expected_codes = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sin": 4, "exp": 5}
    for name, code in expected_codes.items():
        assert op.name_to_code(name) == code
    # Dense round-trip: every code in 0..len-1 maps back to the right name.
    inv = {c: n for n, c in expected_codes.items()}
    for code in range(len(op.operators)):
        assert op.code_to_name(code) == inv[code]


def test_name_to_code_and_code_to_name_are_inverses():
    op = _default()
    for name in op.operators:
        assert op.code_to_name(op.name_to_code(name)) == name
    for code in range(len(op.operators)):
        assert op.name_to_code(op.code_to_name(code)) == code


def test_by_index_matches_getitem_for_every_op():
    op = _default()
    for code in range(len(op.operators)):
        name = op.code_to_name(code)
        assert op.by_index(code) == op[name]


def test_getitem_unknown_opname_raises_keyerror():
    op = _default()
    with pytest.raises(KeyError):
        _ = op["nope"]


def test_code_to_name_out_of_range_raises_keyerror():
    op = _default()
    with pytest.raises(KeyError):
        _ = op.code_to_name(len(op.operators))


def test_operatorset_is_frozen():
    op = _default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.operators = {}


# --------------------------------------------------------------------------- #
# Expression.evaluate  (vectorized over samples)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "opname,arity,ufunc",
    OP_CASES,
    ids=[c[0] for c in OP_CASES],
)
def test_evaluate_single_op_matches_numpy_ufunc(opname, arity, ufunc):
    rng = np.random.default_rng(42)
    num_inputs = 2
    n_samples = 64
    # Use a nonzero range so div stays finite and deterministic.
    X = rng.uniform(0.5, 2.0, size=(num_inputs, n_samples))

    b = ExpressionBuilder(OperatorSet.default(), num_inputs)
    x0, x1 = b.input(0), b.input(1)
    out = b.apply(opname, x0, x1) if arity == 2 else b.apply(opname, x0)
    expr = b.build(out)

    result = expr.evaluate(X)
    assert result.shape == (n_samples,)
    expected = ufunc(X[0], X[1]) if arity == 2 else ufunc(X[0])
    np.testing.assert_allclose(result, expected)


def test_evaluate_output_shape_always_num_samples():
    rng = np.random.default_rng(1)
    for n_samples in (1, 5, 50):
        X = rng.standard_normal((2, n_samples))
        b = ExpressionBuilder(OperatorSet.default(), 2)
        out = b.apply("add", b.input(0), b.input(1))
        result = b.build(out).evaluate(X)
        assert result.shape == (n_samples,)


def test_evaluate_constant_broadcasts_across_samples():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(2, 32))
    b = ExpressionBuilder(OperatorSet.default(), 2)
    out = b.apply("add", b.input(0), b.constant(3.0))
    result = b.build(out).evaluate(X)
    np.testing.assert_allclose(result, X[0] + 3.0)


def test_evaluate_single_sample_shape_and_value():
    X = np.array([[2.0], [5.0]])
    b = ExpressionBuilder(OperatorSet.default(), 2)
    out = b.apply("mul", b.input(0), b.input(1))
    result = b.build(out).evaluate(X)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [10.0])


def test_evaluate_div_elementwise():
    rng = np.random.default_rng(11)
    X = rng.uniform(0.5, 3.0, size=(2, 40))
    b = ExpressionBuilder(OperatorSet.default(), 2)
    out = b.apply("div", b.input(0), b.input(1))
    result = b.build(out).evaluate(X)
    np.testing.assert_allclose(result, np.divide(X[0], X[1]))


def test_evaluate_identity_output_zero_commands():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3, 25))
    b = ExpressionBuilder(OperatorSet.default(), 3)
    expr = b.build(b.input(0))
    assert len(expr.commands) == 0
    result = expr.evaluate(X)
    np.testing.assert_allclose(result, X[0])


def test_evaluate_constant_output_zero_commands():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((2, 20))
    b = ExpressionBuilder(OperatorSet.default(), 2)
    c = b.constant(-2.5)
    expr = b.build(c)
    assert len(expr.commands) == 0
    assert len(expr.constants) == 1
    result = expr.evaluate(X)
    assert result.shape == (20,)
    np.testing.assert_allclose(result, np.full(20, -2.5))


def test_evaluate_zero_constants_expression():
    # No constants at all -> exercises constants[:, np.newaxis] on empty array.
    rng = np.random.default_rng(5)
    X = rng.standard_normal((2, 18))
    b = ExpressionBuilder(OperatorSet.default(), 2)
    out = b.apply("add", b.input(0), b.input(1))
    expr = b.build(out)
    assert len(expr.constants) == 0
    result = expr.evaluate(X)
    np.testing.assert_allclose(result, X[0] + X[1])


def test_evaluate_output_index_decoupling_from_last_command():
    # Regression: an old `return memory[-1]` bug returned the LAST command's
    # value even when output_index pointed at an earlier command.
    rng = np.random.default_rng(9)
    X = rng.uniform(1, 2, size=(2, 30))
    b = ExpressionBuilder(OperatorSet.default(), 2)
    x0, x1 = b.input(0), b.input(1)
    first = b.apply("add", x0, x1)        # cmd 0  -> output target
    _trailing = b.apply("mul", x0, x1)    # cmd 1  -> dead trailing node
    expr = b.build(first)

    # output_index must point at cmd 0's slot, not the last slot.
    cmd_base = expr.constants.shape[0] + 2  # num_inputs(2) + constants(0)
    assert expr.output_index == cmd_base + 0

    result = expr.evaluate(X)
    np.testing.assert_allclose(result, X[0] + X[1])  # add, NOT mul
    # And it must NOT equal the trailing mul result (guards against regression).
    assert not np.allclose(result, X[0] * X[1])


# --------------------------------------------------------------------------- #
# ExpressionBuilder
# --------------------------------------------------------------------------- #


def test_input_returns_correct_ref():
    b = ExpressionBuilder(OperatorSet.default(), 3)
    assert b.input(0) == Ref(Kind.input, 0)
    assert b.input(2) == Ref(Kind.input, 2)


def test_input_out_of_range_raises_indexerror():
    b = ExpressionBuilder(OperatorSet.default(), 3)
    for bad in (-1, 3, 99):
        with pytest.raises(IndexError):
            b.input(bad)


def test_constant_returns_increasing_seq():
    b = ExpressionBuilder(OperatorSet.default(), 1)
    refs = [b.constant(v) for v in (1.0, 2.0, 3.0)]
    assert refs == [Ref(Kind.const, 0), Ref(Kind.const, 1), Ref(Kind.const, 2)]


def test_apply_arity_mismatch_raises_valueerror():
    b = ExpressionBuilder(OperatorSet.default(), 2)
    x0, x1 = b.input(0), b.input(1)
    with pytest.raises(ValueError):
        b.apply("add", x0)            # needs 2, got 1
    with pytest.raises(ValueError):
        b.apply("add", x0, x0, x1)    # needs 2, got 3
    with pytest.raises(ValueError):
        b.apply("sin", x0, x1)        # needs 1, got 2


def test_apply_unknown_opname_raises_keyerror():
    b = ExpressionBuilder(OperatorSet.default(), 2)
    with pytest.raises(KeyError):
        b.apply("nope", b.input(0), b.input(1))


def test_apply_returns_increasing_cmd_refs():
    b = ExpressionBuilder(OperatorSet.default(), 2)
    x0, x1 = b.input(0), b.input(1)
    r0 = b.apply("add", x0, x1)
    r1 = b.apply("mul", x0, x1)
    assert r0 == Ref(Kind.cmd, 0)
    assert r1 == Ref(Kind.cmd, 1)


def test_build_command_chaining():
    # sin(exp(x0)) - a command referencing an earlier command's result.
    rng = np.random.default_rng(2)
    X = rng.uniform(-0.5, 0.5, size=(1, 27))
    b = ExpressionBuilder(OperatorSet.default(), 1)
    x0 = b.input(0)
    e = b.apply("exp", x0)
    out = b.apply("sin", e)
    result = b.build(out).evaluate(X)
    np.testing.assert_allclose(result, np.sin(np.exp(X[0])))


def test_build_arrays_have_correct_dtypes_and_shapes():
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


def test_build_constants_form_contiguous_prefix_block():
    # Constants occupy a contiguous prefix before command results, regardless
    # of interleaving with apply() calls.
    b = ExpressionBuilder(OperatorSet.default(), 2)
    x0, x1 = b.input(0), b.input(1)
    c0 = b.constant(10.0)
    m = b.apply("mul", x0, x1)
    b.constant(20.0)        # declared AFTER an apply -> still in const block
    out = b.apply("add", m, c0)  # cmd 1
    expr = b.build(out)

    # inputs: slots 0,1 ; constants: slots 2,3 ; cmds: slots 4,5
    # output is `add` = cmd 1 -> slot 5.
    assert expr.output_index == 5
    # command rows resolve to flat slots: mul uses inputs 0,1; add uses cmd0(slot4) and const0(slot2).
    assert tuple(expr.commands[0][1:]) == (0, 1)
    assert tuple(expr.commands[1][1:]) == (4, 2)


def test_build_deferred_resolution_regression():
    # Regression: declaring constant() AFTER an apply() must not alias memory.
    # An eager-indexing bug made add(mul(x0,x1), const(3.0)) return x0+x1.
    rng = np.random.default_rng(6)
    X = rng.uniform(0.5, 2.0, size=(2, 35))
    b = ExpressionBuilder(OperatorSet.default(), 2)
    x0, x1 = b.input(0), b.input(1)
    m = b.apply("mul", x0, x1)          # apply FIRST
    c = b.constant(3.0)                 # constant declared AFTER apply
    out = b.apply("add", m, c)
    result = b.build(out).evaluate(X)
    np.testing.assert_allclose(result, X[0] * X[1] + 3.0)
    # And it must NOT equal the buggy x0 + x1 result.
    assert not np.allclose(result, X[0] + X[1])
