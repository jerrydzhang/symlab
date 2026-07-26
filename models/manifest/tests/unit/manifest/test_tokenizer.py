import numpy as np
import pytest

from manifest.tokenizer import XValsTokenizer
from symbolic import OperatorSet
from symbolic.expression import ExpressionBuilder


def _tok(max_inputs: int = 2, opset: OperatorSet | None = None) -> XValsTokenizer:
    return XValsTokenizer(opset or OperatorSet.default(), max_inputs)


def _assert_round_trip(tok: XValsTokenizer, expr, seed: int) -> None:
    """Encode then decode ``expr``; assert it survives on render and evaluate."""
    token_ids, numeric_vals = tok.encode(expr)
    decoded = tok.decode(token_ids, numeric_vals)
    assert decoded is not None, "decode returned None for a well-formed expression"
    assert str(decoded) == str(expr)
    X = np.random.default_rng(seed).uniform(-1, 1, size=(64, tok.max_inputs))
    np.testing.assert_allclose(decoded.evaluate(X), expr.evaluate(X))


class TestVocab:
    def test_special_input_and_operator_ids_match_layout(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        assert v["<PAD>"] == 0
        assert v["<BOS>"] == 1
        assert v["<EOS>"] == 2
        assert v["<NUM>"] == 3
        assert v["<X0>"] == 4
        assert v["<X1>"] == 5
        # operators follow opset.keys() insertion order, starting right after inputs
        for i, name in enumerate(OperatorSet.default().keys()):
            assert v[name] == 6 + i
        # regression anchor spelling out the default opset's known order
        assert [v[n] for n in ("add", "sub", "mul", "div", "sin", "exp")] == [
            6,
            7,
            8,
            9,
            10,
            11,
        ]

    def test_id_to_token_is_list_inverse_of_vocab(self):
        tok = _tok(max_inputs=2)
        for name, token_id in tok.vocab.items():
            assert tok.id_to_token[token_id] == name
        assert len(tok.id_to_token) == len(tok.vocab)

    def test_larger_max_inputs_shifts_operator_block(self):
        tok = _tok(max_inputs=4)
        v = tok.vocab
        assert [v[f"<X{i}>"] for i in range(4)] == [4, 5, 6, 7]
        # with four input slots claimed, operators now start at id 8
        for i, name in enumerate(OperatorSet.default().keys()):
            assert v[name] == 8 + i

    def test_custom_opset_uses_its_own_insertion_order(self):
        opset = OperatorSet({"add": (2, np.add), "sin": (1, np.sin)})
        tok = XValsTokenizer(opset, max_inputs=2)
        v = tok.vocab
        assert list(v.keys()) == [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<NUM>",
            "<X0>",
            "<X1>",
            "add",
            "sin",
        ]
        assert v["add"] == 6
        assert v["sin"] == 7


class TestEncode:
    def test_wraps_with_bos_eos_and_equal_lengths(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        token_ids, numeric_vals = tok.encode(b.build(b.apply("add", x0, x1)))
        assert token_ids[0] == tok.vocab["<BOS>"]
        assert token_ids[-1] == tok.vocab["<EOS>"]
        assert len(token_ids) == len(numeric_vals)

    def test_each_node_kind_emits_expected_token_and_numeric(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        expr = b.build(b.apply("mul", b.input(0), b.constant(2.5)))
        v = tok.vocab
        token_ids, numeric_vals = tok.encode(expr)
        # <BOS> mul <X0> <NUM> <EOS>
        assert token_ids == [v["<BOS>"], v["mul"], v["<X0>"], v["<NUM>"], v["<EOS>"]]
        assert numeric_vals == [1.0, 1.0, 1.0, 2.5, 1.0]

    def test_numeric_vals_are_one_everywhere_except_num_positions(self):
        # The constant's value rides on its <NUM> slot; every other position
        # (BOS/EOS, inputs, operators) carries the placeholder 1.0. Decode each
        # position back through id_to_token to locate the <NUM> slots.
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        c = b.constant(3.0)
        expr = b.build(
            b.apply(
                "add",
                b.apply("sin", b.input(0)),
                b.apply("mul", b.input(1), c),
            )
        )
        token_ids, numeric_vals = tok.encode(expr)
        assert len(token_ids) == len(numeric_vals)
        for token_id, val in zip(token_ids, numeric_vals):
            if tok.id_to_token[token_id] == "<NUM>":
                assert val == 3.0
            else:
                assert val == 1.0

    def test_encode_nested_add_sin_yields_full_id_sequence(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        expr = b.build(b.apply("add", b.apply("sin", x0), x1))
        v = tok.vocab
        token_ids, _ = tok.encode(expr)
        assert token_ids == [
            v["<BOS>"],
            v["add"],
            v["sin"],
            v["<X0>"],
            v["<X1>"],
            v["<EOS>"],
        ]

    def test_encode_input_index_equal_to_max_inputs_raises(self):
        # The builder permits index 2 (num_inputs=3), but the tokenizer's
        # max_inputs=2 must reject it at the boundary (index == max_inputs).
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 3)
        expr = b.build(b.apply("sin", b.input(2)))
        with pytest.raises(ValueError):
            tok.encode(expr)


class TestRoundTrip:
    def test_input_only_root(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        _assert_round_trip(tok, b.build(b.input(0)), seed=11)

    def test_constant_only_root(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        _assert_round_trip(tok, b.build(b.constant(-2.5)), seed=12)

    def test_unary_sin(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        _assert_round_trip(tok, b.build(b.apply("sin", b.input(0))), seed=13)

    def test_binary_sub(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        _assert_round_trip(tok, b.build(b.apply("sub", x0, x1)), seed=14)

    def test_nested_mixed_add_sin_mul(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        expr = b.build(b.apply("add", b.apply("sin", x0), b.apply("mul", x1, x0)))
        _assert_round_trip(tok, expr, seed=15)

    def test_repeated_inputs(self):
        tok = _tok(max_inputs=1)
        b = ExpressionBuilder(OperatorSet.default(), 1)
        x0 = b.input(0)
        _assert_round_trip(tok, b.build(b.apply("add", x0, x0)), seed=16)

    def test_constant_leaf(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        expr = b.build(b.apply("mul", b.input(0), b.constant(2.5)))
        _assert_round_trip(tok, expr, seed=17)


class TestDagBecomesTree:
    def test_shared_subexpression_doubles_storage_on_round_trip(self):
        # Occurrence-expanding encode cannot preserve sharing: each preorder
        # reference re-emits the subtree, so the decoded expression is a tree
        # with duplicated commands and constants. This storage growth is the
        # documented, intended behavior, not a bug to fix.
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        shared = b.apply("mul", b.input(0), b.constant(2.0))
        e = b.build(b.apply("add", shared, shared))

        token_ids, numeric_vals = tok.encode(e)
        d = tok.decode(token_ids, numeric_vals)

        assert d is not None
        # render and evaluate still match...
        assert str(d) == str(e)
        X = np.random.default_rng(21).uniform(-1, 1, size=(64, 2))
        np.testing.assert_allclose(d.evaluate(X), e.evaluate(X))
        # ...but storage strictly grows
        assert len(d.commands) > len(e.commands)
        assert len(d.constants) > len(e.constants)
        # the two decoded constants are independent slots holding the same value
        assert len(d.constants) == 2
        np.testing.assert_allclose(d.constants, [2.0, 2.0])


class TestNumInputsPinned:
    def test_decoded_expression_pins_num_inputs_to_max_inputs(self):
        # The expression references only x0, but decode rebuilds via
        # ExpressionBuilder(opset, max_inputs), pinning num_inputs to the
        # tokenizer's max_inputs rather than inferring a smaller arity.
        tok = _tok(max_inputs=4)
        b = ExpressionBuilder(OperatorSet.default(), 4)
        e = b.build(b.apply("sin", b.input(0)))
        d = tok.decode(*tok.encode(e))
        assert d is not None
        assert d.num_inputs == 4


class TestDecodeMalformedReturnsNone:
    """Malformed streams collapse to ``None`` rather than raising."""

    def test_truncated_binary_operator_returns_none(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        token_ids = [v["<BOS>"], v["add"], v["<EOS>"]]
        assert tok.decode(token_ids, [1.0] * len(token_ids)) is None

    def test_binary_with_one_child_missing_returns_none(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        token_ids = [v["<BOS>"], v["add"], v["<X0>"], v["<EOS>"]]
        assert tok.decode(token_ids, [1.0] * len(token_ids)) is None

    def test_trailing_node_after_completed_root_returns_none(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        token_ids = [v["<BOS>"], v["<X0>"], v["<X0>"], v["<EOS>"]]
        assert tok.decode(token_ids, [1.0] * len(token_ids)) is None

    def test_pad_mid_expression_returns_none(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        # add expects two operands; <PAD> breaks the loop before either
        # arrives, leaving an unsatisfied operator frame on the stack.
        token_ids = [v["<BOS>"], v["add"], v["<PAD>"], v["<EOS>"]]
        assert tok.decode(token_ids, [1.0] * len(token_ids)) is None

    def test_bos_eos_only_returns_none(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        token_ids = [v["<BOS>"], v["<EOS>"]]
        assert tok.decode(token_ids, [1.0] * len(token_ids)) is None


class TestDecodeProgrammerErrorsRaise:
    """These are loud caller errors, not malformed streams: they raise."""

    def test_mismatched_lengths_raise(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        token_ids = [v["<BOS>"], v["<EOS>"]]
        with pytest.raises(ValueError):
            tok.decode(token_ids, [1.0, 1.0, 1.0])

    def test_missing_bos_prefix_raises(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        # starts with an operator id rather than <BOS>
        token_ids = [v["add"], v["<X0>"], v["<X1>"], v["<EOS>"]]
        with pytest.raises(ValueError):
            tok.decode(token_ids, [1.0] * len(token_ids))


class TestPadAndEosTermination:
    def test_trailing_pad_after_eos_is_ignored(self):
        tok = _tok(max_inputs=2)
        b = ExpressionBuilder(OperatorSet.default(), 2)
        x0, x1 = b.input(0), b.input(1)
        e = b.build(b.apply("add", x0, x1))
        token_ids, numeric_vals = tok.encode(e)
        # append padding after <EOS>; decode must stop at <EOS> and skip it
        token_ids = token_ids + [tok.vocab["<PAD>"]] * 3
        numeric_vals = numeric_vals + [1.0] * 3
        d = tok.decode(token_ids, numeric_vals)
        assert d is not None
        assert str(d) == str(e)

    def test_pad_terminates_a_complete_expression(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        # a complete expression (add x0 x1) followed by <PAD> instead of <EOS>:
        # root is set and the stack is empty when <PAD> breaks the loop
        token_ids = [v["<BOS>"], v["add"], v["<X0>"], v["<X1>"], v["<PAD>"]]
        d = tok.decode(token_ids, [1.0] * len(token_ids))
        assert d is not None
        assert str(d) == "add(x0, x1)"

    def test_eos_terminated_valid_stream_succeeds(self):
        tok = _tok(max_inputs=2)
        v = tok.vocab
        token_ids = [v["<BOS>"], v["add"], v["<X0>"], v["<X1>"], v["<EOS>"]]
        d = tok.decode(token_ids, [1.0] * len(token_ids))
        assert d is not None
        assert str(d) == "add(x0, x1)"
