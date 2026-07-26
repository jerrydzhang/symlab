import numpy as np
import torch

from manifest.data import collate_fn
from manifest.tokenizer import PAD_ID, XValsTokenizer
from symbolic import Evaluated, OperatorSet
from symbolic.expression import ExpressionBuilder

OPSET = OperatorSet.default()


def _tok():
    return XValsTokenizer(OPSET, max_inputs=2)


def _evaluated(expr, n_data, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_data, 2))
    y = expr.evaluate(X)
    return Evaluated(opset=OPSET, expression=expr, X=X, y=y)


def _add_xy():
    b = ExpressionBuilder(OPSET, 2)
    return b.build(b.apply("add", b.input(0), b.input(1)))


def _mixed_samples():
    """Three expressions with distinct token and data lengths.

    token lengths:  add(x0,x1)=5, mul(sin(x0),mul(x1,2.0))=8, x0=3  -> max 8
    data lengths:   10, 7, 12                                      -> max 12
    """
    b2 = ExpressionBuilder(OPSET, 2)
    e2 = b2.build(
        b2.apply(
            "mul",
            b2.apply("sin", b2.input(0)),
            b2.apply("mul", b2.input(1), b2.constant(2.0)),
        )
    )
    b3 = ExpressionBuilder(OPSET, 2)
    e3 = b3.build(b3.input(0))
    return [
        _evaluated(_add_xy(), n_data=10, seed=1),
        _evaluated(e2, n_data=7, seed=2),
        _evaluated(e3, n_data=12, seed=3),
    ]


class TestDtypes:
    def test_tensor_dtypes_match_contract(self):
        batch = collate_fn(_mixed_samples(), _tok())
        assert batch["data"].dtype == torch.float32
        assert batch["tokens"].dtype == torch.long
        assert batch["num_values"].dtype == torch.float32
        assert batch["data_mask"].dtype == torch.bool
        assert batch["token_mask"].dtype == torch.bool


class TestShapes:
    def test_padded_shapes_match_max_lengths(self):
        samples = _mixed_samples()
        batch = collate_fn(samples, _tok())
        B = len(samples)
        assert batch["data"].shape == (B, 12, 3)  # D = n_inputs(2) + target(1)
        assert batch["tokens"].shape == (B, 8)
        assert batch["num_values"].shape == (B, 8)
        assert batch["data_mask"].shape == (B, 12)
        assert batch["token_mask"].shape == (B, 8)


class TestPadding:
    def test_token_padding_uses_pad_id(self):
        tok = _tok()
        batch = collate_fn(_mixed_samples(), tok)
        tokens = batch["tokens"]
        # everywhere the mask is False, the token must be PAD_ID
        assert (tokens[~batch["token_mask"]] == PAD_ID).all()
        # shortest sample (3 real tokens) padded out to 8 with PAD_ID
        assert tokens[2].tolist() == [
            tok.vocab["<BOS>"],
            tok.vocab["<X0>"],
            tok.vocab["<EOS>"],
            PAD_ID,
            PAD_ID,
            PAD_ID,
            PAD_ID,
            PAD_ID,
        ]

    def test_num_values_padding_uses_one(self):
        batch = collate_fn(_mixed_samples(), _tok())
        num_values = batch["num_values"]
        # padded slots carry the placeholder 1.0
        assert torch.all(num_values[~batch["token_mask"]] == 1.0)
        # the constant 2.0 rides on its <NUM> slot in sample 1
        tok = _tok()
        row1_tokens = batch["tokens"][1]
        num_idx = (row1_tokens == tok.vocab["<NUM>"]).nonzero().flatten()
        assert num_idx.numel() == 1
        assert num_values[1, num_idx[0].item()].item() == 2.0


class TestMaskDerivation:
    def test_token_mask_marks_real_tokens(self):
        batch = collate_fn(_mixed_samples(), _tok())
        mask = batch["token_mask"]
        # real tokens are exactly the non-PAD positions
        assert torch.equal(mask, batch["tokens"] != PAD_ID)
        # shortest sample: 3 real, 5 padded
        assert mask[2].tolist() == [True, True, True, False, False, False, False, False]

    def test_data_mask_marks_real_rows(self):
        batch = collate_fn(_mixed_samples(), _tok())
        mask = batch["data_mask"]
        # sample 1 has 7 data rows -> first 7 True, rest False
        assert mask[1].tolist() == [True] * 7 + [False] * 5
        # sample 2 has 12 data rows -> all True
        assert mask[2].all()


class TestDataContent:
    def test_data_aligns_column_stack_of_X_and_y(self):
        samples = _mixed_samples()
        batch = collate_fn(samples, _tok())
        data = batch["data"]
        for i, sample in enumerate(samples):
            n = sample.X.shape[0]
            expected = torch.from_numpy(np.column_stack((sample.X, sample.y))).float()
            assert torch.allclose(data[i, :n, :], expected, atol=1e-6)
            # padded rows are zero-filled
            assert torch.all(data[i, n:, :] == 0.0)


class TestVariableLength:
    def test_single_sample_requires_no_padding(self):
        sample = _evaluated(_add_xy(), n_data=5, seed=99)
        batch = collate_fn([sample], _tok())
        assert batch["tokens"].shape == (1, 5)
        assert batch["data"].shape == (1, 5, 3)
        assert batch["token_mask"].all()
        assert batch["data_mask"].all()

    def test_batch_with_homogeneous_lengths(self):
        samples = [_evaluated(_add_xy(), n_data=8, seed=s) for s in (1, 2, 3)]
        batch = collate_fn(samples, _tok())
        # all token/data lengths equal -> no padding, masks all True
        assert batch["token_mask"].all()
        assert batch["data_mask"].all()
