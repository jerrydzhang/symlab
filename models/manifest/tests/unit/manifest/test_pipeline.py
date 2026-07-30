import numpy as np
import torch

from manifest.data import collate_fn
from manifest.loss import compute_loss
from manifest.model import TransformerModel
from manifest.tokenizer import XValsTokenizer
from symbolic import Evaluated, OperatorSet
from symbolic.expression import ExpressionBuilder

OPSET = OperatorSet.default()


def _evaluated(expr, n_data, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_data, 2))
    y = expr.evaluate(X)
    return Evaluated(opset=OPSET, expression=expr, X=X, y=y)


def _samples():
    """Mixed expressions; some carry constants so NUM positions exist."""
    samples = []
    for i, n in enumerate([10, 7, 12, 9]):
        b = ExpressionBuilder(OPSET, 2)
        if i % 2 == 0:
            expr = b.build(
                b.apply(
                    "add",
                    b.apply("sin", b.input(0)),
                    b.apply("mul", b.input(1), b.constant(float(i + 1))),
                )
            )
        else:
            expr = b.build(b.apply("sub", b.input(0), b.input(1)))
        samples.append(_evaluated(expr, n_data=n, seed=i + 1))
    return samples


def _model(vocab):
    torch.manual_seed(0)
    return TransformerModel(
        input_dim=3,
        vocab_size=vocab,
        max_seq_len=64,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_enc_layers=2,
        n_dec_layers=2,
        dropout=0.0,
    )


class TestEndToEnd:
    def test_data_collate_model_loss_backward(self):
        tok = XValsTokenizer(OPSET, max_inputs=2)
        batch = collate_fn(_samples(), tok)
        model = _model(len(tok.vocab))

        logits, num_preds = model(
            batch["data"],
            batch["tokens"],
            batch["num_values"],
            batch["data_mask"],
            batch["token_mask"],
            stats=batch["stats"],
        )
        assert logits.shape[:2] == batch["tokens"].shape
        assert num_preds.shape[:2] == batch["tokens"].shape

        loss = compute_loss(
            logits, num_preds, batch["tokens"], batch["num_values"], lambda_=0.5
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

        loss.backward()
        # encoder, decoder token path, and numeric head all receive gradient
        assert model.encoder.data_proj.weight.grad.abs().sum() > 0
        assert model.decoder.logit_head.weight.grad.abs().sum() > 0
        # NUM positions exist in this batch -> numeric head is trained
        assert model.decoder.numeric_head[0].weight.grad.abs().sum() > 0

    def test_optimizer_step_reduces_loss_on_batch(self):
        tok = XValsTokenizer(OPSET, max_inputs=2)
        batch = collate_fn(_samples(), tok)
        model = _model(len(tok.vocab))
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)

        def run_loss():
            logits, num_preds = model(
                batch["data"],
                batch["tokens"],
                batch["num_values"],
                batch["data_mask"],
                batch["token_mask"],
                stats=batch["stats"],
            )
            return compute_loss(
                logits, num_preds, batch["tokens"], batch["num_values"], lambda_=0.5
            )

        initial = run_loss().item()
        for _ in range(10):
            opt.zero_grad()
            run_loss().backward()
            opt.step()
        final = run_loss().item()
        assert final < initial
