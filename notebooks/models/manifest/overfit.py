import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import numpy as np
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    from symbolic import OperatorSet
    from symbolic.expression import ExpressionBuilder
    from symbolic.generation.types import Evaluated

    from manifest.model import TransformerModel
    from manifest.loss import compute_loss
    from manifest.tokenizer import XValsTokenizer, EOS_ID
    from manifest.data import collate_fn

    return (
        AdamW,
        EOS_ID,
        Evaluated,
        ExpressionBuilder,
        LambdaLR,
        OperatorSet,
        TransformerModel,
        XValsTokenizer,
        clip_grad_norm_,
        collate_fn,
        compute_loss,
        math,
        np,
        torch,
    )


@app.cell
def _(ExpressionBuilder, OperatorSet):
    opset = OperatorSet.default()
    max_inputs = 2

    builder = ExpressionBuilder(opset, max_inputs)
    expr = builder.build(
        builder.apply(
            "mul",
            builder.apply("add", builder.input(0), builder.constant(2.5)),
            builder.input(1),
        )
    )
    return expr, max_inputs, opset


@app.cell
def _(expr, np):
    rng = np.random.default_rng(42)
    n_points = 100
    X = rng.uniform(-5, 5, size=(n_points, 2))
    y = expr.evaluate(X)
    print(f"Target: {expr}")
    return X, y


@app.cell
def _(TransformerModel, XValsTokenizer, max_inputs, opset):
    tokenizer = XValsTokenizer(opset, max_inputs=max_inputs)
    vocab_size = len(tokenizer.vocab)

    model = TransformerModel(
        input_dim=max_inputs + 1,
        vocab_size=vocab_size,
        max_seq_len=48,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_enc_layers=2,
        n_dec_layers=4,
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Vocab: {tokenizer.vocab}")
    model
    return model, tokenizer


@app.cell
def _():
    n_steps = 400
    batch_size = 16
    lr = 3e-4
    warmup_steps = 20
    lambda_ = 1.0
    return batch_size, lambda_, lr, n_steps, warmup_steps


@app.cell
def _(AdamW, lr, model):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    return (optimizer,)


@app.cell
def _(LambdaLR, lr_lambda, optimizer):
    scheduler = LambdaLR(optimizer, lr_lambda)
    return (scheduler,)


@app.cell
def _(math, n_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return (lr_lambda,)


@app.cell
def _(
    Evaluated,
    X,
    batch_size,
    clip_grad_norm_,
    collate_fn,
    compute_loss,
    expr,
    lambda_,
    model,
    n_steps,
    np,
    opset,
    optimizer,
    scheduler,
    tokenizer,
    torch,
    y,
):
    torch.manual_seed(42)
    np.random.seed(42)

    _samples = [
        Evaluated(X=X, y=y, expression=expr, opset=opset)
        for _ in range(batch_size)
    ]

    model.train()
    for step in range(n_steps):
        _batch = collate_fn(_samples, tokenizer)

        input_tokens = _batch["tokens"][:, :-1]
        input_nums = _batch["num_values"][:, :-1]
        target_tokens = _batch["tokens"][:, 1:]
        target_nums = _batch["num_values"][:, 1:]
        target_mask = _batch["token_mask"][:, :-1]

        optimizer.zero_grad()
        logits, num_preds = model(
            _batch["data"],
            input_tokens,
            input_nums,
            _batch["data_mask"],
            target_mask,
            stats=_batch["stats"],
        )
        loss = compute_loss(
            logits, num_preds, target_tokens, target_nums, lambda_=lambda_
        )
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == n_steps - 1:
            print(f"step {step:3d}/{n_steps}  loss={loss.item():.6f}")

    print("Training complete.")
    return


@app.cell
def _(
    EOS_ID,
    Evaluated,
    X,
    batch_size,
    collate_fn,
    expr,
    model,
    np,
    opset,
    tokenizer,
    y,
):
    # Verify generation: can the model reproduce the target expression?
    model.eval()

    samples = [
        Evaluated(X=X, y=y, expression=expr, opset=opset)
        for _ in range(batch_size)
    ]
    batch = collate_fn(samples, tokenizer)

    gen_tokens, gen_nums = model.generate(
        batch["data"][:1], batch["data_mask"][:1], stats=batch["stats"][:1]
    )
    tokens_list = gen_tokens[0].tolist()
    nums_list = gen_nums[0].tolist()

    eos_pos = tokens_list.index(EOS_ID) if EOS_ID in tokens_list else len(tokens_list)
    length = eos_pos + 1

    print(f"Generated tokens: {tokens_list[:length]}")
    print(f"Generated nums:   {nums_list[:length]}")

    decoded = tokenizer.decode(tokens_list[:length], nums_list[:length])
    print(f"\nTarget:    {expr}")
    print(f"Generated: {decoded}")

    if decoded is not None:
        y_pred = decoded.evaluate(X)
        max_diff = np.max(np.abs(y - y_pred))
        print(f"Max output diff: {max_diff:.8f}")
    else:
        print("Decode failed — expression is invalid")
    return


if __name__ == "__main__":
    app.run()
