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
    from symbolic.generation import (
        Pipeline,
        RandomBinaryTree,
        MantissaExponentConstants,
        UniformSamplePoints,
        is_valid,
    )

    from manifest.model import TransformerModel
    from manifest.loss import compute_loss
    from manifest.tokenizer import XValsTokenizer
    from manifest.data import collate_fn

    return (
        AdamW,
        LambdaLR,
        MantissaExponentConstants,
        OperatorSet,
        Pipeline,
        RandomBinaryTree,
        TransformerModel,
        UniformSamplePoints,
        XValsTokenizer,
        clip_grad_norm_,
        collate_fn,
        compute_loss,
        is_valid,
        math,
        np,
        torch,
    )


@app.cell
def _(
    MantissaExponentConstants,
    OperatorSet,
    Pipeline,
    RandomBinaryTree,
    UniformSamplePoints,
    is_valid,
    np,
):
    rng = np.random.default_rng(7)
    opset = OperatorSet.default()
    pipeline = (
        Pipeline(RandomBinaryTree(opset, max_ops=5, num_vars=(2, 2), rng=rng))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(rng=rng))
        .filter(is_valid())
    )
    return opset, pipeline


@app.cell
def _(XValsTokenizer, opset):
    tokenizer = XValsTokenizer(opset, max_inputs=2)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}")
    print(f"Vocab: {tokenizer.vocab}")
    return tokenizer, vocab_size


@app.cell
def _(TransformerModel, torch, vocab_size):
    # Model config — small for fast iteration
    config = dict(
        input_dim=3,
        vocab_size=vocab_size,
        max_seq_len=48,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_enc_layers=2,
        n_dec_layers=4,
        dropout=0.1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")
    return device, model


@app.cell
def _():
    # Training hyperparameters
    n_steps = 2000
    batch_size = 64
    lr = 3e-4
    weight_decay = 0.01
    warmup_steps = 200
    max_grad_norm = 1.0
    lambda_ = 1.0
    log_every = 50
    return (
        batch_size,
        lambda_,
        log_every,
        lr,
        max_grad_norm,
        n_steps,
        warmup_steps,
        weight_decay,
    )


@app.cell
def _(math, n_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return (lr_lambda,)


@app.cell
def _(AdamW, lr, model, weight_decay):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return (optimizer,)


@app.cell
def _(LambdaLR, lr_lambda, optimizer):
    scheduler = LambdaLR(optimizer, lr_lambda)
    return (scheduler,)


@app.cell
def _(
    batch_size,
    clip_grad_norm_,
    collate_fn,
    compute_loss,
    device,
    lambda_,
    log_every,
    max_grad_norm,
    model,
    n_steps,
    optimizer,
    pipeline,
    scheduler,
    tokenizer,
):
    model.train()
    losses = []
    for step in range(n_steps):
        samples = list(pipeline.iter(batch_size))
        batch = collate_fn(samples, tokenizer)

        data = batch["data"].to(device)
        token_ids = batch["tokens"].to(device)
        num_values = batch["num_values"].to(device)
        data_mask = batch["data_mask"].to(device)
        token_mask = batch["token_mask"].to(device)

        # Teacher forcing shift
        input_tokens = token_ids[:, :-1]
        input_nums = num_values[:, :-1]
        target_tokens = token_ids[:, 1:]
        target_nums = num_values[:, 1:]
        target_mask = token_mask[:, :-1]

        optimizer.zero_grad()
        logits, num_preds = model(data, input_tokens, input_nums, data_mask, target_mask)
        loss = compute_loss(logits, num_preds, target_tokens, target_nums, lambda_=lambda_)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == n_steps - 1:
            recent = sum(losses[-log_every:]) / max(1, len(losses[-log_every:]))
            print(
                f"step {step:4d}/{n_steps}  "
                f"loss {loss.item():.4f}  "
                f"avg({log_every}) {recent:.4f}  "
                f"lr {scheduler.get_last_lr()[0]:.6f}"
            )

    print("Training complete.")
    return


if __name__ == "__main__":
    app.run()
