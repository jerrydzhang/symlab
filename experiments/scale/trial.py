import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from manifest.model import TransformerModel
from manifest.tokenizer import XValsTokenizer, PAD_ID, NUM_ID, EOS_ID
from manifest.data import collate_fn
from manifest.loss import compute_loss, decomposed_loss
from symbolic import OperatorSet
from symbolic.generation import (
    Pipeline,
    RandomBinaryTree,
    MantissaExponentConstants,
    UniformSamplePoints,
    is_valid,
)
from symbolic.scoring import r2


def trial(config, tracker):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    if config.get("opset", "comprehensive") == "default":
        opset = OperatorSet.default()
    else:
        opset = OperatorSet.comprehensive()
    tokenizer = XValsTokenizer(opset, max_inputs=config["max_inputs"])

    rng = np.random.default_rng(config["seed"])
    pipeline = (
        Pipeline(
            RandomBinaryTree(
                opset,
                max_ops=config["max_ops"],
                num_vars=(1, config["max_inputs"]),
                rng=rng,
            )
        )
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(rng=rng))
        .filter(is_valid())
    )

    for key in [
        "seed", "max_inputs", "max_ops", "max_seq_len",
        "d_model", "n_heads", "d_ff", "n_enc_layers", "n_dec_layers",
        "n_steps", "batch_size", "lr", "weight_decay", "warmup_steps",
        "lambda_", "val_every", "log_every",
    ]:
        tracker.log_param(key, config[key])

    model = TransformerModel(
        input_dim=config["max_inputs"] + 1,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        dropout=config.get("dropout", 0.1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    tracker.log_param("n_params", n_params)
    tracker.log_param("vocab_size", len(tokenizer.vocab))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}, Params: {n_params:,}")

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    warmup = config["warmup_steps"]
    n_steps = config["n_steps"]
    loss = torch.tensor(float("inf"))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    use_bf16 = device.type == "cuda"

    val_samples = list(pipeline.iter(config["val_batch_size"]))

    model.train()
    for step in range(n_steps):
        samples = list(pipeline.iter(config["batch_size"]))
        batch = collate_fn(samples, tokenizer)

        data = batch["data"].to(device)
        stats = batch["stats"].to(device)
        tokens = batch["tokens"].to(device)
        num_values = batch["num_values"].to(device)
        data_mask = batch["data_mask"].to(device)
        token_mask = batch["token_mask"].to(device)

        input_tokens = tokens[:, :-1]
        input_nums = num_values[:, :-1]
        target_tokens = tokens[:, 1:]
        target_nums = num_values[:, 1:]
        target_mask = token_mask[:, :-1]

        optimizer.zero_grad()

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, num_preds = model(
                    data, input_tokens, input_nums,
                    data_mask=data_mask, token_mask=target_mask,
                    stats=stats,
                )
                loss = compute_loss(
                    logits, num_preds, target_tokens, target_nums,
                    lambda_=config["lambda_"],
                )
        else:
            logits, num_preds = model(
                data, input_tokens, input_nums,
                data_mask=data_mask, token_mask=target_mask,
                stats=stats,
            )
            loss = compute_loss(
                logits, num_preds, target_tokens, target_nums,
                lambda_=config["lambda_"],
            )

        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % config["log_every"] == 0 or step == n_steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = target_tokens != PAD_ID
                acc = (preds == target_tokens)[mask].float().mean().item()
                ce, mse = decomposed_loss(
                    logits, num_preds, target_tokens, target_nums,
                )

            tracker.log_value("loss", loss.item(), step=step)
            tracker.log_value("ce_loss", ce.item(), step=step)
            tracker.log_value("mse_loss", mse.item(), step=step)
            tracker.log_value("token_acc", acc, step=step)
            tracker.log_value("grad_norm", grad_norm.item(), step=step)
            tracker.log_value("lr", scheduler.get_last_lr()[0], step=step)
            print(
                f"step {step:5d}/{n_steps}  loss={loss.item():.4f}  "
                f"ce={ce.item():.4f}  mse={mse.item():.2f}  "
                f"acc={acc:.3f}  gn={grad_norm.item():.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if step % config["val_every"] == 0 or step == n_steps - 1:
            _evaluate(model, val_samples, tokenizer, tracker, step, device)

    import tempfile, os
    ckpt_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    tracker.log_artifact("model.pt", ckpt_path)

    return {"final_loss": loss.item(), "n_params": n_params}


@torch.no_grad()
def _evaluate(model, samples, tokenizer, tracker, step, device):
    model.eval()
    batch = collate_fn(samples, tokenizer)

    data = batch["data"].to(device)
    stats = batch["stats"].to(device)
    data_mask = batch["data_mask"].to(device)

    gen_tokens, gen_nums = model.generate(data, data_mask=data_mask, stats=stats)

    expressions = []
    Xs = []
    ys = []
    n_valid = 0
    n_invalid = 0

    for i, sample in enumerate(samples):
        tokens_row = gen_tokens[i].tolist()
        nums_row = gen_nums[i].tolist()

        if EOS_ID in tokens_row:
            length = tokens_row.index(EOS_ID) + 1
        else:
            length = len(tokens_row)

        expr = tokenizer.decode(tokens_row[:length], nums_row[:length])

        if expr is None:
            n_invalid += 1
            continue

        try:
            expr.evaluate(sample.X)
        except Exception:
            n_invalid += 1
            continue

        n_valid += 1
        expressions.append(expr)
        Xs.append(sample.X)
        ys.append(sample.y)

    if expressions:
        r2_scores = r2(expressions, Xs, ys)
        mean_r2 = float(np.mean(r2_scores))
    else:
        mean_r2 = 0.0

    n_total = n_valid + n_invalid
    valid_rate = n_valid / max(n_total, 1)

    tracker.log_value("val_r2", mean_r2, step=step)
    tracker.log_value("val_valid_rate", valid_rate, step=step)
    tracker.log_value("val_n_valid", float(n_valid), step=step)
    tracker.log_value("val_n_invalid", float(n_invalid), step=step)

    print(
        f"  [val step {step}] r2={mean_r2:.4f}  valid={valid_rate:.2f}  "
        f"({n_valid}/{n_total})"
    )
    model.train()
