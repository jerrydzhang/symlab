import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from manifest.model import TransformerModel
from manifest.tokenizer import XValsTokenizer, PAD_ID, NUM_ID
from manifest.data import collate_fn
from symbolic import OperatorSet
from symbolic.expression import ExpressionBuilder

LAMBDA = 1.0


def decomposed_loss(logits, num_preds, token_targets, num_targets):
    # CE over token predictions, ignoring PAD
    flat_logits = logits.flatten(0, 1)
    flat_targets = token_targets.flatten(0, 1)
    ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=PAD_ID)

    # MSE over numeric predictions at NUM positions only
    flat_preds = num_preds.squeeze(-1).flatten(0, 1)
    flat_nums = num_targets.flatten(0, 1)
    num_mask = flat_targets == NUM_ID
    if num_mask.any():
        mse = F.mse_loss(flat_preds[num_mask], flat_nums[num_mask])
    else:
        mse = torch.tensor(0.0, device=logits.device)
    return ce, mse


def token_accuracy(logits, token_targets):
    preds = logits.argmax(dim=-1)
    mask = token_targets != PAD_ID
    correct = (preds == token_targets) & mask
    return correct.sum().item() / mask.sum().clamp(min=1).item()


@torch.no_grad()
def greedy_decode(model, batch, tokenizer, max_len):
    # Greedy autoregressive token generation from the first sample
    data = batch["data"][:1]
    data_mask = batch["data_mask"][:1]
    stats = batch["stats"][:1]
    tokens = torch.full((1, 1), tokenizer.vocab["<BOS>"], dtype=torch.long)
    nums = torch.ones((1, 1))
    mask = torch.ones((1, 1), dtype=torch.bool)

    for _ in range(max_len):
        logits, num_preds = model(data, tokens, nums, data_mask, mask, stats=stats)
        next_id = logits[0, -1].argmax().item()
        if next_id == tokenizer.vocab["<EOS>"]:
            break
        next_num = float(num_preds[0, -1, 0])
        tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)
        nums = torch.cat([nums, torch.tensor([[next_num if next_id == NUM_ID else 1.0]])], dim=1)
        mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.bool)], dim=1)

    # Render to readable string
    parts = []
    for tid, nv in zip(tokens[0].tolist(), nums[0].tolist()):
        name = tokenizer.id_to_token[tid]
        if name == "<NUM>":
            parts.append(f"NUM({nv:.3f})")
        else:
            parts.append(name)
    return " ".join(parts)


def trial(config, tracker):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    opset = OperatorSet.default()
    tokenizer = XValsTokenizer(opset, max_inputs=config["max_inputs"])

    # Build one fixed expression: mul(add(x0, 2.5), x1)
    builder = ExpressionBuilder(opset, config["max_inputs"])
    expr = builder.build(
        builder.apply(
            "mul",
            builder.apply("add", builder.input(0), builder.constant(2.5)),
            builder.input(1),
        )
    )

    # Generate fixed data from this expression
    rng = np.random.default_rng(config["seed"])
    X = rng.uniform(-5, 5, size=(config["n_points"], config["max_inputs"]))
    y = expr.evaluate(X)

    # Log experiment params
    tracker.log_param("expression", str(expr))
    tracker.log_param("max_inputs", config["max_inputs"])
    tracker.log_param("n_points", config["n_points"])
    tracker.log_param("d_model", config["d_model"])
    tracker.log_param("n_enc_layers", config["n_enc_layers"])
    tracker.log_param("n_dec_layers", config["n_dec_layers"])
    tracker.log_param("n_steps", config["n_steps"])
    tracker.log_param("lr", config["lr"])
    tracker.log_param("batch_size", config["batch_size"])

    # Build model
    model = TransformerModel(
        input_dim=config["max_inputs"] + 1,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    tracker.log_param("n_params", n_params)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    warmup = config.get("warmup_steps", 100)
    n_steps = config["n_steps"]
    loss = torch.tensor(float("inf"))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Build batch — same expression repeated batch_size times
    from symbolic.generation.types import Evaluated

    samples = []
    for _ in range(config["batch_size"]):
        samples.append(Evaluated(X=X, y=y, expression=expr, opset=opset))

    model.train()
    for step in range(n_steps):
        batch = collate_fn(samples, tokenizer)

        input_tokens = batch["tokens"][:, :-1]
        input_nums = batch["num_values"][:, :-1]
        target_tokens = batch["tokens"][:, 1:]
        target_nums = batch["num_values"][:, 1:]
        target_mask = batch["token_mask"][:, :-1]

        optimizer.zero_grad()
        logits, num_preds = model(
            batch["data"],
            input_tokens,
            input_nums,
            batch["data_mask"],
            target_mask,
            stats=batch["stats"],
        )
        ce_loss, mse_loss = decomposed_loss(
            logits, num_preds, target_tokens, target_nums
        )
        loss = ce_loss + LAMBDA * mse_loss

        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == n_steps - 1:
            tracker.log_value("loss", loss.item(), step=step)
            tracker.log_value("ce_loss", ce_loss.item(), step=step)
            tracker.log_value("mse_loss", mse_loss.item(), step=step)
            tracker.log_value("token_acc", token_accuracy(logits, target_tokens), step=step)
            tracker.log_value("grad_norm", grad_norm.item(), step=step)
            tracker.log_value("lr", scheduler.get_last_lr()[0], step=step)

        if step % 50 == 0 or step == n_steps - 1:
            model.eval()
            pred_expr = greedy_decode(model, batch, tokenizer, config["max_seq_len"])
            model.train()
            tracker.log_json("pred_expr", pred_expr, step=step)

    final_loss = loss.item()
    tracker.log_value("final_loss", final_loss, step=n_steps)

    # Save model checkpoint as artifact
    import os
    import tempfile
    ckpt_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    tracker.log_artifact("model.pt", ckpt_path)

    return {"final_loss": final_loss, "n_params": n_params}
