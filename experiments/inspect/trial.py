import json
import os
import tempfile

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from manifest.model import TransformerModel
from manifest.tokenizer import XValsTokenizer, PAD_ID, EOS_ID, BOS_ID
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

    # Save checkpoint.
    ckpt_dir = tempfile.mkdtemp()
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    tracker.log_artifact("model.pt", ckpt_path)

    # ===== Inspection: held-out generations =====
    report = _inspect(
        model, tokenizer, opset, tracker, device,
        n_test=config["n_test"], seed=config["seed"],
        max_inputs=config["max_inputs"], max_ops=config["max_ops"],
    )

    return {"final_loss": loss.item(), "n_params": n_params, **report["summary"]}


def _token_kind(token_id, tokenizer, opset):
    # Classify a token: operator (with arity), operand, or special.
    if token_id in (PAD_ID, BOS_ID, EOS_ID):
        return "special", 0
    name = tokenizer.id_to_token[token_id]
    if name == "<NUM>":
        return "operand", 0
    if name.startswith("<X"):
        return "operand", 0
    arity, _ = opset[name]
    return "op", arity


def _stack_analysis(token_ids, tokenizer, opset):
    # Trace the preorder stack-depth counter (slots needed to complete tree).
    # A complete valid tree drives `need` to exactly 0.
    need = 1  # we need one root sub-expression
    depth_traj = []
    for tid in token_ids:
        kind, arity = _token_kind(tid, tokenizer, opset)
        if kind == "special":
            break
        if kind == "op":
            need += arity - 1  # consumed one slot, opened arity children
        else:  # operand
            need -= 1
        depth_traj.append(need)
        if need < 0:
            return {
                "category": "extra_operand",
                "need_at_end": need,
                "len": len(depth_traj),
            }
    if need == 0:
        return {"category": "complete", "need_at_end": 0, "len": len(depth_traj)}
    return {
        "category": "incomplete",
        "need_at_end": need,
        "len": len(depth_traj),
    }


@torch.no_grad()
def _inspect(model, tokenizer, opset, tracker, device, n_test, seed,
             max_inputs, max_ops):
    model.eval()

    # Held-out test expressions from an independent stream.
    test_rng = np.random.default_rng(seed + 1000)
    test_pipeline = (
        Pipeline(
            RandomBinaryTree(
                opset, max_ops=max_ops,
                num_vars=(1, max_inputs), rng=test_rng,
            )
        )
        .then(MantissaExponentConstants(rng=test_rng))
        .then(UniformSamplePoints(rng=test_rng))
        .filter(is_valid())
    )
    test_samples = list(test_pipeline.iter(n_test))

    batch = collate_fn(test_samples, tokenizer)
    data = batch["data"].to(device)
    stats = batch["stats"].to(device)
    data_mask = batch["data_mask"].to(device)

    gen_tokens, gen_nums = model.generate(data, data_mask=data_mask, stats=stats)

    rows = []
    from collections import Counter
    cat_counts = Counter()
    first_mismatch_pos = []

    for i, sample in enumerate(test_samples):
        gt = sample.expression
        gt_tokens, _ = tokenizer.encode(gt)
        gt_str = str(gt)

        tokens_row = gen_tokens[i].tolist()
        nums_row = gen_nums[i].tolist()

        if EOS_ID in tokens_row:
            length = tokens_row.index(EOS_ID) + 1
            ended_eos = True
        else:
            length = len(tokens_row)
            ended_eos = False

        tok_view = tokens_row[:length]
        nums_view = nums_row[:length]

        stack = _stack_analysis(tok_view, tokenizer, opset)
        category = stack["category"]

        # Token-name rendering of generated sequence.
        names = []
        for tid, nv in zip(tok_view, nums_view):
            nm = tokenizer.id_to_token[tid]
            if nm == "<NUM>":
                names.append(f"NUM({nv:.3g})")
            else:
                names.append(nm)

        expr = tokenizer.decode(tok_view, nums_view)
        valid = expr is not None
        r2_val = None
        if valid:
            try:
                y_pred = expr.evaluate(sample.X)
                ss_res = float(np.mean((sample.y - y_pred) ** 2))
                ss_tot = float(np.var(sample.y))
                r2_val = 1.0 - ss_res / max(ss_tot, 1e-9)
            except Exception:
                valid = False
                category = "eval_error"

        # First position where generated diverges from ground truth token stream.
        fmp = None
        for j in range(min(len(tok_view), len(gt_tokens))):
            if tok_view[j] != gt_tokens[j]:
                fmp = j
                break
        if fmp is None:
            if len(tok_view) == len(gt_tokens):
                fmp = -1  # exact match
            else:
                fmp = min(len(tok_view), len(gt_tokens))
        first_mismatch_pos.append(fmp)

        cat_counts[category] += 1
        rows.append({
            "i": i,
            "gt": gt_str,
            "gen": " ".join(names),
            "decoded": str(expr) if valid else None,
            "category": category,
            "ended_eos": ended_eos,
            "need_at_end": stack["need_at_end"],
            "r2": r2_val,
            "first_mismatch": fmp,
            "exact": fmp == -1,
        })

    # Summary.
    n_exact = sum(1 for r in rows if r["exact"])
    n_valid = sum(1 for r in rows if r["decoded"] is not None)
    valid_r2 = [r["r2"] for r in rows if r["r2"] is not None]
    mean_r2 = float(np.mean(valid_r2)) if valid_r2 else 0.0
    n_high_r2 = sum(1 for v in valid_r2 if v > 0.9)

    print("\n" + "=" * 78)
    print(f"INSPECTION REPORT  (n={n_test})")
    print("=" * 78)
    print(f"exact_match      : {n_exact}/{n_test}  ({n_exact/n_test:.1%})")
    print(f"valid (decodable): {n_valid}/{n_test}  ({n_valid/n_test:.1%})")
    print(f"high r2 (>0.9)   : {n_high_r2}/{n_test}")
    print(f"mean r2 (valid)  : {mean_r2:.4f}")
    print(f"failure category : {dict(cat_counts)}")
    mm = [x for x in first_mismatch_pos if x is not None and x >= 0]
    if mm:
        print(
            f"first-mismatch pos (non-exact): "
            f"min={min(mm)} mean={np.mean(mm):.1f} median={np.median(mm):.0f} "
            f"max={max(mm)}"
        )
    print("-" * 78)
    for r in rows:
        tag = "OK " if r["decoded"] is not None else "BAD"
        r2s = f"r2={r['r2']:+.2f}" if r["r2"] is not None else "r2=  -  "
        ex = "*" if r["exact"] else " "
        print(f"[{i:3d}]{ex}[{tag}] {r2s} need={r['need_at_end']:+d} "
              f"eos={'Y' if r['ended_eos'] else 'N'} "
              f"cat={r['category']:<13} fmp={r['first_mismatch']}")
        print(f"     GT : {r['gt']}")
        print(f"     GEN: {r['gen']}")
        if r["decoded"]:
            print(f"     DEC: {r['decoded']}")
    print("=" * 78)

    summary = {
        "n_test": n_test,
        "n_exact": n_exact,
        "n_valid": n_valid,
        "mean_r2_valid": mean_r2,
        "n_high_r2": n_high_r2,
        "categories": dict(cat_counts),
    }
    for k, v in summary.items():
        tracker.log_value(f"inspect_{k}", float(v) if isinstance(v, (int, float)) else 0.0, step=0)

    report_dir = tempfile.mkdtemp()
    report_path = os.path.join(report_dir, "inspection.json")
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)
    tracker.log_artifact("inspection.json", report_path)

    return {"summary": summary, "rows": rows}


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
