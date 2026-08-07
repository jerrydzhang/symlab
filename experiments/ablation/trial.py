"""Shared trial harness for the coupled-vs-skeleton ablation.

Extends ``experiments/probe/trial.py`` with three additions:
  A. ``skeleton_mode`` config flag — zeros all constant values to 1.0 during
     training so the model never sees constant magnitudes (structure-only
     supervision).  The NUM token is still emitted; only its value is masked.
  B. Structure-only cross-entropy metric — CE computed only on non-NUM,
     non-PAD tokens, isolating structure prediction quality.
  C. Post-hoc ``Expression.fit()`` in ``_inspect`` — refits constants on
     (X, y) after decoding, then recomputes R².  Because both coupled and
     skeleton models receive the same fit() pass at eval, any difference in
     ``func_equiv_fit`` is attributable purely to structure quality.
"""

import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from manifest.model import TransformerModel
from manifest.tokenizer import XValsTokenizer, PAD_ID, EOS_ID, BOS_ID, NUM_ID
from manifest.data import collate_fn
from manifest.loss import compute_loss, decomposed_loss
from symbolic import OperatorSet
from symbolic.expression import Expression
from symbolic.generation import (
    Pipeline,
    RandomBinaryTree,
    MantissaExponentConstants,
    UniformSamplePoints,
    Simplify,
    Populated,
    is_valid,
)
from symbolic.scoring import r2

class _SafeSimplify:
    # Canonicalize via sympy but drop (return None) if simplify introduces an
    # operator outside the opset or otherwise fails — keeps the pipeline alive.
    def __call__(self, input: Populated):
        try:
            simplified = input.expression.simplify()
        except Exception:
            return None
        if simplified is None:
            return None
        return Populated(
            opset=input.opset,
            num_inputs=input.num_inputs,
            expression=simplified,
        )


def _build_pipeline(opset, config, rng):
    num_vars = config.get("num_vars", (1, config["max_inputs"]))
    p = Pipeline(
        RandomBinaryTree(
            opset, max_ops=config["max_ops"], num_vars=num_vars, rng=rng,
        )
    ).then(MantissaExponentConstants(rng=rng))
    if config.get("canonicalize", False):
        p = p.then(_SafeSimplify())
        mc = config.get("max_const")
        if mc:
            p = p.filter(
                lambda pop, mc=mc: len(pop.expression.constants) == 0
                or float(np.abs(pop.expression.constants).max()) <= mc
            )
    p = p.then(UniformSamplePoints(rng=rng)).filter(is_valid())
    return p

def trial(config, tracker):
    # Line-buffer stdout so SLURM .out is live-readable via SSH (tracker
    # may be unreachable from compute nodes).
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    tag = config.get("tag", "run")
    metrics_log = []

    if config.get("opset", "comprehensive") == "default":
        opset = OperatorSet.default()
    else:
        opset = OperatorSet.comprehensive()
    tokenizer = XValsTokenizer(opset, max_inputs=config["max_inputs"])

    rng = np.random.default_rng(config["seed"])
    pipeline = _build_pipeline(opset, config, rng)

    for key in [
        "seed", "max_inputs", "max_ops", "max_seq_len",
        "d_model", "n_heads", "d_ff", "n_enc_layers", "n_dec_layers",
        "n_steps", "batch_size", "lr", "weight_decay", "warmup_steps",
        "lambda_", "val_every", "log_every", "canonicalize", "num_vars",
        "skeleton_mode",
    ]:
        tracker.log_param(key, str(config.get(key)))

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
    use_bf16 = device.type == "cuda" and config.get("bf16", True)

    # Optional pre-generated data pool (avoids slow per-step generation,
    # e.g. sympy canonicalization). Holds out val + test slices.
    pool_file = config.get("pool_file")
    test_samples = None
    train_pool = None
    if pool_file:
        with open(pool_file, "rb") as f:
            pool = pickle.load(f)
        n_test = config.get("n_test", 100)
        n_val = config["val_batch_size"]
        val_samples = pool[:n_val]
        test_samples = pool[n_val:n_val + n_test]
        train_pool = pool[n_val + n_test:]
        print(f"Loaded pool {pool_file}: {len(pool)} total "
              f"(train={len(train_pool)}, val={n_val}, test={n_test})")
    else:
        val_samples = list(pipeline.iter(config["val_batch_size"]))

    skeleton_mode = config.get("skeleton_mode", False)
    if skeleton_mode:
        print("SKELETON MODE: all constant values zeroed to 1.0 during training")

    model.train()
    for step in range(n_steps):
        if train_pool is not None:
            idx = np.random.randint(0, len(train_pool), size=config["batch_size"])
            samples = [train_pool[i] for i in idx]
        else:
            samples = list(pipeline.iter(config["batch_size"]))
        batch = collate_fn(samples, tokenizer)

        # --- A. skeleton_mode: zero all constant values to 1.0 ---------------
        # The NUM token is still emitted (token IDs unchanged); only its
        # scalar value is masked.  In the decoder x = token_emb * num_values
        # becomes x = token_emb * 1.0, so NUM embeddings are unscaled.
        if skeleton_mode:
            batch["num_values"] = torch.ones_like(batch["num_values"])

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

                # --- B. structure-only CE (exclude NUM and PAD tokens) ------
                # Isolates structure prediction quality from constant
                # prediction.  A high structure_ce / low overall ce gap means
                # most error is in constant values, not structure.
                structure_mask = (target_tokens != PAD_ID) & (target_tokens != NUM_ID)
                structure_ce_val = float("nan")
                if structure_mask.any():
                    per_pos_ce = F.cross_entropy(
                        logits.flatten(0, 1), target_tokens.flatten(0, 1),
                        ignore_index=PAD_ID, reduction='none',
                    ).reshape_as(target_tokens)
                    structure_ce_val = per_pos_ce[structure_mask].mean().item()

            tracker.log_value("loss", loss.item(), step=step)
            tracker.log_value("ce_loss", ce.item(), step=step)
            tracker.log_value("mse_loss", mse.item(), step=step)
            tracker.log_value("token_acc", acc, step=step)
            tracker.log_value("grad_norm", grad_norm.item(), step=step)
            tracker.log_value("lr", scheduler.get_last_lr()[0], step=step)
            tracker.log_value("structure_ce", structure_ce_val, step=step)
            print(
                f"step {step:5d}/{n_steps}  loss={loss.item():.4f}  "
                f"ce={ce.item():.4f}  struct_ce={structure_ce_val:.4f}  "
                f"mse={mse.item():.2f}  "
                f"acc={acc:.3f}  gn={grad_norm.item():.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )
            metrics_log.append({
                "step": step, "loss": loss.item(), "ce": ce.item(),
                "structure_ce": structure_ce_val,
                "mse": mse.item(), "acc": acc,
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
            })

        if step % config["val_every"] == 0 or step == n_steps - 1:
            _evaluate(model, val_samples, tokenizer, tracker, step, device)

    # Persist checkpoint + results to the project dir (SSH-readable from the
    # login node) so analysis does not depend on the tracking server.
    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))

    report = None
    if config.get("inspect", True):
        report = _inspect(
            model, tokenizer, opset, tracker, device,
            n_test=config.get("n_test", 100), seed=config["seed"],
            max_inputs=config["max_inputs"], max_ops=config["max_ops"],
            num_vars=config.get("num_vars", (1, config["max_inputs"])),
            canonicalize=config.get("canonicalize", False),
            test_samples=test_samples,
        )

    results = {
        "tag": tag, "config": {k: str(v) for k, v in config.items()},
        "n_params": n_params, "final_loss": loss.item(),
        "metrics": metrics_log,
        "summary": report["summary"] if report else None,
        "rows": report["rows"] if report else None,
    }
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"RESULTS WRITTEN: {os.path.join(out_dir, tag + '.json')}")

    summary = report["summary"] if report else {}
    return {"final_loss": loss.item(), "n_params": n_params, **summary}


def _token_kind(token_id, tokenizer, opset):
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
    need = 1
    depth_traj = []
    for tid in token_ids:
        kind, arity = _token_kind(tid, tokenizer, opset)
        if kind == "special":
            break
        if kind == "op":
            need += arity - 1
        else:
            need -= 1
        depth_traj.append(need)
        if need < 0:
            return {"category": "extra_operand", "need_at_end": need,
                    "len": len(depth_traj)}
    if need == 0:
        return {"category": "complete", "need_at_end": 0, "len": len(depth_traj)}
    return {"category": "incomplete", "need_at_end": need, "len": len(depth_traj)}


@torch.no_grad()
def _inspect(model, tokenizer, opset, tracker, device, n_test, seed,
             max_inputs, max_ops, num_vars, canonicalize, test_samples=None):
    model.eval()
    if test_samples is None:
        test_rng = np.random.default_rng(seed + 1000)
        test_cfg = {
            "max_inputs": max_inputs, "max_ops": max_ops,
            "num_vars": num_vars, "canonicalize": canonicalize,
        }
        test_pipeline = _build_pipeline(opset, test_cfg, test_rng)
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

        stack = _stack_analysis(tok_view[1:], tokenizer, opset)
        category = stack["category"]

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
        fitted_r2 = None
        if valid:
            n = sample.X.shape[1]
            if n < tokenizer.max_inputs:
                X_eval = np.column_stack([sample.X, np.zeros((sample.X.shape[0], tokenizer.max_inputs - n))])
            else:
                X_eval = sample.X
            y_pred = expr.evaluate(X_eval)
            if np.isfinite(y_pred).all():
                ss_res = float(np.mean((sample.y - y_pred) ** 2))
                ss_tot = float(np.var(sample.y))
                r2_val = 1.0 - ss_res / max(ss_tot, 1e-9)

                # --- C. post-hoc fit() constant refinement ----------------
                # Refit constants via nonlinear least-squares on (X_eval, y),
                # then recompute R².  Both coupled and skeleton models receive
                # this pass, so differences in fitted_r2 / func_equiv_fit are
                # attributable to structure, not constant quality.
                if len(expr.constants) > 0:
                    try:
                        fitted_expr = expr.fit(X_eval, sample.y)
                        y_fitted = fitted_expr.evaluate(X_eval)
                        if np.isfinite(y_fitted).all():
                            ss_res_f = float(
                                np.mean((sample.y - y_fitted) ** 2)
                            )
                            fitted_r2 = 1.0 - ss_res_f / max(ss_tot, 1e-9)
                    except Exception:
                        pass
                else:
                    fitted_r2 = r2_val
            else:
                valid = False
                category = "eval_error"

        # Functional equivalence: does generated match GT outputs?
        func_equiv = False
        if valid and r2_val is not None:
            func_equiv = r2_val > 0.99

        # Functional equivalence AFTER constant fitting — the key metric.
        func_equiv_fit = False
        if fitted_r2 is not None:
            func_equiv_fit = fitted_r2 > 0.99

        fmp = None
        for j in range(min(len(tok_view), len(gt_tokens))):
            if tok_view[j] != gt_tokens[j]:
                fmp = j
                break
        if fmp is None:
            if len(tok_view) == len(gt_tokens):
                fmp = -1
            else:
                fmp = min(len(tok_view), len(gt_tokens))
        first_mismatch_pos.append(fmp)

        cat_counts[category] += 1
        rows.append({
            "i": i, "gt": gt_str, "gen": " ".join(names),
            "decoded": str(expr) if valid else None, "category": category,
            "ended_eos": ended_eos, "need_at_end": stack["need_at_end"],
            "r2": r2_val, "r2_fit": fitted_r2,
            "first_mismatch": fmp, "exact": fmp == -1,
            "func_equiv": func_equiv, "func_equiv_fit": func_equiv_fit,
        })

    n_exact = sum(1 for r in rows if r["exact"])
    n_valid = sum(1 for r in rows if r["decoded"] is not None)
    n_equiv = sum(1 for r in rows if r["func_equiv"])
    n_equiv_fit = sum(1 for r in rows if r["func_equiv_fit"])
    valid_r2 = [r["r2"] for r in rows if r["r2"] is not None]
    valid_r2_fit = [r["r2_fit"] for r in rows if r["r2_fit"] is not None]
    mean_r2 = float(np.mean(valid_r2)) if valid_r2 else 0.0
    mean_r2_fit = float(np.mean(valid_r2_fit)) if valid_r2_fit else 0.0
    n_high_r2 = sum(1 for v in valid_r2 if v > 0.9)
    n_high_r2_fit = sum(1 for v in valid_r2_fit if v > 0.9)

    print("\n" + "=" * 78)
    print(f"INSPECTION REPORT  (n={n_test})")
    print("=" * 78)
    print(f"exact_match          : {n_exact}/{n_test}  ({n_exact/n_test:.1%})")
    print(f"valid (decodable)    : {n_valid}/{n_test}  ({n_valid/n_test:.1%})")
    print(f"func_equiv (r2>.99)  : {n_equiv}/{n_test}  ({n_equiv/n_test:.1%})")
    print(f"func_equiv_fit (>0.99): {n_equiv_fit}/{n_test}  ({n_equiv_fit/n_test:.1%})")
    print(f"high r2 (>0.9)       : {n_high_r2}/{n_test}")
    print(f"high r2_fit (>0.9)   : {n_high_r2_fit}/{n_test}")
    print(f"mean r2 (valid)      : {mean_r2:.4f}")
    print(f"mean r2_fit (valid)  : {mean_r2_fit:.4f}")
    print(f"failure category     : {dict(cat_counts)}")
    mm = [x for x in first_mismatch_pos if x is not None and x >= 0]
    if mm:
        print(f"first-mismatch pos   : min={min(mm)} mean={np.mean(mm):.1f} "
              f"median={np.median(mm):.0f} max={max(mm)}")
    print("-" * 78)
    for r in rows:
        tag = "OK " if r["decoded"] is not None else "BAD"
        r2s = f"r2={r['r2']:+.2f}" if r["r2"] is not None else "r2=  -  "
        r2fs = f"r2f={r['r2_fit']:+.2f}" if r["r2_fit"] is not None else "r2f= -  "
        ex = "*" if r["exact"] else " "
        eq = "~" if r["func_equiv"] else " "
        eqf = "#" if r["func_equiv_fit"] else " "
        print(f"[{r['i']:3d}]{ex}{eq}{eqf}[{tag}] {r2s} {r2fs} "
              f"need={r['need_at_end']:+d} "
              f"eos={'Y' if r['ended_eos'] else 'N'} "
              f"cat={r['category']:<13} fmp={r['first_mismatch']}")
        print(f"     GT : {r['gt']}")
        print(f"     GEN: {r['gen']}")
        if r["decoded"]:
            print(f"     DEC: {r['decoded']}")
    print("=" * 78)

    summary = {
        "n_test": n_test, "n_exact": n_exact, "n_valid": n_valid,
        "n_func_equiv": n_equiv, "n_func_equiv_fit": n_equiv_fit,
        "mean_r2_valid": mean_r2, "mean_r2_fit_valid": mean_r2_fit,
        "n_high_r2": n_high_r2, "n_high_r2_fit": n_high_r2_fit,
        "categories": str(dict(cat_counts)),
    }
    for k, v in summary.items():
        try:
            tracker.log_value(f"inspect_{k}", float(v), step=0)
        except (TypeError, ValueError):
            pass

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

    # Teacher-forced validation loss
    tokens = batch["tokens"].to(device)
    num_values = batch["num_values"].to(device)
    token_mask = batch["token_mask"].to(device)
    input_tokens = tokens[:, :-1]
    input_nums = num_values[:, :-1]
    target_tokens = tokens[:, 1:]
    target_nums = num_values[:, 1:]
    target_mask = token_mask[:, :-1]

    logits, num_preds = model(
        data, input_tokens, input_nums,
        data_mask=data_mask, token_mask=target_mask, stats=stats,
    )
    val_ce, val_mse = decomposed_loss(
        logits, num_preds, target_tokens, target_nums,
    )

    # Structure-only CE (exclude NUM tokens)
    structure_mask = (target_tokens != PAD_ID) & (target_tokens != NUM_ID)
    if structure_mask.any():
        per_pos_ce = F.cross_entropy(
            logits.flatten(0, 1), target_tokens.flatten(0, 1),
            ignore_index=PAD_ID, reduction="none",
        ).reshape_as(target_tokens)
        val_structure_ce = per_pos_ce[structure_mask].mean().item()
    else:
        val_structure_ce = float("nan")

    tracker.log_value("val_loss", val_ce.item() + val_mse.item(), step=step)
    tracker.log_value("val_ce", val_ce.item(), step=step)
    tracker.log_value("val_structure_ce", val_structure_ce, step=step)
    tracker.log_value("val_mse", val_mse.item(), step=step)

    # Generation metrics
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

        n = sample.X.shape[1]
        if n < tokenizer.max_inputs:
            X_eval = np.column_stack([sample.X, np.zeros((sample.X.shape[0], tokenizer.max_inputs - n))])
        else:
            X_eval = sample.X
        y_pred = expr.evaluate(X_eval)
        if not np.isfinite(y_pred).all():
            n_invalid += 1
            continue

        n_valid += 1
        expressions.append(expr)
        Xs.append(X_eval)
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
