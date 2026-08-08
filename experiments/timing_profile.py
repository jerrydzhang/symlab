#!/usr/bin/env python3
"""
Timing profiler for the symlab probe training pipeline.

Instruments each component of the probe trial's training loop to identify
where wall-clock time is spent per training step. Measures:

  1. Data generation   — pipeline.iter(batch_size) [live] or pool sampling [pool]
  2. collate_fn        — tokenization + tensor batching
  3. Model forward      — model(...) + compute_loss
  4. Backward pass      — loss.backward()
  5. Optimizer + sched  — clip_grad_norm_ + optimizer.step + scheduler.step
  6. Tracker logging    — tracker.log_value (in-process no-op lower bound)
  7. Generation eval    — model.generate() during validation
  8. Decode + evaluate  — tokenizer.decode + expr.evaluate per sample

Runs twice: (a) live generation, (b) pool-based (pools/raw_pool.pkl).
Tiny model config so it finishes fast on CPU. NOT for HPC.
"""

import os
import sys
import time
import pickle
import statistics
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from manifest.model import TransformerModel
from manifest.tokenizer import XValsTokenizer, PAD_ID, EOS_ID
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


# ---------------------------------------------------------------------------
# Tiny in-process tracker — measures the *interface* overhead (the real tracker
# in trial.py does network I/O, so this is a strict lower bound).
# ---------------------------------------------------------------------------
class TimingTracker:
    def log_value(self, name, value, step=0):
        pass

    def log_param(self, name, value):
        pass

    def log_artifact(self, name, path):
        pass


# ---------------------------------------------------------------------------
# Pipeline builder (mirrors experiments/probe/trial.py::_build_pipeline)
# ---------------------------------------------------------------------------
def _build_pipeline(opset, config, rng):
    num_vars = config.get("num_vars", (1, config["max_inputs"]))
    p = Pipeline(
        RandomBinaryTree(
            opset, max_ops=config["max_ops"], num_vars=num_vars, rng=rng,
        )
    ).then(MantissaExponentConstants(rng=rng))
    p = p.then(UniformSamplePoints(rng=rng)).filter(is_valid())
    return p


def make_config():
    """Tiny config that runs fast on CPU."""
    return {
        "seed": 42,
        "tag": "timing_profile",
        "opset": "default",
        "max_inputs": 2,
        "max_ops": 3,
        "num_vars": (1, 2),
        "max_seq_len": 32,
        # --- tiny model per task spec ---
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 128,
        "n_enc_layers": 1,
        "n_dec_layers": 2,
        "dropout": 0.1,
        # --- training ---
        "n_steps": 50,
        "batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "lambda_": 0.0,
        "val_every": 25,
        "log_every": 10,
        "val_batch_size": 16,
        "n_test": 16,
        "bf16": False,
    }


# ---------------------------------------------------------------------------
# Core profiling loop
# ---------------------------------------------------------------------------
def profile_run(config, pool_file=None, n_warmup=3):
    """
    Run a timed training loop and collect per-component timings.

    n_warmup leading steps are executed but excluded from reported averages
    (lazy init / JIT settle). Validation timings are collected on every val
    step and reported separately (they are not part of the per-step total).

    Returns:
        dict with 'timings' (component -> list[seconds]),
                 'step_times' (per-step totals, post-warmup),
                 'n_params', 'vocab_size'.
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    opset = (
        OperatorSet.default()
        if config.get("opset") == "default"
        else OperatorSet.comprehensive()
    )
    tokenizer = XValsTokenizer(opset, max_inputs=config["max_inputs"])
    rng = np.random.default_rng(config["seed"])
    pipeline = _build_pipeline(opset, config, rng)

    device = torch.device("cpu")
    model = TransformerModel(
        input_dim=config["max_inputs"] + 1,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        dropout=config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    warmup = config["warmup_steps"]
    n_steps = config["n_steps"]

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---- load pool or generate val set ----
    train_pool = None
    if pool_file:
        with open(pool_file, "rb") as f:
            pool = pickle.load(f)
        n_val = config["val_batch_size"]
        val_samples = pool[:n_val]
        train_pool = pool[n_val:]
        print(f"  Loaded pool: {len(pool)} total (train={len(train_pool)})")
    else:
        val_samples = list(pipeline.iter(config["val_batch_size"]))

    # ---- timing accumulators ----
    # Per-training-step components (one entry per step)
    timings = defaultdict(list)
    # Validation components (one entry per val step) — kept separate
    val_timings = defaultdict(list)
    tracker = TimingTracker()

    model.train()
    step_times = []  # post-warmup per-step totals

    total_iters = n_steps + n_warmup
    for step in range(total_iters):
        is_warmup = step < n_warmup
        step_t0 = time.perf_counter()

        # 1. DATA GENERATION
        t0 = time.perf_counter()
        if train_pool is not None:
            idx = np.random.randint(0, len(train_pool), size=config["batch_size"])
            samples = [train_pool[i] for i in idx]
        else:
            samples = list(pipeline.iter(config["batch_size"]))
        _dt = time.perf_counter() - t0
        if not is_warmup:
            timings["data_gen"].append(_dt)

        # 2. COLLATE_FN
        t0 = time.perf_counter()
        batch = collate_fn(samples, tokenizer)
        _dt = time.perf_counter() - t0
        if not is_warmup:
            timings["collate"].append(_dt)

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

        # 3. FORWARD + LOSS
        t0 = time.perf_counter()
        logits, num_preds = model(
            data, input_tokens, input_nums,
            data_mask=data_mask, token_mask=target_mask, stats=stats,
        )
        loss = compute_loss(
            logits, num_preds, target_tokens, target_nums,
            lambda_=config["lambda_"],
        )
        torch.cpu.synchronize()
        _dt = time.perf_counter() - t0
        if not is_warmup:
            timings["forward"].append(_dt)

        # 4. BACKWARD
        t0 = time.perf_counter()
        loss.backward()
        torch.cpu.synchronize()
        _dt = time.perf_counter() - t0
        if not is_warmup:
            timings["backward"].append(_dt)

        # 5. OPTIMIZER + SCHEDULER STEP
        t0 = time.perf_counter()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        torch.cpu.synchronize()
        _dt = time.perf_counter() - t0
        if not is_warmup:
            timings["optim_step"].append(_dt)

        # 6. TRACKER LOGGING (on log steps)
        if step % config["log_every"] == 0:
            t0 = time.perf_counter()
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
            _dt = time.perf_counter() - t0
            if not is_warmup:
                timings["tracker"].append(_dt)

        step_total = time.perf_counter() - step_t0
        if not is_warmup:
            step_times.append(step_total)

        # 7 & 8. VALIDATION (generation + decode/evaluate)
        if step % config["val_every"] == 0:
            model.eval()
            vbatch = collate_fn(val_samples, tokenizer)
            vdata = vbatch["data"].to(device)
            vstats = vbatch["stats"].to(device)
            vdata_mask = vbatch["data_mask"].to(device)

            # 7. GENERATION
            t0 = time.perf_counter()
            with torch.no_grad():
                gen_tokens, gen_nums = model.generate(
                    vdata, data_mask=vdata_mask, stats=vstats,
                )
            torch.cpu.synchronize()
            val_timings["gen_eval"].append(time.perf_counter() - t0)

            # 8. DECODE + EVALUATE (per sample)
            t0 = time.perf_counter()
            for i, sample in enumerate(val_samples):
                tokens_row = gen_tokens[i].tolist()
                nums_row = gen_nums[i].tolist()
                if EOS_ID in tokens_row:
                    length = tokens_row.index(EOS_ID) + 1
                else:
                    length = len(tokens_row)
                expr = tokenizer.decode(tokens_row[:length], nums_row[:length])
                if expr is None:
                    continue
                n = sample.X.shape[1]
                if n < tokenizer.max_inputs:
                    X_eval = np.column_stack([
                        sample.X,
                        np.zeros((sample.X.shape[0], tokenizer.max_inputs - n)),
                    ])
                else:
                    X_eval = sample.X
                y_pred = expr.evaluate(X_eval)
                if not np.isfinite(y_pred).all():
                    continue
            torch.cpu.synchronize()
            val_timings["decode_eval"].append(time.perf_counter() - t0)
            model.train()

    return {
        "timings": dict(timings),
        "val_timings": dict(val_timings),
        "step_times": step_times,
        "n_params": n_params,
        "vocab_size": len(tokenizer.vocab),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
TRAIN_COMPONENTS = [
    ("data_gen",   "Data generation"),
    ("collate",    "collate_fn"),
    ("forward",    "Forward + loss"),
    ("backward",   "Backward"),
    ("optim_step", "Optim + sched"),
    ("tracker",    "Tracker logging"),
]

VAL_COMPONENTS = [
    ("gen_eval",    "model.generate()"),
    ("decode_eval", "decode + evaluate"),
]


def _stats_ms(vals):
    """Return (mean_ms, median_ms, min_ms, max_ms, n)."""
    if not vals:
        return (0.0, 0.0, 0.0, 0.0, 0)
    ms = [v * 1000 for v in vals]
    return (
        statistics.mean(ms),
        statistics.median(ms),
        min(ms),
        max(ms),
        len(ms),
    )


def summarize(results, label):
    """Print a summary table of average ms per component. Returns means dict."""
    timings = results["timings"]
    val_timings = results["val_timings"]
    n_params = results["n_params"]

    print(f"\n{'=' * 76}")
    print(f"  {label}")
    print(f"  Params: {n_params:,} | Vocab: {results['vocab_size']}")
    print(f"{'=' * 76}")

    means = {}
    for key, name in TRAIN_COMPONENTS:
        vals = timings.get(key, [])
        mean_ms, *_ = _stats_ms(vals)
        means[key] = mean_ms

    step_mean_ms = statistics.mean(results["step_times"]) * 1000 if results["step_times"] else 0.0
    instr_sum_ms = sum(means.values())

    # Per-step training breakdown
    print(f"\n  Per-step training breakdown (post-warmup, n={len(results['step_times'])} steps):")
    print(f"  {'Component':<20} {'mean ms':>9} {'median':>9} {'min':>8} {'max':>8} {'% step':>8} {'n':>5}")
    print(f"  {'-' * 20} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 5}")
    for key, name in TRAIN_COMPONENTS:
        vals = timings.get(key, [])
        mean_ms, med_ms, min_ms, max_ms, n = _stats_ms(vals)
        pct = (mean_ms / step_mean_ms * 100) if step_mean_ms > 0 else 0.0
        print(f"  {name:<20} {mean_ms:>9.3f} {med_ms:>9.3f} {min_ms:>8.3f} {max_ms:>8.3f} {pct:>7.1f}% {n:>5}")
    print(f"  {'-' * 20} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 5}")
    print(f"  {'Instrumented sum':<20} {instr_sum_ms:>9.3f} {'':>9} {'':>8} {'':>8} "
          f"{instr_sum_ms / step_mean_ms * 100 if step_mean_ms > 0 else 0:>7.1f}%")
    print(f"  {'Measured step total':<20} {step_mean_ms:>9.3f} {'':>9} {'':>8} {'':>8} {'100.0%':>8}")

    # Validation breakdown (per val step)
    print(f"\n  Validation breakdown (per val-step, n={len(val_timings.get('gen_eval', []))}):")
    print(f"  {'Component':<20} {'mean ms':>9} {'median':>9} {'min':>8} {'max':>8} {'n':>5}")
    print(f"  {'-' * 20} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 5}")
    for key, name in VAL_COMPONENTS:
        vals = val_timings.get(key, [])
        mean_ms, med_ms, min_ms, max_ms, n = _stats_ms(vals)
        if n:
            print(f"  {name:<20} {mean_ms:>9.3f} {med_ms:>9.3f} {min_ms:>8.3f} {max_ms:>8.3f} {n:>5}")
        else:
            print(f"  {name:<20} {'N/A':>9} {'':>9} {'':>8} {'':>8} {0:>5}")

    print()
    means["__step_total__"] = step_mean_ms
    return means


def main():
    config = make_config()
    pool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pools", "raw_pool.pkl")

    print("CONFIG:")
    for k in ["d_model", "n_heads", "d_ff", "n_enc_layers", "n_dec_layers",
              "n_steps", "batch_size", "max_inputs", "max_ops", "val_batch_size"]:
        print(f"  {k} = {config[k]}")

    all_means = {}

    # ---- Run 1: Live generation ----
    print("\n" + "#" * 76)
    print("# RUN 1: LIVE GENERATION (pipeline.iter)")
    print("#" * 76)
    t0 = time.perf_counter()
    results_live = profile_run(dict(config), pool_file=None)
    elapsed_live = time.perf_counter() - t0
    print(f"  Wall time: {elapsed_live:.1f}s")
    all_means["live"] = summarize(results_live, "LIVE GENERATION (on-the-fly)")

    # ---- Run 2: Pool-based ----
    if os.path.exists(pool_path):
        print("\n" + "#" * 76)
        print("# RUN 2: POOL-BASED (pools/raw_pool.pkl)")
        print("#" * 76)
        t0 = time.perf_counter()
        results_pool = profile_run(dict(config), pool_file=pool_path)
        elapsed_pool = time.perf_counter() - t0
        print(f"  Wall time: {elapsed_pool:.1f}s")
        all_means["pool"] = summarize(results_pool, "POOL-BASED (raw_pool.pkl)")
    else:
        print(f"\n  [SKIP] Pool file not found: {pool_path}")

    # ---- Side-by-side comparison ----
    if "live" in all_means and "pool" in all_means:
        print("=" * 76)
        print("  COMPARISON: Live vs Pool (mean ms per training step)")
        print("=" * 76)
        live = all_means["live"]
        pool = all_means["pool"]
        print(f"\n  {'Component':<16} {'Live ms':>10} {'Pool ms':>10} {'Ratio':>9}")
        print(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 9}")
        for key, name in TRAIN_COMPONENTS:
            lm = live[key]
            pm = pool[key]
            ratio = f"{pm / lm:.2f}x" if lm > 0.001 else "N/A"
            print(f"  {name:<16} {lm:>10.3f} {pm:>10.3f} {ratio:>9}")
        print(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 9}")
        lstep, pstep = live["__step_total__"], pool["__step_total__"]
        ratio = f"{pstep / lstep:.2f}x" if lstep > 0.001 else "N/A"
        print(f"  {'Step total':<16} {lstep:>10.3f} {pstep:>10.3f} {ratio:>9}")
        print(f"\n  Data-gen speedup (live/pool): "
              f"{live['data_gen'] / max(pool['data_gen'], 1e-9):.1f}x faster with pool")

    print("\nDone.")


if __name__ == "__main__":
    main()
