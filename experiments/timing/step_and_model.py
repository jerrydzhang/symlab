#!/usr/bin/env python3
"""Task 1 (per-step timing breakdown) + Task 2 (model param/FLOP sizing).

Standard config = canon/config.py (d_model=512).  Small config = diagnostic
d_model=128.  CPU timing (4 cores) — relative breakdown is what matters.
"""
import os
import sys
import time
import statistics

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT, "models", "manifest", "src"))
sys.path.insert(0, os.path.join(PROJECT, "libs", "symbolic", "src"))

from symbolic import OperatorSet
from symbolic.generation import (
    Pipeline, RandomBinaryTree, MantissaExponentConstants, UniformSamplePoints, is_valid,
)
from manifest.tokenizer import XValsTokenizer
from manifest.data import collate_fn
from manifest.model import TransformerModel
from manifest.loss import compute_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


def build_pipeline(opset, max_ops, max_inputs, rng):
    return (
        Pipeline(RandomBinaryTree(opset, max_ops=max_ops, num_vars=(1, max_inputs), rng=rng))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(rng=rng))
        .filter(is_valid())
    )


def count_params(model):
    """Break parameter count down by submodule."""
    totals = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top == "encoder":
            # split into data_proj/stats_proj/blocks/final_norm
            if "blocks" in name:
                key = "encoder.blocks"
            else:
                key = "encoder." + name.split(".")[1]
        elif top == "decoder":
            if "blocks" in name:
                key = "decoder.blocks"
            else:
                key = "decoder." + name.split(".")[1]
        else:
            key = top
        totals[key] = totals.get(key, 0) + param.numel()
    return totals


def flops_per_forward(cfg, n_points, vocab, seq_len):
    """Estimate forward-pass FLOPs (multiply-accumulate * 2) for one batch.

    cfg: dict with d_model, n_heads, d_ff, n_enc_layers, n_dec_layers,
         batch_size, input_dim.  n_points = data points per sample (encoder
         seq = n_points + 1 stats token).  seq_len = decoder token length.
    """
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]
    B = cfg["batch_size"]
    L_enc = n_points + 1          # stats token prepended
    L_dec = seq_len
    n_enc = cfg["n_enc_layers"]
    n_dec = cfg["n_dec_layers"]
    vid = cfg["input_dim"]

    def linear(a, b):
        return 2 * a * b

    # ---- encoder ----
    enc_projs = linear(vid, d) + linear(2 * vid, d)   # data_proj + stats_proj
    # one encoder block: self-attn (packed QKV 6d^2 + out 2d^2 + scores 4*L*d) + FFN 4*d*d_ff
    attn_enc = 6 * d * d + 2 * d * d + 4 * L_enc * d
    ffn = 4 * d * d_ff
    enc_block = attn_enc + ffn
    enc_total = (enc_projs + n_enc * enc_block) * L_enc * B
    enc_final_norm = 2 * d * L_enc * B
    enc = enc_total + enc_final_norm

    # ---- decoder ----
    token_emb = d * vocab   # embedding lookup (index, not MAC) -- size proxy
    pos_emb = d * seq_len
    # one decoder block: self-attn (causal) + cross-attn + FFN
    self_attn = 6 * d * d + 2 * d * d + 4 * L_dec * d
    cross_attn = 6 * d * d + 2 * d * d + 4 * L_enc * d   # keys from encoder
    dec_block = self_attn + cross_attn + ffn
    dec_total = n_dec * dec_block * L_dec * B
    logit_head = linear(d, vocab) * L_dec * B
    num_head = linear(d, d_ff) + linear(d_ff, 1)
    num_head *= L_dec * B
    dec = dec_total + logit_head + num_head

    emb = token_emb + pos_emb
    return {
        "encoder": enc,
        "decoder_compute": dec,
        "embeddings": emb,
        "total_forward": enc + dec + emb,
    }


def time_step(model, batch, optimizer, n_warmup=3, n_timed=10, separate=True):
    data, stats = batch["data"], batch["stats"]
    tokens, num_values = batch["tokens"], batch["num_values"]
    data_mask, token_mask = batch["data_mask"], batch["token_mask"]
    input_tokens, input_nums = tokens[:, :-1], num_values[:, :-1]
    target_tokens, target_nums = tokens[:, 1:], num_values[:, 1:]
    target_mask = token_mask[:, :-1]

    fwd, bwd, opt = [], [], []
    combined = []

    for _ in range(n_warmup):
        t0 = time.perf_counter()
        logits, num_preds = model(data, input_tokens, input_nums,
                                  data_mask=data_mask, token_mask=target_mask, stats=stats)
        loss = compute_loss(logits, num_preds, target_tokens, target_nums, lambda_=0.0)
        torch.cpu.synchronize()
        f = time.perf_counter() - t0
        t0 = time.perf_counter()
        loss.backward()
        torch.cpu.synchronize()
        b = time.perf_counter() - t0
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cpu.synchronize()

    for _ in range(n_timed):
        t0 = time.perf_counter()
        logits, num_preds = model(data, input_tokens, input_nums,
                                  data_mask=data_mask, token_mask=target_mask, stats=stats)
        loss = compute_loss(logits, num_preds, target_tokens, target_nums, lambda_=0.0)
        torch.cpu.synchronize()
        f = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        torch.cpu.synchronize()
        b = time.perf_counter() - t0

        t0 = time.perf_counter()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cpu.synchronize()
        o = time.perf_counter() - t0

        fwd.append(f); bwd.append(b); opt.append(o)
        combined.append(f + b + o)

    return fwd, bwd, opt, combined


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    opset = OperatorSet.default()
    tokenizer = XValsTokenizer(opset, max_inputs=2)
    rng = np.random.default_rng(42)
    pipeline = build_pipeline(opset, max_ops=3, max_inputs=2, rng=rng)
    vocab = len(tokenizer.vocab)

    # ---- Task 1: per-step breakdown at STANDARD config ----
    print("=" * 68)
    print("TASK 1: PER-STEP TIMING BREAKDOWN (standard config d_model=512)")
    print("=" * 68)

    # data gen
    t0 = time.perf_counter()
    samples = list(pipeline.iter(64))
    t_gen = time.perf_counter() - t0

    # collate
    t0 = time.perf_counter()
    batch = collate_fn(samples, tokenizer)
    t_collate = time.perf_counter() - t0

    n_points = batch["data"].shape[1]
    seq_len = batch["tokens"].shape[1]
    print(f"  vocab={vocab}  data_shape={tuple(batch['data'].shape)}  "
          f"(enc seq={n_points+1})  tok_seq={seq_len}")
    print(f"  Data generation : {t_gen*1000:8.1f} ms  (64 samples)")
    print(f"  collate_fn      : {t_collate*1000:8.1f} ms")

    cfg = dict(d_model=512, n_heads=8, d_ff=2048, n_enc_layers=3, n_dec_layers=6,
               batch_size=64, input_dim=3)
    model = TransformerModel(
        input_dim=3, vocab_size=vocab, max_seq_len=32,
        d_model=512, n_heads=8, d_ff=2048, n_enc_layers=3, n_dec_layers=6, dropout=0.1,
    )
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    fwd, bwd, opt, comb = time_step(model, batch, optimizer, n_warmup=3, n_timed=10)
    mf, mb, mo, mc = (statistics.mean(x) for x in (fwd, bwd, opt, comb))
    total = t_gen + t_collate + mc
    print(f"\n  Training loop (10 steps, post-warmup):")
    print(f"    forward+loss  : {mf*1000:8.1f} ms   ({mf/total*100:5.1f}% of step)")
    print(f"    backward      : {mb*1000:8.1f} ms   ({mb/total*100:5.1f}% of step)")
    print(f"    optim step    : {mo*1000:8.1f} ms   ({mo/total*100:5.1f}% of step)")
    print(f"  -----------------------------------------------")
    print(f"    data gen      : {t_gen*1000:8.1f} ms   ({t_gen/total*100:5.1f}% of step)")
    print(f"    collate       : {t_collate*1000:8.1f} ms   ({t_collate/total*100:5.1f}% of step)")
    print(f"    train (f+b+o) : {mc*1000:8.1f} ms   ({mc/total*100:5.1f}% of step)")
    print(f"  ===============================================")
    print(f"    TOTAL/step    : {total*1000:8.1f} ms")
    print(f"    bottleneck    : {'TRAINING' if mc > t_gen + t_collate else 'DATA PIPELINE'}")

    # store for later
    results = dict(t_gen=t_gen, t_collate=t_collate, fwd=mf, bwd=mb, opt=mo,
                   total=total, n_points=n_points, seq_len=seq_len, vocab=vocab)

    # ---- Task 2: model sizing comparison ----
    print("\n" + "=" * 68)
    print("TASK 2: MODEL ARCHITECTURE SIZING")
    print("=" * 68)

    configs = {
        "standard (d=512)": dict(d_model=512, n_heads=8, d_ff=2048, n_enc_layers=3,
                                 n_dec_layers=6, batch_size=64, input_dim=3),
        "small (d=128)":    dict(d_model=128, n_heads=4, d_ff=512, n_enc_layers=2,
                                 n_dec_layers=4, batch_size=64, input_dim=3),
    }
    flops_table = {}
    step_table = {}
    for label, c in configs.items():
        m = TransformerModel(
            input_dim=3, vocab_size=vocab, max_seq_len=32,
            d_model=c["d_model"], n_heads=c["n_heads"], d_ff=c["d_ff"],
            n_enc_layers=c["n_enc_layers"], n_dec_layers=c["n_dec_layers"], dropout=0.1,
        )
        n_params = sum(p.numel() for p in m.parameters())
        breakdown = count_params(m)
        fl = flops_per_forward(c, n_points, vocab, seq_len - 1)  # decoder sees seq_len-1
        flops_table[label] = (n_params, breakdown, fl)

        # time it
        opt2 = AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        f2, b2, o2, c2 = time_step(m, batch, opt2, n_warmup=2, n_timed=8)
        step_table[label] = statistics.mean(c2)
        del m

    print(f"\n  {'Component':<26}{'standard':>14}{'small':>14}{'ratio':>10}")
    print(f"  {'-'*26}{'-'*14}{'-'*14}{'-'*10}")
    s_n, s_b, s_f = flops_table["standard (d=512)"]
    m_n, m_b, m_f = flops_table["small (d=128)"]

    all_keys = sorted(set(s_b) | set(m_b))
    for k in all_keys:
        sv, mv = s_b.get(k, 0), m_b.get(k, 0)
        print(f"  {k:<26}{sv:>14,}{mv:>14,}{sv/max(mv,1):>9.1f}x")
    print(f"  {'-'*26}{'-'*14}{'-'*14}{'-'*10}")
    print(f"  {'TOTAL params':<26}{s_n:>14,}{m_n:>14,}{s_n/m_n:>9.1f}x")
    print(f"  {'param memory (MB, fp32)':<26}{s_n*4/1e6:>14.2f}{m_n*4/1e6:>14.2f}{s_n/m_n:>9.1f}x")

    print(f"\n  Forward-pass FLOP estimate (batch=64, {n_points} pts, {seq_len-1} tok):")
    print(f"  {'Component':<26}{'standard':>16}{'small':>16}{'ratio':>10}")
    print(f"  {'-'*26}{'-'*16}{'-'*16}{'-'*10}")
    for k in ["encoder", "decoder_compute", "embeddings"]:
        sv, mv = s_f[k], m_f[k]
        print(f"  {k:<26}{sv:>16,.0f}{mv:>16,.0f}{sv/max(mv,1):>9.1f}x")
    print(f"  {'-'*26}{'-'*16}{'-'*16}{'-'*10}")
    print(f"  {'TOTAL forward FLOPs':<26}{s_f['total_forward']:>16,.0f}"
          f"{m_f['total_forward']:>16,.0f}{s_f['total_forward']/m_f['total_forward']:>9.1f}x")
    print(f"  {' (GFLOPs)':<26}{s_f['total_forward']/1e9:>16.2f}"
          f"{m_f['total_forward']/1e9:>16.2f}")

    print(f"\n  Measured per-step training time (forward+backward+optim, CPU):")
    for label in configs:
        print(f"    {label:<22}: {step_table[label]*1000:8.1f} ms")
    std_step = step_table["standard (d=512)"]
    sm_step = step_table["small (d=128)"]
    print(f"    speedup small/standard: {std_step/sm_step:.2f}x")
    print(f"    FLOP ratio (standard/small): {s_f['total_forward']/m_f['total_forward']:.1f}x")


if __name__ == "__main__":
    main()
