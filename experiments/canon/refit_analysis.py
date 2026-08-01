#!/usr/bin/env python3
"""Constant-refinement analysis for a trained probe model.

Loads a saved checkpoint (results/{source_tag}.pt), regenerates expressions on
the held-out test split of the pool, and for each generated expression compares
the functional fit (R^2) BEFORE and AFTER fitting constants to the true (X, y)
via Expression.fit() (nonlinear least-squares). This isolates the
"right-structure / wrong-constants" gap: if fit() lifts R^2 substantially, the
structure head is correct and only the numeric (xVal) head is weak.

Run like any trial:
    jernerics run experiments/canon/refit_analysis.py experiments/canon/<cfg>/config.py
The config must set `source_tag` (the trained model's tag) plus the usual model /
pool fields so the architecture and test split match the training run exactly.
"""

import os
import sys
import json
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libs"))
from manifest.model import TransformerModel  # noqa: E402
from manifest.tokenizer import XValsTokenizer, PAD_ID, BOS_ID, EOS_ID  # noqa: E402
from manifest.data import collate_fn  # noqa: E402
from symbolic import OperatorSet  # noqa: E402
from symbolic.scoring import r2 as r2_score  # noqa: E402


def _pad_x(X, max_inputs):
    n = X.shape[1]
    if n < max_inputs:
        return np.column_stack([X, np.zeros((X.shape[0], max_inputs - n))])
    return X


def trial(config, tracker):
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    tag = config.get("tag", "refit")
    source_tag = config["source_tag"]

    opset = (OperatorSet.default()
             if config.get("opset", "default") == "default"
             else OperatorSet.comprehensive())
    tokenizer = XValsTokenizer(opset, max_inputs=config["max_inputs"])

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt = os.path.join("results", f"{source_tag}.pt")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint {ckpt} on {device}")

    # Same held-out split as the training trial.
    with open(config["pool_file"], "rb") as f:
        pool = pickle.load(f)
    n_val = config["val_batch_size"]
    n_test = config.get("n_test", 100)
    test_samples = pool[n_val:n_val + n_test]
    print(f"Test split: {len(test_samples)} samples from {config['pool_file']}")

    batch = collate_fn(test_samples, tokenizer)
    data = batch["data"].to(device)
    stats = batch["stats"].to(device)
    data_mask = batch["data_mask"].to(device)

    with torch.no_grad():
        gen_tokens, gen_nums = model.generate(data, data_mask=data_mask, stats=stats)

    rows = []
    raw_r2, fit_r2 = [], []
    n_decoded = 0
    for i, sample in enumerate(test_samples):
        toks = gen_tokens[i].tolist()
        nums = gen_nums[i].tolist()
        length = toks.index(EOS_ID) + 1 if EOS_ID in toks else len(toks)
        expr = tokenizer.decode(toks[:length], nums[:length])
        if expr is None:
            rows.append({"i": i, "decoded": None, "raw_r2": None, "fit_r2": None,
                         "n_const": 0})
            continue
        X_eval = _pad_x(sample.X, config["max_inputs"])
        try:
            y_raw = expr.evaluate(X_eval)
        except Exception:
            rows.append({"i": i, "decoded": str(expr), "raw_r2": None,
                         "fit_r2": None, "n_const": len(expr.constants)})
            continue
        if not np.isfinite(y_raw).all():
            rows.append({"i": i, "decoded": str(expr), "raw_r2": None,
                         "fit_r2": None, "n_const": len(expr.constants)})
            continue

        n_decoded += 1
        ss_tot = float(np.var(sample.y))
        raw_r2v = 1.0 - float(np.mean((sample.y - y_raw) ** 2)) / max(ss_tot, 1e-9)
        raw_r2.append(raw_r2v)

        fit_r2v = None
        try:
            fitted = expr.fit(X_eval, sample.y)
            y_fit = fitted.evaluate(X_eval)
            if np.isfinite(y_fit).all():
                fit_r2v = 1.0 - float(np.mean((sample.y - y_fit) ** 2)) / max(ss_tot, 1e-9)
                fit_r2.append(fit_r2v)
        except Exception:
            pass
        if fit_r2v is None:
            fit_r2.append(raw_r2v)
            fit_r2v = raw_r2v

        rows.append({"i": i, "decoded": str(expr), "raw_r2": raw_r2v,
                     "fit_r2": fit_r2v, "n_const": len(expr.constants)})

    def frac(lst, thr):
        return sum(1 for v in lst if v > thr)

    summary = {
        "source_tag": source_tag,
        "n_test": len(test_samples),
        "n_decoded": n_decoded,
        "raw_mean_r2": float(np.mean(raw_r2)) if raw_r2 else None,
        "raw_median_r2": float(np.median(raw_r2)) if raw_r2 else None,
        "raw_high_r2_gt09": frac(raw_r2, 0.9),
        "raw_func_equiv_gt099": frac(raw_r2, 0.99),
        "fit_mean_r2": float(np.mean(fit_r2)) if fit_r2 else None,
        "fit_median_r2": float(np.median(fit_r2)) if fit_r2 else None,
        "fit_high_r2_gt09": frac(fit_r2, 0.9),
        "fit_func_equiv_gt099": frac(fit_r2, 0.99),
        "delta_func_equiv": frac(fit_r2, 0.99) - frac(raw_r2, 0.99),
        "mean_n_const": float(np.mean([r["n_const"] for r in rows if r["decoded"]]))
        if any(r["decoded"] for r in rows) else 0.0,
    }

    print("\n" + "=" * 70)
    print(f"REFIT ANALYSIS  source={source_tag}  n_decoded={n_decoded}/{len(test_samples)}")
    print("=" * 70)
    print(f"  RAW : median_r2={summary['raw_median_r2']:.3f}  "
          f"high(>.9)={summary['raw_high_r2_gt09']}  "
          f"func_equiv(>.99)={summary['raw_func_equiv_gt099']}")
    print(f"  FIT : median_r2={summary['fit_median_r2']:.3f}  "
          f"high(>.9)={summary['fit_high_r2_gt09']}  "
          f"func_equiv(>.99)={summary['fit_func_equiv_gt099']}")
    print(f"  func_equiv lift from fit(): "
          f"{summary['raw_func_equiv_gt099']} -> {summary['fit_func_equiv_gt099']} "
          f"(+{summary['delta_func_equiv']})")
    print("=" * 70)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump({"tag": tag, "summary": summary, "rows": rows}, f, indent=2,
                  default=str)
    print(f"RESULTS WRITTEN: {os.path.join(out_dir, tag + '.json')}")

    return {"final_loss": 0.0, **summary}


if __name__ == "__main__":
    # Manual local run: python refit_analysis.py <tag> <pool_file>
    if len(sys.argv) == 3:
        cfg = {
            "tag": sys.argv[1] + "_refit", "source_tag": sys.argv[1],
            "pool_file": sys.argv[2], "opset": "default", "max_inputs": 2,
            "max_ops": 3, "max_seq_len": 32, "d_model": 512, "n_heads": 8,
            "d_ff": 2048, "n_enc_layers": 3, "n_dec_layers": 6, "dropout": 0.1,
            "seed": 42, "val_batch_size": 32, "n_test": 100,
        }

        class _T:
            def log_value(self, *a, **k): pass
            def log_param(self, *a, **k): pass
            def log_artifact(self, *a, **k): pass
        trial(cfg, _T())
