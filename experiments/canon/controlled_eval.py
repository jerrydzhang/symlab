#!/usr/bin/env python3
"""Controlled functional-recovery comparison on a SHARED test set.

The canon vs raw training runs hold out *different* test splits (canonicalized
vs raw expressions), so their funcEq numbers are not directly comparable. This
script removes that confound: it loads several trained checkpoints and
evaluates them all on ONE freshly-generated test set (raw distribution), with
and without Expression.fit() constant refinement.

Answers: on a level playing field, does canonicalization improve functional
recovery (R^2), or only token-level metrics?

    jernerics run experiments/canon/controlled_eval.py experiments/canon/<cfg>/config.py
config sets `source_tags` (list) + `test_pool_file` (the shared pool).
"""

import os
import sys
import json
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libs"))
from manifest.model import TransformerModel  # noqa: E402
from manifest.tokenizer import XValsTokenizer, EOS_ID  # noqa: E402
from manifest.data import collate_fn  # noqa: E402
from symbolic import OperatorSet  # noqa: E402


def _pad_x(X, max_inputs):
    n = X.shape[1]
    if n < max_inputs:
        return np.column_stack([X, np.zeros((X.shape[0], max_inputs - n))])
    return X


def _build_model(config, tokenizer):
    return TransformerModel(
        input_dim=config["max_inputs"] + 1,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        d_ff=config["d_ff"], n_enc_layers=config["n_enc_layers"],
        n_dec_layers=config["n_dec_layers"],
        dropout=config.get("dropout", 0.1),
    )


def _eval_model(model, tokenizer, test_samples, max_inputs, device):
    model.eval()
    batch = collate_fn(test_samples, tokenizer)
    data = batch["data"].to(device)
    stats = batch["stats"].to(device)
    data_mask = batch["data_mask"].to(device)
    with torch.no_grad():
        gen_tokens, gen_nums = model.generate(data, data_mask=data_mask, stats=stats)
    raw_r2, fit_r2 = [], []
    for i, sample in enumerate(test_samples):
        toks = gen_tokens[i].tolist()
        nums = gen_nums[i].tolist()
        length = toks.index(EOS_ID) + 1 if EOS_ID in toks else len(toks)
        expr = tokenizer.decode(toks[:length], nums[:length])
        if expr is None:
            continue
        X_eval = _pad_x(sample.X, max_inputs)
        try:
            y_raw = expr.evaluate(X_eval)
        except Exception:
            continue
        if not np.isfinite(y_raw).all():
            continue
        ss_tot = max(float(np.var(sample.y)), 1e-9)
        raw_v = 1.0 - float(np.mean((sample.y - y_raw) ** 2)) / ss_tot
        raw_r2.append(raw_v)
        try:
            y_fit = expr.fit(X_eval, sample.y).evaluate(X_eval)
            fit_v = (1.0 - float(np.mean((sample.y - y_fit) ** 2)) / ss_tot
                     if np.isfinite(y_fit).all() else raw_v)
        except Exception:
            fit_v = raw_v
        fit_r2.append(fit_v)
    return raw_r2, fit_r2


def trial(config, tracker):
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opset = (OperatorSet.default()
             if config.get("opset", "default") == "default"
             else OperatorSet.comprehensive())

    with open(config["test_pool_file"], "rb") as f:
        test_samples = pickle.load(f)[: config["n_test"]]
    print(f"Shared test set: {len(test_samples)} samples from "
          f"{config['test_pool_file']}")

    results = {}
    for src in config["source_tags"]:
        tokenizer = XValsTokenizer(opset, max_inputs=config["max_inputs"])
        model = _build_model(config, tokenizer).to(device)
        ckpt = os.path.join("results", f"{src}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        raw_r2, fit_r2 = _eval_model(model, tokenizer, test_samples,
                                     config["max_inputs"], device)
        n = len(raw_r2)
        results[src] = {
            "n": n,
            "raw_funcEq": sum(1 for v in raw_r2 if v > 0.99),
            "raw_hiR2": sum(1 for v in raw_r2 if v > 0.9),
            "raw_median": float(np.median(raw_r2)) if raw_r2 else None,
            "fit_funcEq": sum(1 for v in fit_r2 if v > 0.99),
            "fit_hiR2": sum(1 for v in fit_r2 if v > 0.9),
            "fit_median": float(np.median(fit_r2)) if fit_r2 else None,
        }
        print(f"  {src:<22} n={n}  RAW funcEq={results[src]['raw_funcEq']} "
              f"hiR2={results[src]['raw_hiR2']} med={results[src]['raw_median']:.3f}"
              f"  |  FIT funcEq={results[src]['fit_funcEq']} "
              f"hiR2={results[src]['fit_hiR2']} med={results[src]['fit_median']:.3f}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{config['tag']}.json"), "w") as f:
        json.dump({"tag": config["tag"], "test_pool": config["test_pool_file"],
                   "n_test": len(test_samples), "results": results}, f, indent=2,
                  default=str)
    print(f"RESULTS WRITTEN: results/{config['tag']}.json")
    return {"final_loss": 0.0}
