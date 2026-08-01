# symlab Timing & Profiling Report

**Generated:** 2026-07-31 · **Host:** 4-core CPU (i3-9100T), torch 2.13, sympy 1.14
**Scope:** relative timing of the canonical training loop, not absolute throughput.
Reproducible: `experiments/timing/{step_and_model,pipeline_throughput,evaluate_bench}.py`

> **Hardware caveat (read first).** All timings are CPU. The training math is
> ~50–100× faster on a single GPU, while data generation (Python/sympy) is
> hardware-independent. **This flips the bottleneck depending on where you run:**
> on this CPU box *training* dominates; on the HPC GPU *canonicalization*
> dominates. Both conclusions are derived below from the same measurements.

---

## 1. Per-step breakdown (standard config)

Config = `experiments/canon/config.py`: `d_model=512, n_heads=8, d_ff=2048,
n_enc=3, n_dec=6, batch=64`, default opset (vocab=12), 100 sample points/expression
(encoder sequence length = 101 incl. stats token; decoder ≈8 tokens).

| Component | ms / step | % of step |
|---|---:|---:|
| Data generation (raw, no canon) | 10.9 | 0.1% |
| `collate_fn` | 10.3 | 0.1% |
| **Forward + loss** | **2,427.7** | **25.3%** |
| **Backward** | **6,195.9** | **64.5%** |
| Optimizer step (clip + AdamW + zero_grad) | 963.1 | 10.0% |
| **Total** | **9,607.9** | 100% |

**Bottleneck (CPU): the backward pass** — 64.5% of every step. Forward is 25%,
optimizer 10%. The raw data pipeline (21 ms combined) is **negligible: 0.2%**.
Backward runs ~2.5× forward, the expected range for attention-heavy transformers.

### What changes if canonicalization is live (not pooled)
At measured **75 samples/s** (§3), generating one canonicalized batch of 64 costs
`64 / 75 = 853 ms`. Re-run the table with that substitution:

| Component | CPU (this box) | GPU (HPC, est. ÷100 on train) |
|---|---:|---:|
| Canonicalized data gen | 853 ms (8%) | 853 ms (**90%**) |
| Training (fwd+bwd+opt) | 9,587 ms (92%) | ~96 ms (10%) |
| **Bottleneck** | training | **data generation** |

This is exactly the regime `experiments/overnight-report.md` hit: *"runtime
sympy.simplify per-sample is too slow (data-bound; stuck generating). Switched to
a pre-generated cached dataset."* The fix (pools) is validated in §3.

---

## 2. Model architecture sizing

| Component | standard (d=512) | small (d=128) | ratio |
|---|---:|---:|---:|
| encoder.blocks | 9,457,152 | 396,544 | 23.8× |
| decoder.blocks | 25,224,192 | 1,058,304 | 23.8× |
| decoder.numeric_head | 1,052,673 | 66,561 | 15.8× |
| embeddings + proj + norms | 29,660 | 8,100 | 3.7× |
| **Total params** | **35,770,381** | **1,530,509** | **23.4×** |
| param memory (fp32) | 143.1 MB | 6.1 MB | 23.4× |

**Forward-pass FLOPs** (batch=64, 100 pts, 8 decoder tokens; MAC×2):

| | standard | small | ratio |
|---|---:|---:|---:|
| encoder (101-token self-attn × 3 layers) | 126.1 G | 5.8 G | 21.9× |
| decoder (8-token self+cross × 6 layers) | 27.5 G | 1.3 G | 21.9× |
| **Total forward** | **153.6 GFLOP** | **7.0 GFLOP** | **21.9×** |

**Measured per-step speedup (fwd+bwd+opt, CPU):**
`5,772.7 ms → 412.4 ms` = **14.0× faster** for the small config.

The measured 14× is below the 21.9× FLOP ratio because a chunk of each step does
**not** scale with `d_model²`: the AdamW step over params (≈linear in params but
memory-bandwidth-bound on CPU), LayerNorms, embeddings, the `O(L·d)` attention
score terms, and fixed Python/kernel overhead. These fixed costs are a larger
fraction of the small model's step. **Net: the small config buys ~14× more
steps/sec, not 22×.**

Two structural notes:
- **Params concentrate in the decoder** (25.2M of 35.8M = 70%) because `n_dec=6`
  vs `n_enc=3` and each decoder block carries *two* attention sublayers.
- **FLOPs concentrate in the encoder** (126 G of 154 G = 82%) because it attends
  over 101 tokens (the 100 sample points), while the decoder attends over only ~8.
  *The 100-point encoder sequence is the single biggest compute sink.*

---

## 3. Data pipeline throughput (1000 samples, default opset, max_ops=3)

| Regime | Throughput | vs raw |
|---|---:|---:|
| Raw generation (no canon) | **6,278 samples/s** | 1× |
| + `sympy.simplify` | **75 samples/s** | **84× slower** |
| + simplify + `max_const=100` filter | 74 samples/s | 85× slower |
| Pool **load** (pkl → list) | 44,000–92,000 s/s | — |
| Pool **sample** (slice N, w/ replacement) | 360k–4,650k s/s | — |

**Canonicalization is the entire cost of data prep.** `sympy.simplify` alone is
84× the rest of the pipeline combined (tree gen + constant fill + point eval +
validity filter). The `max_const` filter is itself free — it runs *after* simplify
on already-canonicalized constants, so it adds no measurable time; it only changes
the accept rate (here ~same, since few expressions breach |c|>100 at max_ops=3).

For this run, **0 of 1000 simplifies timed out** (2 s cap) and 0 errored at
max_ops=3 with the default opset — but `simplify` cost is still ~13 ms/expression
with high variance, which is what makes live canonicalization untenable at GPU
training speeds.

**Verdict:** live generation is viable for **raw** training (~6.3k/s is far above
any batch demand). Live generation is **not viable once canonicalization is on**
unless you accept being data-bound — hence the pool strategy. Pools are
essentially free: a 12,800-sample (28 MB) pool loads in ~0.2 s and samples at
millions/sec.

---

## 4. `Expression.evaluate` performance

`evaluate` is a vectorized NumPy bytecode interpreter: one call evaluates one
expression across *all* rows of `X`. Cost per call scales with (#points ×
#commands); wall time scales linearly with (#expressions).

| Workload | ms/expr | point-evals/s |
|---|---:|---:|
| 100 expr × 100 pts | 0.009 | 11.5 M |
| 500 expr × 100 pts | 0.009 | 11.3 M |
| 1000 expr × 100 pts | 0.009 | 11.4 M |
| single expr × 10,000 pts | — | 332 M |
| single expr × 100,000 pts | — | 320 M |

`evaluate` is **not a bottleneck anywhere** — ~9 µs per expression at 100 points,
and it *amortizes* as point count grows (up to 320 M point-evals/s on big arrays).
Validation/inspection loops that call `evaluate` per predicted expression are
trivially cheap relative to generation or training. The `RuntimeWarning: overflow
in exp` seen in runs is benign — `evaluate` intentionally returns `inf`/`nan` on
math edge cases, which `is_valid()` then filters during generation.

---

## 5. Bottleneck summary

| Scenario | Bottleneck | Evidence |
|---|---|---|
| CPU training, raw data | **Backward pass** (64.5%/step) | §1 |
| CPU training, live canon | Backward (92%), canon minor (8%) | §1+§3 |
| **GPU training, live canon** | **Canonicalization** (~90%) | §1 extrapolation, matches overnight-report |
| Any setup | `evaluate` / validation | never (§4) |

---

## 6. Recommendations

1. **Keep using pools whenever canonicalization is on.** Live `sympy.simplify` is
   84× the raw pipeline and becomes the wall-clock bottleneck the moment training
   moves to GPU (overnight-a already learned this). Pre-generate, serialize to
   `.pkl`, sample at runtime — pool load is <0.25 s for 12,800 samples.

2. **The raw path can stay live.** At 6,278 samples/s, raw generation is never the
   limit; only canonicalization forces pooling. Diagnostic runs (lambda=0, no
   canon) don't need pools at all.

3. **For faster iteration, shrink the model before shrinking anything else.** The
   small config (d=128, 2+4 layers) is **14× faster/step** and 23× smaller
   (1.5M vs 35.8M params) with a 22× FLOP reduction. It's the highest-leverage
   knob for CPU prototyping and ablation sweeps. Capabilities that need the full
   opset/depth can run the standard config on GPU.

4. **The encoder's 100-point sequence is the compute sink (82% of FLOPs).** If
   you want to cut training cost without losing capacity: reduce sample points
   per expression (e.g. 100→50) or subsample points in the encoder. This lowers
   encoder FLOPs roughly linearly and is orthogonal to model width.

5. **Backward dominates forward 2.5:1 on CPU.** Memory-bandwidth-bound ops (AdamW
   state, grad clipping) inflate the optimizer+backward slice; on GPU this ratio
   compresses toward the usual ~2:1. No action needed — just don't profile the
   optimizer in isolation on CPU and assume it carries over to GPU.

6. **Don't optimize `evaluate` or `collate_fn`.** Both are <0.2% of a step (raw)
   and `evaluate` is negligible even at 100k points. Any time spent here is
   wasted.
