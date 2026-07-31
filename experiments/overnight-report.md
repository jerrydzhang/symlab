# Overnight Diagnostic Report — Symbolic Regression Plateau

**Date:** 2026-07-30 → 2026-07-31
**Question:** Why does `TransformerModel` plateau at CE≈0.71, token_acc≈74%, valid_rate≈41%?

## TL;DR (headline findings)

1. **The reported `valid_rate ≈ 41%` is a measurement artifact, not a model failure.** The eval path builds decoded expressions with `num_inputs = max_inputs` but evaluates them on `X` with fewer columns (1-variable samples) → `IndexError` → counted invalid. With the fix (pad `X`), **structural validity is ~100% and valid_rate is 1.00.** The model was never failing to produce valid trees.
2. **The CE plateau is an irreducible label-ambiguity floor, not an architecture or capacity limit.** CE scales with distribution complexity (1-op/1-var → CE 0.21; 2-var/3-op → CE 0.71) and 4× more training (20k steps) does not break it. **Canonicalizing the training data cuts the CE floor 4× (0.285 → 0.066) and triples exact-match (14% → 45%).** The ambiguity comes from non-canonical generation: `add(x0,c)` and `add(c,x0)` produce identical `(X,y)` but different token streams.
3. **`lambda_=1.0` hurts structure learning** (CE 1.27 vs 0.71). The MSE head gradient (~15–25) dominates the loss and retards token prediction. The MSE head is decoupled *and* harmful at high λ.
4. **The model is healthier than metrics suggested:** on the plateau config it recovers **35% of held-out expressions at R²>0.9** (bimodal: excellent or wildly wrong).

---

## Infrastructure issues encountered (flagged for Jerry)

- **Tracking server unreachable from compute nodes.** `atlas.taile454b.ts.net:443` is reachable from the login node / locally, but compute-node jobs log `Failed to send event ... Dropping remaining events` after 300s. No metrics reach the server from HPC. Worked around by (a) line-buffering stdout and reading SLURM `.out` over SSH, and (b) dumping `results/<tag>.json` + checkpoints to the project dir (SSH-readable). **The `jernerics logs` tool looks in the wrong dir** (`.../jernerics/logs/` instead of `.../jernerics/symlab/logs/`); read logs directly over SSH instead.
- **`gtx18` has a dead GPU** (`Unable to determine the device handle for GPU2`); jobs landing there fall back to CPU (CUDA init fails). Excluded `gtx18` from all submissions via `--exclude`.
- **QOS `qiy18011a100` caps concurrent GPUs at 2** (group limit `QOSGrpGRES`). Only ~2 GPU jobs run at once.
- **`uv.lock` acquired spurious changes** (96 lines) from `uv run`; left unstaged.

---

## Decision log

### SEED 1 — inspect generated expressions
**HYPOTHESIS:** Inspecting 100 held-out generations reveals *where/why* sequences break (structure vs constants vs decoding).
**CONFIG:** diagnostic (`default` 6-op, `max_ops=3`, 2 vars), 5000 steps. Ran two variants: `inspect` (λ=1.0, per spec) and `inspect_l0` (λ=0.0, the true plateau config).
**RESULT:**
- With λ=0 (plateau): CE 0.711, acc 0.745. **100% of generations are structurally complete trees.** Corrected functional recovery: **35% R²>0.9, 42% R²>0.5** (bimodal).
- Generated op-count skews simple: 46% 1-op / 47% 2-op / 7% 3-op (GT ≈⅓ each).
- Recovery varies sharply by operator: **sub 64%, exp 40%, add 36%, mul 32%, sin 26%, div 19%.**
- With λ=1: CE **1.27**, acc 0.59, collapse to 1-op expressions (78%). λ=1 retards structure learning.
**DECISION:** The "valid_rate 41%" is not structural failure — investigate the eval path; it's an artifact. CE plateau is ambiguity-driven; test canonicalization and complexity scaling.

### SEED 2 — scale to 50k steps
**HYPOTHESIS:** More training breaks CE past 0.71 (data-scale limited).
**CONFIG:** diagnostic, `n_steps=50000`.
**RESULT:** Cancelled at step 20500 (4× the original) to free a GPU — **CE was still flat at 0.68–0.85 around 0.71, valid_rate stuck at 0.41**, identical to the 5k run.
**DECISION:** The plateau is **not** scale-limited. Stopped early; the answer (no break) was conclusive.

### EXPERIMENT: easy (lowest-ambiguity baseline)
**HYPOTHESIS:** On the simplest distribution (1 op, 1 var) the model performs much better → CE scales with ambiguity/complexity.
**RATIONALE:** Isolate whether the plateau is intrinsic difficulty vs architecture.
**CONFIG:** `max_ops=1`, `num_vars=(1,1)`, 5000 steps, λ=0.
**RESULT:** **CE 0.207, acc 0.891, 50% exact token match, 36% R²>0.9.**
**DECISION:** CE scales hard with distribution complexity (0.21 → 0.71). The plateau tracks problem ambiguity, not a broken model. Strongly implicates label ambiguity.

### EXPERIMENT: canon (canonicalization) — DIRECT TEST
**HYPOTHESIS:** Canonicalizing expressions (sympy) removes commutative/equivalent-form ambiguity → lower CE floor + higher exact-match.
**RATIONALE:** easy vs plateau showed CE scales with ambiguity; canonicalization removes the structural ambiguity (`add(x0,c)≡add(c,x0)`).
**CONFIG:** diagnostic + `canonicalize=True`.
**RESULT (attempt log):**
- v1 (bf16): CE dropped fast (0.92 @ step 100 vs ~1.7 baseline) then **NaN @ step 150** at peak LR.
- v2 (fp32): same NaN @ step 150 → not bf16 overflow. Root cause: **sympy.simplify inflates constants to ~1.5e4** (raw data bounded to ≤100); `embedding * num_values` explodes activations.
- v3 (fp32 + `max_const=100` filter): stable, but **runtime sympy.simplify per-sample is too slow** (data-bound; stuck generating). Switched to a pre-generated cached dataset.
**CONFIG (final, `pool_canon` vs `pool_raw` control):** identical 3200-sample pools (seed 123), one canonicalized+constant-bounded, one raw; 3000 steps each. Same pool size/steps → memorization controlled; the *difference* isolates canonicalization.
**RESULT:**

| metric (held-out, 100 samples) | pool_raw | pool_canon |
|---|---|---|
| final CE (memorized) | 0.285 | **0.066** |
| final token_acc | 0.880 | **0.975** |
| exact token match | 14% | **45%** |
| structurally valid | 100% | 100% |
| func_equiv (R²>0.99) | 24% | 19% |
| val valid_rate (X-pad fix) | **1.00** | **1.00** |

**DECISION:** **Canonicalization cuts the CE floor 4× and triples exact-match** — direct confirmation that non-canonical generation is the dominant cause of the CE plateau. Recommend canonicalizing the training distribution. (Caveat: pools are memorized, so absolute CE is a lower bound; the *relative* 4× gap is the clean signal. Runtime sympy is too slow — pre-canonicalize a large cache.)

### Bonus finding — constant-value head underperforms
Even at 45% exact token match (canon), only 19% reach R²>0.99: the structure is right but **constant values are often off.** R² is invariant to additive constant shifts, which masks this. The numeric (xVal) head is a distinct, weaker sub-problem from structure prediction.

---

## Summary of key findings

- **Architecture is sound.** 100% structurally valid generations; recovers 35% of plateau-distribution functions at R²>0.9.
- **`valid_rate` metric is broken** (shape-mismatch `IndexError`) in `diagnostic/trial.py`, `inspect/trial.py`, `scale/trial.py`. Real validity ≈100%.
- **CE 0.71 = ambiguity floor**, driven by non-canonical expression generation. Scales with complexity; invariant to training scale; **4× reducible by canonicalization.**
- **λ=1.0 is harmful** to structure learning.
- **Constant prediction is a separate weakness** (right structure, wrong values).

## Recommended next steps for Jerry

1. **[HIGH] Fix the `valid_rate` / eval bug.** Pad `X` to `max_inputs` before `expr.evaluate(...)` in every `_evaluate`/inspection path (see `experiments/probe/trial.py:_pad_x`). This alone reframes the "41% plateau" — it was never real. *Unambiguous.*
2. **[HIGH] Canonicalize the training distribution.** Insert canonicalization into the generation pipeline. Concretely: pre-canonicalize a large cached dataset (sympy.simplify is too slow to run per-sample at train time — ~per-step data-bound). Even a lighter commutative-argument-sort canonicalizer (no full sympy) would capture most of the gain and run fast. Expected: CE floor drops substantially, exact-match and recovery rise. *Strong evidence; magnitude on fresh (non-memorized) data to be confirmed.*
3. **[MED] Re-tune λ.** λ=1 retards CE. The MSE head should either be small λ, applied only after structure stabilizes, or dropped. The "lambda ablation = identical" claim in the prior diagnosis does not hold at λ=1 (CE 1.27 vs 0.71). *Re-examine the ablation.*
4. **[MED] Improve the constant (xVal) head.** Right-structure / wrong-value is common. Consider constant refinement (the `Expression.fit` least-squares post-processing exists) or a stronger numeric head. *Inference from R² vs exact-match gap.*
5. **[LOW] Investigate per-operator difficulty** (div 19%, sin 26% recovery) — these are nonlinear/singular; may need targeted data or opset handling.

### Ambiguities to resolve with Jerry
- Is canonicalization an acceptable *data-distribution* change, or does it need to stay in the original (non-canonical) frame? I treated it as an allowed data-distribution knob; it's the highest-leverage fix but changes the target space.
- The full-run distribution (comprehensive 22-op, 10 vars, max_ops=15) is far harder than the diagnostic; canonicalization gains there are *inferred*, not measured. Recommend re-running the diagnostic→canonicalize pipeline on the full distribution.

## What worked / didn't / surprised me

**Worked:** SSH-based monitoring + `results/*.json` dumps fully replaced the unreachable tracker; the configurable `probe/` harness (canonicalize/num_vars/bf16/max_const/pool knobs + corrected structural & R² analysis) ran every experiment from one codebase.

**Didn't work:** Live per-sample sympy canonicalization (too slow on HPC → data-bound; had to pre-cache). bf16 + canonicalized data NaN'd (large constants × embedding). Two config edits silently dropped keys (`max_inputs`, `max_seq_len`) by replacing the wrong line — caught by py_compile / fast-fail.

**Surprised me:**
- The "41% valid_rate" centerpiece of the original diagnosis was almost entirely a harness bug (real ≈100%).
- The model does *better* on 3-op GTs (42% R²>0.9) than 1-op (28%) — recovery is bimodal and not monotone in complexity.
- Canonicalization's effect is large and clean (4× CE, 3× exact-match) even with memorization controlled.
