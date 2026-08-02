# symlab Research Findings — Compiled

**Last updated:** 2026-08-01
**Status:** All experiments complete. Findings ready for analysis.

---

## Architecture

Transformer encoder-decoder for symbolic regression:
- **Encoder** (DataEncoder): takes (X, y) sample points, normalizes, projects to d_model, 3 self-attention layers. A stats token (X_mean, X_std, y_mean, y_std) is prepended.
- **Decoder** (TokenDecoder): autoregressive, 6 layers with self-attention + cross-attention. Emits two outputs per position:
  - `logit_head`: classification over vocabulary (operators, inputs, special tokens)
  - `numeric_head`: regression head predicting constant values at NUM token positions
- **35.8M params** (standard config: d_model=512, n_heads=8, d_ff=2048, n_enc=3, n_dec=6)
- **Constants flow:** tokenizer emits NUM token with raw constant value → `x = token_emb(tokens) * num_values.unsqueeze(-1)` → numeric head predicts log-transformed constant via MSE loss → `inverse_transform_constant` at generation time

---

## Experiment Inventory

All experiments use the **default opset** (6 ops: add, sub, mul, div, sin, exp) unless noted.
The **comprehensive opset** (22 ops, 10 vars, max_ops=15) is the real target — not yet run.

### Configurations Tested

| Config | Ops | Vars | max_ops | Pool size | Steps | λ | Canon? | Notes |
|---|---|---|---|---|---|---|---|---|
| easy | 1 | 1 | 1 | — | 5000 | 0 | No | Lowest-ambiguity baseline |
| diagnostic | 6 | 2 | 3 | — | 5000 | 0 | No | Architecture validation |
| pool_canon | 6 | 2 | 3 | 3200 | 3000 | 0 | Yes | Canon vs raw (memorized) |
| pool_raw | 6 | 2 | 3 | 3200 | 3000 | 0 | No | Canon control |
| canon_large | 6 | 2 | 3 | 12800 | 10000 | 0 | Yes | Canon at scale (non-memorized) |
| raw_large | 6 | 2 | 3 | 12800 | 10000 | 0 | No | Raw at scale |
| lambda_0_0 | 6 | 2 | 3 | 12800 | 10000 | 0 | Yes | λ sweep |
| lambda_0_01 | 6 | 2 | 3 | 12800 | 10000 | 0.01 | Yes | λ sweep |
| lambda_0_1 | 6 | 2 | 3 | 12800 | 10000 | 0.1 | Yes | λ sweep |
| **coupled** | 6 | 2 | 3 | 12800 | 10000 | 0.01 | Yes | Ablation: real constants |
| **skeleton** | 6 | 2 | 3 | 12800 | 10000 | 0.01 | Yes | Ablation: constants=1.0 |

---

## Finding 1: The "41% valid_rate" was a bug, not a model failure

**Status:** Resolved (fixed in library)

The original diagnostic showed valid_rate=41%. Root cause: eval path padded the decoded expression's `num_inputs` to `max_inputs` but passed X with fewer columns → `IndexError` → counted invalid.

**After fix:** structural validity is ~100%. The model was always producing valid expression trees.

**Fix:** `Expression.evaluate()` now validates shape and raises `ValueError` on mismatch. Experiments do explicit `np.pad(X, ((0,0), (0, max_inputs - X.shape[1])))` before calling evaluate.

---

## Finding 2: CE plateau is an ambiguity floor, not an architecture limit

**Status:** Confirmed

| Distribution | CE | token_acc | exact_match |
|---|---|---|---|
| 1 op, 1 var | 0.207 | 0.891 | 50% |
| 6 ops, 2 vars, 3 ops deep | 0.711 | 0.745 | ~14% |

- CE scales with distribution complexity (1-var/1-op → 2-var/3-op: CE 3.4×)
- 4× more training (50k steps) does not break the plateau — it's not scale-limited
- **Label ambiguity** is the cause: `add(x0, c)` and `add(c, x0)` produce identical (X, y) but different token streams. The model can't distinguish these — the CE loss is irreducible without removing the ambiguity.

---

## Finding 3: Canonicalization is the single biggest lever (at this scale)

Canonicalizing expressions (via sympy.simplify) removes commutative/associative ambiguity.

### Memorized (3200-sample pools, 3000 steps)

| Metric | Raw | Canon |
|---|---|---|
| CE | 0.285 | **0.066** |
| token_acc | 0.880 | **0.975** |
| exact_match | 14% | **45%** |
| func_equiv (R²>0.99) | 24% | 19% |

### Non-memorized (12800-sample pools, 10000 steps)

| Metric | Canon | Raw |
|---|---|---|
| CE | **0.178** | 0.281 |
| func_equiv (R²>0.99) | 16% | ~16% |
| func_equiv after fit() | 62-88% | 62-88% |

**Key insight:** canonicalization cuts CE dramatically (4× memorized, ~1.6× non-memorized) and triples exact-match. But func_equiv is surprisingly similar between canon and raw at scale — the model finds the right skeleton either way. The CE improvement is from reduced label ambiguity, not better function recovery.

---

## Finding 4: Expression.fit() is the dominant recovery lever

Post-processing generated expressions with least-squares constant fitting (scipy.optimize.least_squares) jumps func_equiv dramatically:

| | Without fit() | With fit() |
|---|---|---|
| Skeleton model | 13% | **80%** |
| Coupled model | 29% | **84%** |

The model consistently predicts the right skeleton but wrong constants. fit() recovers the constants and the expression matches.

**Constant error magnitude (from overnight analysis):**
- Only ~16% of expressions have correct constants (raw R² > 0.99)
- 63% have correct structure but constants so wrong that R² is **negative** (median raw R² = -0.013, mean = -554K)
- Example: `mul(0.114, exp(x0))` with R² = -29M — structure is perfect, constant is off by orders of magnitude
- These are not "refinements" — the model is wrong by 10-1000× on the constant value

---

## Finding 5: λ sweep — low MSE weight helps

| λ | CE | func_equiv |
|---|---|---|
| 0.0 | 0.178 | 16% |
| **0.01** | **0.145** | **34%** |
| 0.1 | 0.281 | 32% |

- λ=0.01 is the sweet spot: CE 0.145, funcEq 34%
- λ=0.1 hurts CE (0.281 vs 0.178 at λ=0) — the MSE gradient dominates
- Even a small constant prediction signal (λ=0.01) improves both CE and funcEq

---

## Finding 6: Coupled vs Skeleton — the thesis test

**The thesis:** forcing the model to predict meaningful constants improves its internal representations, leading to better structure prediction.

**The experiment:** two identical models, same data, same architecture. Only difference: skeleton model sees all constants as 1.0, coupled model sees real constants. Both get post-hoc fit() at eval.

### Results

| Metric | Skeleton (const=1.0) | Coupled (real const) | Delta |
|---|---|---|---|
| exact_match | 56% | 56% | **0** |
| **func_equiv_fit** (after fit) | 80% | 84% | +4% |
| func_equiv (raw constants) | 13% | 29% | +16% |
| high R² (>0.9, raw) | 27% | 45% | +18% |
| high R²_fit (>0.9) | 83% | 87% | +4% |
| mean R²_fit | -5.35 | -0.07 | much better |
| valid | 98% | 100% | +2% |

### Interpretation

**The thesis is weakly supported at this scale.** The coupled model predicts meaningfully better constants (29% vs 13% raw funcEq), and its failures are much closer to correct (mean R²_fit of -0.07 vs -5.35). But **structure prediction is identical** — 56% exact match in both.

The 4-point funcEq_fit edge (84% vs 80%) is real but modest. It's not coming from better structure — it's coming from fit() having a better starting point (coupled model's constants are closer to correct, so least-squares converges better).

**Conclusion:** at this problem scale (6 ops, 2 vars, max_ops=3), constants and structure appear to be *mostly orthogonal*. Learning constants helps you predict constants. It does not meaningfully change the internal representations for structure prediction.

**Caveat:** this is a toy problem. The structure space is small enough that both models can learn it equally well without needing constant signal. The comprehensive opset (22 ops, 10 vars, max_ops=15) has an exponentially larger structure space where representational pressure from constants could matter.

---

## Finding 7: Timing & Compute

| Component | GPU share | Notes |
|---|---|---|
| Training (fwd+bwd+opt) | ~91% | Backward 2.5× forward |
| collate_fn | ~6% | Negligible |
| Data gen (pooled) | <0.1% | Pools are free |
| Live canonicalization | 84× slower | Only viable with pre-generated pools |
| Expression.evaluate | ~9µs/expr | Never a bottleneck |

- **Encoder = 82% of FLOPs** (attends over 100 sample points vs ~8 decoder tokens)
- **Decoder = 70% of params** (6 layers × 2 attention sublayers vs 3 encoder layers)
- **Full training run: ~30 min** on 1× A100 for 10k steps (diagnostic config)
- This is very fast compared to other transformer SR papers (days of training)

---

## Finding 8: Architecture notes

- **Embedding multiplication issue:** `x = token_emb(tokens) * num_values` scales the entire NUM embedding by the raw constant value. For c=50, this multiplies the embedding by 50. The log-transform is only applied to the MSE target, never to the input scaling. This can cause activation explosion for large constants (discovered during canonicalization experiments).
- **Stats token** (X_mean, X_std, y_mean, y_std) is computed and passed to encoder but **constants are not normalized** relative to data statistics. The model must figure out the relationship between data scale and correct constant entirely through attention.
- **Numeric head architecture:** `Linear(d_model, d_ff) → GELU → Linear(d_ff, 1)`. Predicts log-transformed constant. Log transform: `sign(c) * log1p(|c|)` compresses [-100, 100] to [-4.6, 4.6].

---

## Open Questions

1. **Does the thesis hold at scale?** The coupled-vs-skeleton experiment at diagnostic scale (6 ops, 2 vars, max_ops=3) shows constants and structure are mostly orthogonal. The comprehensive opset (22 ops, 10 vars, max_ops=15) has a much larger structure space where the coupling could matter. This is the most important next experiment.

2. **Canonicalization design.** sympy.simplify is the current canonicalizer but is too slow for live generation (84× overhead). Options: (a) lightweight commutative-argument-sort canonicalizer (no full sympy), (b) always use pre-generated pools, (c) different canonicalization strategy. The choice affects the entire training distribution.

3. **Constant prediction architecture.** Current design has known issues (embedding multiplication, no constant normalization, value predicted before full structure is generated). Are there architecture changes that would make constant prediction more useful for representation learning?

4. **Training scale.** These experiments use 10k steps on 12800 samples — ~30 min on one A100. Other transformer SR papers train for days. Is the model undertrained at this scale? Would more compute change the findings?

5. **Per-operator difficulty.** div 19%, sin 26%, mul 32%, add 36%, exp 40%, sub 64% recovery. Why are div/sin so much harder? Is this a data distribution issue or an architectural one?
