import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from symbolic import (
        Expression,
        OperatorSet,
        Pipeline,
        RandomBinaryTree,
        MantissaExponentConstants,
        UniformSamplePoints,
        is_valid,
        r2,
        complexity,
    )
    from pysr import PySRRegressor
    import sympy as sp

    return (
        Expression,
        MantissaExponentConstants,
        OperatorSet,
        Pipeline,
        PySRRegressor,
        RandomBinaryTree,
        UniformSamplePoints,
        complexity,
        is_valid,
        mo,
        np,
        r2,
        sp,
    )


@app.cell
def _(mo):
    mo.md("""
    # Harness Validation: PySR Round-Trip

    Generate expressions through the pipeline, fit PySR on the data,
    convert PySR's output back to our Expression, and score it.

    This validates the end-to-end loop: **generate → model → score**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Generate ground-truth expressions
    """)
    return


@app.cell
def _(
    MantissaExponentConstants,
    OperatorSet,
    Pipeline,
    RandomBinaryTree,
    UniformSamplePoints,
    is_valid,
    np,
):
    rng = np.random.default_rng(42)
    gen = (
        Pipeline(RandomBinaryTree(
            opset=OperatorSet.default(),
            max_ops=5,
            num_vars=(1, 2),
            p_constant=0.3,
            rng=rng,
        ))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(lo=-10, hi=10, n=200, rng=rng))
        .filter(is_valid())
    )

    entries = list(gen.iter(5))
    print(f"Generated {len(entries)} valid entries")
    for _i, _e in enumerate(entries):
        _names = [f"x{j}" for j in range(_e.expression.num_inputs)]
        print(f"  {_i}: {_e.expression}  | y range [{_e.y.min():.2f}, {_e.y.max():.2f}]")
    return (entries,)


@app.cell
def _(mo):
    mo.md("""
    ## Run PySR on each expression
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    For each `(X, y)` pair, fit PySR and extract its best equation.
    PySR doesn't see the ground-truth expression — only the data.
    """)
    return


@app.cell
def _(PySRRegressor, entries):
    def run_pysr(X, y, n_iterations=50):
        """Fit PySR on (X, y), return the best equation as a sympy string."""
        _n_vars = X.shape[1]
        _var_names = [f"x{i}" for i in range(_n_vars)]
        est = PySRRegressor(
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "exp"],
            niterations=n_iterations,
            maxsize=25,
            deterministic=True,
            parallelism="serial",
            random_state=0,
            verbosity=0,
            model_selection="accuracy",
        )
        est.fit(X, y, variable_names=_var_names)
        return str(est.sympy())

    results = []
    for _i, entry in enumerate(entries[:10]):
        _n_ops = len(entry.expression.commands)
        if _n_ops < 2:
            continue

        model_str = run_pysr(entry.X, entry.y, n_iterations=100)
        _names = [f"x{j}" for j in range(entry.expression.num_inputs)]

        results.append({
            "idx": _i,
            "ground_truth": entry.expression,
            "pysr_model": model_str,
            "X": entry.X,
            "y": entry.y,
            "var_names": _names,
        })
        print(f"[{_i}] PySR found: {model_str}")
        print(f"     Ground truth: {entry.expression}")

    print(f"\nRan PySR on {len(results)} expressions")
    return (results,)


@app.cell
def _(mo):
    mo.md("""
    ## Convert and score
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Convert each PySR output back to our `Expression` via `from_sympy`,
    then score with `r2` and `complexity`.

    Symbolic comparison uses `to_sympy()` on our side — our internal
    DAG string format isn't sympy-parseable, so we go through sympy.
    """)
    return


@app.cell
def _(Expression, complexity, r2, results, sp):
    print(f"{'Idx':>3}  {'R2':>8}  {'Complex':>7}  {'Symbolic match':>15}  Ground truth / PySR")
    print("-" * 100)

    for r in results:
        expr = r["ground_truth"]
        _names = r["var_names"]

        try:
            predicted = Expression.from_sympy(r["pysr_model"], _names)
            pysr_r2 = r2(predicted, r["X"], r["y"])
            pysr_complexity = complexity(predicted, r["X"], r["y"])
        except ValueError as e:
            print(f"[{r['idx']:3d}]  CONVERSION FAILED: {e}")
            print(f"        Model: {r['pysr_model']}")
            continue

        true_sym = expr.to_sympy(_names)
        pred_sym = sp.sympify(r["pysr_model"])
        diff = sp.simplify(true_sym - pred_sym)
        symbolic_match = "EXACT" if diff == 0 else ("CONST_DIFF" if diff.is_constant() else "DIFF")

        print(
            f"{r['idx']:3d}  {pysr_r2:8.6f}  {pysr_complexity:7.0f}  "
            f"{symbolic_match:>15}  {expr}  |  {r['pysr_model']}"
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## What this tells us
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **If R² is high and symbolic match is EXACT/CONST_DIFF:**
    The harness loop works end-to-end. Pipeline produces valid data,
    PySR consumes it, conversion back to Expression works, scorers work.

    **If conversion fails:**
    PySR produced an operator outside our 6-op set (e.g. `square`, `pow`).
    Expected — tells us where the opset boundary bites.

    **If R² is low:**
    Either PySR couldn't find the equation (too few iterations, too complex),
    or the generated expression has numerical issues. Signal about
    expression difficulty, not a harness bug.
    """)
    return


if __name__ == "__main__":
    app.run()
