#!/usr/bin/env python3
"""Task 4: Expression.evaluate batch performance.

evaluate() is a vectorized numpy bytecode interpreter: a single call evaluates
one expression over all rows of X.  Cost per call scales with (#points) and
(#commands); total wall time scales linearly with (#expressions).  We time
batches of 100 / 500 / 1000 expressions, each evaluated on 100 points, and
report per-expression and per-point throughput.
"""
import os
import sys
import time
import statistics

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT, "libs", "symbolic", "src"))

from symbolic import OperatorSet
from symbolic.generation import (
    Pipeline, RandomBinaryTree, MantissaExponentConstants, UniformSamplePoints, is_valid,
)


def gen_expressions(opset, n, rng):
    p = (
        Pipeline(RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=rng))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(rng=rng))
        .filter(is_valid())
    )
    samples = list(p.iter(n))
    return samples


def bench(exprs_and_X, label, repeats=3):
    """exprs_and_X: list of (expr, X). Time evaluating every expr on its X."""
    # warmup
    for expr, X in exprs_and_X[:10]:
        expr.evaluate(X)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for expr, X in exprs_and_X:
            expr.evaluate(X)
        times.append(time.perf_counter() - t0)
    best = min(times)
    n = len(exprs_and_X)
    pts = exprs_and_X[0][1].shape[0]
    total_points = n * pts
    print(f"  {label:<28} {n:>5} expr x {pts} pts -> "
          f"{best*1000:8.1f} ms total | "
          f"{best/n*1000:7.3f} ms/expr | {total_points/best:10.0f} point-evals/s")
    return best


def main():
    np.random.seed(42)
    opset = OperatorSet.default()
    rng = np.random.default_rng(42)

    # generate a large pool of evaluated expressions once
    N = 1000
    samples = gen_expressions(opset, N, rng)
    cmds = [len(s.expression.commands) for s in samples]
    consts = [len(s.expression.constants) for s in samples]
    print("=" * 68)
    print(f"TASK 4: Expression.evaluate PERFORMANCE (default opset, max_ops=3)")
    print("=" * 68)
    print(f"  generated {len(samples)} expressions | "
          f"commands: mean={statistics.mean(cmds):.1f} max={max(cmds)} | "
          f"constants: mean={statistics.mean(consts):.1f} max={max(consts)}")

    # build (expr, X) pairs at each point count; reuse the same expressions,
    # vary the batch size.
    for pts in (100,):
        Xs = {s.expression.num_inputs: np.random.uniform(-10, 10, (pts, s.expression.num_inputs))
              .astype(np.float64) for s in samples}
        pairs = [(s.expression, Xs[s.expression.num_inputs]) for s in samples]

        print(f"\n  --- {pts} points per expression ---")
        bench(pairs[:100], f"batch=100 x {pts}pts", repeats=5)
        bench(pairs[:500], f"batch=500 x {pts}pts", repeats=3)
        bench(pairs[:1000], f"batch=1000 x {pts}pts", repeats=3)

    # also: single expression over very large point sets (does evaluate scale?)
    print(f"\n  --- single expression, scaling point count ---")
    big_expr = samples[0].expression
    for pts in (1_000, 10_000, 100_000):
        X = np.random.uniform(-10, 10, (pts, big_expr.num_inputs)).astype(np.float64)
        for _ in range(3):
            big_expr.evaluate(X)
        t0 = time.perf_counter()
        for _ in range(20):
            big_expr.evaluate(X)
        dt = (time.perf_counter() - t0) / 20
        print(f"  single expr x {pts:>7,} pts  -> {dt*1000:8.2f} ms/call | "
              f"{pts/dt:10,.0f} point-evals/s")


if __name__ == "__main__":
    main()
