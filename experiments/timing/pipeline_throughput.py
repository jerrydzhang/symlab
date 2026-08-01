#!/usr/bin/env python3
"""Task 3: data-pipeline throughput. Generates 1000 samples under four regimes
and reports samples/sec:

  1. raw generation (no canonicalization)
  2. with canonicalization (sympy.simplify, 2 s timeout per expr)
  3. with canonicalization + constant filter (max_const=100)
  4. pool loading from .pkl

Sympy.simplify can hang on pathological expressions, so canonicalization uses a
signal-based per-expression timeout and reports the success / timeout split.
"""
import os
import sys
import time
import pickle
import signal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT, "libs", "symbolic", "src"))

from symbolic import OperatorSet
from symbolic.generation import (
    Pipeline, RandomBinaryTree, MantissaExponentConstants, UniformSamplePoints,
    is_valid, Populated, Simplify,
)


class TimedSafeSimplify:
    """Canonicalize via sympy.simplify with a hard per-call timeout.

    Returns None on timeout or error so the pipeline keeps flowing; those
    samples are dropped (matching the _SafeSimplify behaviour in trial.py).
    """
    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self.n_timeout = 0
        self.n_error = 0

    def _handler(self, signum, frame):
        raise TimeoutError

    def __call__(self, input: Populated):
        signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            simplified = input.expression.simplify()
        except TimeoutError:
            self.n_timeout += 1
            return None
        except Exception:
            self.n_error += 1
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        return Populated(opset=input.opset, num_inputs=input.num_inputs,
                         expression=simplified)


def const_filter(max_const):
    def ok(pop: Populated) -> bool:
        c = pop.expression.constants
        return len(c) == 0 or float(np.abs(c).max()) <= max_const
    return ok


def build(opset, rng, canonicalize=False, max_const=None, timeout=2.0):
    p = (
        Pipeline(RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=rng))
        .then(MantissaExponentConstants(rng=rng))
    )
    simp = None
    if canonicalize:
        simp = TimedSafeSimplify(timeout=timeout)
        p = p.then(simp)
        if max_const is not None:
            p = p.filter(const_filter(max_const))
    p = p.then(UniformSamplePoints(rng=rng)).filter(is_valid())
    return p, simp


def time_pipeline(label, opset, seed, n, canonicalize=False, max_const=None, timeout=2.0):
    rng = np.random.default_rng(seed)
    p, simp = build(opset, rng, canonicalize, max_const, timeout)
    t0 = time.perf_counter()
    samples = list(p.iter(n))
    dt = time.perf_counter() - t0
    rate = len(samples) / dt
    extra = ""
    if simp is not None:
        extra = f" | timeouts={simp.n_timeout} errors={simp.n_error}"
    print(f"  {label:<44} {len(samples):>5} in {dt:6.2f}s -> {rate:8.1f} samples/s{extra}")
    return len(samples), dt, rate


def main():
    opset = OperatorSet.default()
    N = 1000

    print("=" * 68)
    print(f"TASK 3: DATA PIPELINE THROUGHPUT (N={N}, default opset, max_ops=3)")
    print("=" * 68)

    # 1. raw
    time_pipeline("1. raw (no canonicalization)", opset, seed=42, n=N)

    # 2. canon
    time_pipeline("2. + canonicalization (simplify)", opset, seed=42, n=N, canonicalize=True)

    # 3. canon + max_const filter
    time_pipeline("3. + canon + max_const=100 filter", opset, seed=42, n=N,
                  canonicalize=True, max_const=100)

    # 4. pool loading from .pkl  (load whole pool, then measure sampling cost)
    print()
    for pf in ["raw_pool.pkl", "canon_pool.pkl", "raw_pool_large.pkl", "canon_pool_large.pkl"]:
        path = os.path.join(PROJECT, "pools", pf)
        if not os.path.exists(path):
            print(f"  pool {pf:<24} (missing, skip)")
            continue
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            pool = pickle.load(f)
        t_load = time.perf_counter() - t0
        size_mb = os.path.getsize(path) / 1e6

        # measure sampling N (with replacement) + the per-batch slice cost
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        idx = rng.integers(0, len(pool), size=N)
        _ = [pool[i] for i in idx]
        t_sample = time.perf_counter() - t0
        print(f"  pool {pf:<24} |{len(pool):>6} samples, {size_mb:5.1f} MB | "
              f"load {t_load*1000:6.1f} ms ({len(pool)/t_load:.0f} load-s/s) | "
              f"sample {N} {t_sample*1000:5.1f} ms ({N/t_sample:.0f} s/s)")


if __name__ == "__main__":
    main()
