#!/usr/bin/env python3
"""Flexible pool generator. Run on the HPC login node inside the container so
the pool lands directly in the project dir (no slow scp):

    apptainer exec --contain \
        --bind /home/jez21005/projects/symlab-overfit:/work --pwd /work \
        container.sif python experiments/canon/gen_pool_flex.py \
        --max-ops 5 --n 12800 --mode canon --out pools/canon_pool_m5.pkl

--mode raw  : no canonicalization (fast).
--mode canon: sympy.simplify per sample (2s cap) + |constant|<=100 filter (slow).
"""
import argparse
import pickle
import signal
import time

import numpy as np

from symbolic import OperatorSet
from symbolic.generation import (
    Pipeline,
    RandomBinaryTree,
    MantissaExponentConstants,
    UniformSamplePoints,
    is_valid,
    Populated,
)


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout()


def safe_simplify(expr, cap=2.0):
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, cap)
    try:
        return expr.simplify()
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ops", type=int, default=3)
    ap.add_argument("--n", type=int, default=12800)
    ap.add_argument("--mode", choices=["raw", "canon"], default="canon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=789)
    ap.add_argument("--max-const", type=float, default=100.0)
    ap.add_argument("--opset", choices=["default", "comprehensive"], default="default")
    args = ap.parse_args()

    opset = (OperatorSet.default() if args.opset == "default"
             else OperatorSet.comprehensive())
    tree_rng = np.random.default_rng(args.seed)
    const_rng = np.random.default_rng(args.seed + 1)
    pt_rng = np.random.default_rng(args.seed + 2)

    sampler = UniformSamplePoints(rng=pt_rng)

    if args.mode == "raw":
        # Fast vectorized pipeline.
        p = (
            Pipeline(RandomBinaryTree(opset, max_ops=args.max_ops,
                                      num_vars=(1, 2), rng=tree_rng))
            .then(MantissaExponentConstants(rng=const_rng))
            .then(UniformSamplePoints(rng=pt_rng))
            .filter(is_valid())
        )
        pool = list(p.iter(args.n * 2))[: args.n]
    else:
        gen = RandomBinaryTree(opset, max_ops=args.max_ops,
                               num_vars=(1, 2), rng=tree_rng)
        pool, attempts = [], 0
        t0 = time.time()
        while len(pool) < args.n and attempts < args.n * 40:
            attempts += 1
            skel = gen(None)
            pop = MantissaExponentConstants(rng=const_rng)(skel)
            simp = safe_simplify(pop.expression)
            if simp is None:
                continue
            pop2 = Populated(opset=opset, num_inputs=pop.num_inputs,
                             expression=simp)
            c = np.abs(pop2.expression.constants)
            if len(c) and float(c.max()) > args.max_const:
                continue
            ev = sampler(pop2)
            if not is_valid()(ev):
                continue
            pool.append(ev)
            if len(pool) % 1000 == 0:
                el = time.time() - t0
                print(f"  canon {len(pool)}/{args.n} in {el:.0f}s "
                      f"({len(pool)/max(el,1):.1f}/s, attempts={attempts})",
                      flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(pool, f)
    el = time.time() - t0 if args.mode == "canon" else 0.0
    print(f"saved {len(pool)} {args.mode} samples (max_ops={args.max_ops}) "
          f"to {args.out}" + (f" in {el:.0f}s" if args.mode == "canon" else ""),
          flush=True)


if __name__ == "__main__":
    main()
