import pickle
import signal
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

opset = OperatorSet.default()

# Seed 456 — distinct from the small pools (seed 123/42) so the large pools
# are not a superset of the small ones.
tree_rng = np.random.default_rng(456)
const_rng = np.random.default_rng(456)
pt_rng = np.random.default_rng(456)


class TimeoutError(Exception):
    pass


def _handler(signum, frame):
    raise TimeoutError()


def safe_simplify(expr):
    # sympy.simplify can hang on nested expressions; cap at 2s and drop.
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        return expr.simplify()
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


# Pre-build skeletons + constants (fast), then simplify with timeout.
gen = RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=tree_rng)
sampler = UniformSamplePoints(rng=pt_rng)

target = 12800
pool = []
attempts = 0
while len(pool) < target and attempts < target * 20:
    attempts += 1
    skel = gen(None)
    pop = MantissaExponentConstants(rng=const_rng)(skel)
    simp = safe_simplify(pop.expression)
    if simp is None:
        continue
    pop2 = Populated(
        opset=opset, num_inputs=pop.num_inputs, expression=simp
    )
    c = np.abs(pop2.expression.constants)
    if len(c) and float(c.max()) > 100:
        continue
    ev = sampler(pop2)
    if not is_valid()(ev):
        continue
    pool.append(ev)
    if len(pool) % 1000 == 0:
        print(f"generated {len(pool)}/{target}", flush=True)

with open("pools/canon_pool_large.pkl", "wb") as f:
    pickle.dump(pool, f)
print(
    f"saved {len(pool)} canonicalized samples to pools/canon_pool_large.pkl "
    f"(attempts={attempts})",
    flush=True,
)
