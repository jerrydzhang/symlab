import pickle, signal, sys
import numpy as np
from symbolic import OperatorSet
from symbolic.generation import (
    RandomBinaryTree,
    MantissaExponentConstants,
    UniformSamplePoints,
    is_valid,
    Populated,
)

opset = OperatorSet.comprehensive()

class TimeoutError(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutError()

def safe_simplify(expr):
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        return expr.simplify()
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)

target = int(sys.argv[1]) if len(sys.argv) > 1 else 12800
output = sys.argv[2] if len(sys.argv) > 2 else "pools/comprehensive_pool.pkl"

tree_rng = np.random.default_rng(789)
const_rng = np.random.default_rng(789)
pt_rng = np.random.default_rng(789)

gen = RandomBinaryTree(opset, max_ops=5, num_vars=(1, 2), rng=tree_rng)
sampler = UniformSamplePoints(rng=pt_rng)

pool = []
attempts = 0
while len(pool) < target and attempts < target * 30:
    attempts += 1
    skel = gen(None)
    pop = MantissaExponentConstants(rng=const_rng)(skel)
    simp = safe_simplify(pop.expression)
    if simp is None:
        continue
    pop2 = Populated(opset=opset, num_inputs=pop.num_inputs, expression=simp)
    c = np.abs(pop2.expression.constants)
    if len(c) and float(c.max()) > 100:
        continue
    ev = sampler(pop2)
    if not is_valid()(ev):
        continue
    pool.append(ev)
    if len(pool) % 1000 == 0:
        print(f"generated {len(pool)}/{target} (attempts={attempts})", flush=True)

with open(output, "wb") as f:
    pickle.dump(pool, f)
n_zero = sum(1 for s in pool if len(s.expression.constants) == 0)
print(f"saved {len(pool)} samples to {output} ({n_zero} zero-constant, {n_zero/len(pool):.1%})", flush=True)
print(f"acceptance rate: {len(pool)}/{attempts} ({len(pool)/attempts:.1%})", flush=True)
