import pickle
import signal
import time
import numpy as np
from symbolic import OperatorSet
from symbolic.generation import RandomBinaryTree, MantissaExponentConstants, UniformSamplePoints, is_valid, Populated

opset = OperatorSet.default()
rng = np.random.default_rng(123)


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


tree_rng = np.random.default_rng(123)
gen = RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=tree_rng)
const_rng = np.random.default_rng(999)
pt_rng = np.random.default_rng(7)
sampler = UniformSamplePoints(rng=pt_rng)

target = 12800
pool = []
attempts = 0
t0 = time.time()
while len(pool) < target and attempts < target * 20:
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
    if len(pool) % 2000 == 0:
        elapsed = time.time() - t0
        print(f"generated {len(pool)}/{target} in {elapsed:.0f}s ({len(pool)/elapsed:.1f}/s)", flush=True)

with open("pools/canon_pool_large.pkl", "wb") as f:
    pickle.dump(pool, f)
elapsed = time.time() - t0
print(f"saved {len(pool)} canonicalized samples (attempts={attempts}, {elapsed:.0f}s)", flush=True)
