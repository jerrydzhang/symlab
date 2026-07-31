import pickle
import signal
import numpy as np
from symbolic import OperatorSet
from symbolic.generation import Pipeline, RandomBinaryTree, MantissaExponentConstants, UniformSamplePoints, is_valid, Populated

opset = OperatorSet.default()
rng = np.random.default_rng(123)


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
tree_rng = np.random.default_rng(123)
gen = RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=tree_rng)
const_rng = np.random.default_rng(999)
pt_rng = np.random.default_rng(7)
sampler = UniformSamplePoints(rng=pt_rng)

target = 3200
pool = []
attempts = 0
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
    if len(pool) % 500 == 0:
        print(f"generated {len(pool)}/{target}", flush=True)

with open("results/canon_pool.pkl", "wb") as f:
    pickle.dump(pool, f)
print(f"saved {len(pool)} canonicalized samples (attempts={attempts})", flush=True)
