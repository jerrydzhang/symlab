import pickle
import numpy as np
from symbolic import OperatorSet
from symbolic.generation import (
    Pipeline,
    RandomBinaryTree,
    MantissaExponentConstants,
    UniformSamplePoints,
    is_valid,
)

opset = OperatorSet.default()

# Seed 456 — distinct from the small raw pool (seed 42).
rng = np.random.default_rng(456)
p = (
    Pipeline(RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=rng))
    .then(MantissaExponentConstants(rng=rng))
    .then(UniformSamplePoints(rng=rng))
    .filter(is_valid())
)
pool = list(p.iter(20000))  # iterate extra; will slice to 12800 below
pool = pool[:12800]
with open("pools/raw_pool_large.pkl", "wb") as f:
    pickle.dump(pool, f)
print(
    f"saved {len(pool)} raw samples to pools/raw_pool_large.pkl",
    flush=True,
)
