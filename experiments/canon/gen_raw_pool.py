import pickle
import numpy as np
from symbolic import OperatorSet
from symbolic.generation import Pipeline, RandomBinaryTree, MantissaExponentConstants, UniformSamplePoints, is_valid

opset = OperatorSet.default()
rng = np.random.default_rng(42)
p = (
    Pipeline(RandomBinaryTree(opset, max_ops=3, num_vars=(1, 2), rng=rng))
    .then(MantissaExponentConstants(rng=rng))
    .then(UniformSamplePoints(rng=rng))
    .filter(is_valid())
)
pool = list(p.iter(5000))
with open("results/raw_pool.pkl", "wb") as f:
    pickle.dump(pool, f)
print(f"saved {len(pool)} raw samples to results/raw_pool.pkl", flush=True)
