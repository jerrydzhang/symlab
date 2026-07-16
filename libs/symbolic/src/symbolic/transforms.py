from typing import Tuple

import numpy as np

from symbolic.pipeline.types import Evaluated


def split(
    evaluated: Evaluated,
    test_size: float = 0.25,
    rng: np.random.Generator | None = None,
) -> Tuple[Evaluated, Evaluated]:
    """Split points into train/test by ``test_size``; both share expression/opset."""
    gen = rng if rng is not None else np.random.default_rng()
    n = evaluated.y.shape[0]
    order = gen.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    return (
        Evaluated(
            opset=evaluated.opset,
            expression=evaluated.expression,
            X=evaluated.X[train_idx],
            y=evaluated.y[train_idx],
        ),
        Evaluated(
            opset=evaluated.opset,
            expression=evaluated.expression,
            X=evaluated.X[test_idx],
            y=evaluated.y[test_idx],
        ),
    )


def add_noise(
    evaluated: Evaluated,
    level: float,
    rng: np.random.Generator | None = None,
) -> Evaluated:
    """Return a copy with Gaussian noise (std ``level * RMS(y)``) added to ``y``."""
    gen = rng if rng is not None else np.random.default_rng()
    rms = np.sqrt(np.mean(evaluated.y ** 2))
    noise = gen.normal(0.0, level * rms, size=evaluated.y.shape)
    return Evaluated(
        opset=evaluated.opset,
        expression=evaluated.expression,
        X=evaluated.X,
        y=evaluated.y + noise,
    )
