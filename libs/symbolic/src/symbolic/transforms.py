import numpy as np

from symbolic.generation.types import Evaluated


def split(
    problems: list[Evaluated],
    test_size: float = 0.25,
    rng: np.random.Generator | None = None,
) -> tuple[list[Evaluated], list[Evaluated]]:
    """Split each problem's points into train/test lists."""
    gen = rng if rng is not None else np.random.default_rng()
    trains: list[Evaluated] = []
    tests: list[Evaluated] = []
    for evaluated in problems:
        n = evaluated.y.shape[0]
        order = gen.permutation(n)
        n_test = int(round(n * test_size))
        test_idx = order[:n_test]
        train_idx = order[n_test:]
        trains.append(
            Evaluated(
                opset=evaluated.opset,
                expression=evaluated.expression,
                X=evaluated.X[train_idx],
                y=evaluated.y[train_idx],
            )
        )
        tests.append(
            Evaluated(
                opset=evaluated.opset,
                expression=evaluated.expression,
                X=evaluated.X[test_idx],
                y=evaluated.y[test_idx],
            )
        )
    return trains, tests


def add_noise(
    problems: list[Evaluated],
    level: float,
    rng: np.random.Generator | None = None,
) -> list[Evaluated]:
    """Add Gaussian noise (std ``level * RMS(y)``) to each problem's y."""
    gen = rng if rng is not None else np.random.default_rng()
    out: list[Evaluated] = []
    for evaluated in problems:
        rms = np.sqrt(np.mean(evaluated.y ** 2))
        noise = gen.normal(0.0, level * rms, size=evaluated.y.shape)
        out.append(
            Evaluated(
                opset=evaluated.opset,
                expression=evaluated.expression,
                X=evaluated.X,
                y=evaluated.y + noise,
            )
        )
    return out
