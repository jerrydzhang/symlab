import numpy as np

from symbolic import (
    Evaluated,
    MantissaExponentConstants,
    OperatorSet,
    Pipeline,
    RandomBinaryTree,
    UniformSamplePoints,
    add_noise,
    is_valid,
    split,
)


def _make_evaluated(n: int = 100) -> Evaluated:
    rng = np.random.default_rng(0)
    p: Pipeline = (
        Pipeline(RandomBinaryTree(OperatorSet.default(), max_ops=4, rng=rng))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(lo=-5, hi=5, n=n, rng=rng))
        .filter(is_valid())
    )
    return next(p.iter(1))


def test_split_partitions_data():
    ev = _make_evaluated(100)
    train, test = split(ev, test_size=0.25, rng=np.random.default_rng(42))

    assert train.X.shape[0] == 75
    assert test.X.shape[0] == 25
    assert train.X.shape[0] + test.X.shape[0] == 100
    assert train.y.shape[0] == 75
    assert test.y.shape[0] == 25
    assert train.expression is ev.expression
    assert test.expression is ev.expression

    train_rows = {tuple(row) for row in train.X}
    test_rows = {tuple(row) for row in test.X}
    assert train_rows.isdisjoint(test_rows)


def test_split_zero_test_size_all_train():
    ev = _make_evaluated(50)
    train, test = split(ev, test_size=0.0, rng=np.random.default_rng(1))

    assert train.X.shape[0] == 50
    assert test.X.shape[0] == 0


def test_split_full_test_size_all_test():
    ev = _make_evaluated(50)
    train, test = split(ev, test_size=1.0, rng=np.random.default_rng(1))

    assert train.X.shape[0] == 0
    assert test.X.shape[0] == 50


def test_split_reproducible_with_same_seed():
    ev = _make_evaluated(80)
    train_a, test_a = split(ev, test_size=0.3, rng=np.random.default_rng(7))
    train_b, test_b = split(ev, test_size=0.3, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(train_a.X, train_b.X)
    np.testing.assert_array_equal(test_a.X, test_b.X)
    np.testing.assert_array_equal(train_a.y, train_b.y)
    np.testing.assert_array_equal(test_a.y, test_b.y)


def test_add_noise_modifies_y_keeps_x_and_expression():
    ev = _make_evaluated(100)
    noisy = add_noise(ev, level=0.1, rng=np.random.default_rng(3))

    assert not np.allclose(noisy.y, ev.y)
    np.testing.assert_array_equal(noisy.X, ev.X)
    assert noisy.expression is ev.expression


def test_add_noise_zero_level_unchanged():
    ev = _make_evaluated(100)
    noisy = add_noise(ev, level=0.0, rng=np.random.default_rng(3))

    np.testing.assert_array_equal(noisy.y, ev.y)


def test_add_noise_std_matches_level_times_rms():
    ev = _make_evaluated(10000)
    level = 0.15
    noisy = add_noise(ev, level=level, rng=np.random.default_rng(5))

    rms = np.sqrt(np.mean(ev.y ** 2))
    noise = noisy.y - ev.y
    empirical_std = np.std(noise)
    expected_std = level * rms
    assert abs(empirical_std - expected_std) < 0.03 * expected_std
