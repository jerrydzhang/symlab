import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    from pysr_model import PySRModel
    from symbolic import OperatorSet
    from symbolic.transforms import split, add_noise
    from symbolic.scoring import r2, complexity
    from symbolic.generation import (
        Pipeline,
        RandomBinaryTree,
        MantissaExponentConstants,
        UniformSamplePoints,
        is_valid,
    )

    return (
        MantissaExponentConstants,
        OperatorSet,
        Pipeline,
        PySRModel,
        RandomBinaryTree,
        UniformSamplePoints,
        add_noise,
        complexity,
        is_valid,
        np,
        r2,
        split,
    )


@app.cell
def _(np):
    rng = np.random.default_rng(7)
    return (rng,)


@app.cell
def _(
    MantissaExponentConstants,
    OperatorSet,
    Pipeline,
    RandomBinaryTree,
    UniformSamplePoints,
    is_valid,
    rng,
):
    pipeline = (
        Pipeline(RandomBinaryTree(OperatorSet.default(), rng=rng))
        .then(MantissaExponentConstants(rng=rng))
        .then(UniformSamplePoints(rng=rng))
        .filter(is_valid())
    )
    return (pipeline,)


@app.cell
def _(OperatorSet, PySRModel, rng):
    model = PySRModel(OperatorSet.default(), niterations=10, maxsize=20, rng=rng)
    return (model,)


@app.cell
def _(pipeline):
    results = list(pipeline.iter(5))
    results
    return (results,)


@app.cell
def _(add_noise, results, rng, split):
    train, test = split(results, 0.25, rng=rng)
    train = add_noise(train, 0.1, rng=rng)
    return test, train


@app.cell
def _(test, train):
    for _train, _test in zip(train, test):
        print(_train.expression, _test.expression)
    return


@app.cell
def _(model, train):
    preds = model.fit([(t.X, t.y) for t in train])
    return (preds,)


@app.cell
def _(preds, r2, test):
    scores = r2(preds, [t.X for t in test], [t.y for t in test])
    for _pred, _test, _score in zip(preds, test, scores):
        print(f"r2={_score:.3f}\ttruth={_test.expression}\tpred={_pred}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
