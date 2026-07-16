import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    from random_bfgs import RandomBFGSModel
    from symbolic import OperatorSet
    from symbolic.scoring import r2, complexity ,
    from symbolic.generation import (
        Pipeline,
        RandomBinaryTree,
        MantissaExponentConstants,
        UniformSamplePoints,
        is_valid,
    )
    """,
    name="_"
)


@app.cell
def _(
    MantissaExponentConstants,
    OperatorSet,
    Pipeline,
    RandomBinaryTree,
    UniformSamplePoints,
    is_valid,
):
    pipeline = (
        Pipeline(RandomBinaryTree(OperatorSet.default()))
        .then(MantissaExponentConstants())
        .then(UniformSamplePoints())
        .filter(is_valid())
    )
    return (pipeline,)


@app.cell
def _(RandomBFGSModel):
    model = RandomBFGSModel()
    return (model,)


@app.cell
def _(pipeline):
    results = list(pipeline.iter(5))
    results
    return (results,)


@app.cell
def _(OperatorSet, model, results):
    preds = model.fit([(res.X, res.y) for res in results], OperatorSet.default())
    return (preds,)


@app.cell
def _(preds, r2, results):
    for res, pred in zip(results, preds):
        print(f"orig: {res.expression}")
        print(f"pred: {pred}")
        print(r2(pred, res.X, res.y))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
