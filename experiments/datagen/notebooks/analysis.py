import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from symbolic import OperatorSet
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
        RandomBinaryTree,
        UniformSamplePoints,
        complexity,
        is_valid,
        r2,
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
def _(pipeline):
    result = next(pipeline.iter(1))
    return (result,)


@app.cell
def _(result):
    expression = result.expression
    return (expression,)


@app.cell
def _(expression):
    expression
    return


@app.cell
def _(expression):
    expression.to_sympy()
    return


@app.cell
def _(expression):
    expression.simplify()
    return


@app.cell
def _(complexity, r2, result):
    r2(result.expression, result.X, result.y),  complexity(result.expression, result.X, result.y)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
