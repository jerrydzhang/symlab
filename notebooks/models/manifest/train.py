import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

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

    from manifest.model import TransformerModel
    from manifest.loss import compute_loss
    from manifest.tokenizer import XValsTokenizer

    return (
        MantissaExponentConstants,
        OperatorSet,
        Pipeline,
        RandomBinaryTree,
        UniformSamplePoints,
        is_valid,
        np,
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
    return


if __name__ == "__main__":
    app.run()
