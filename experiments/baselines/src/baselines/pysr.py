"""PySR (SymbolicRegression.jl) wrapped as an SRBench-consumable method.

PySR is downstream of SRBench: it provides an sklearn-compatible estimator;
SRBench's official pipeline calls its fit/predict and our ``model()`` returns a
sympy-compatible string. PySR stays under its own license (AGPL-3.0); this thin
wrapper is the only thing that touches it.
"""

from __future__ import annotations

from pysr import PySRRegressor

from srbench import Method


def make_estimator(random_state: int = 0) -> PySRRegressor:
    return PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "exp"],
        niterations=40,
        maxsize=20,
        deterministic=True,
        parallelism="serial",
        random_state=random_state,
        verbosity=0,
        model_selection="accuracy",
    )


def sympy_model(est, X=None) -> str:
    """Return the fitted best equation as a sympy-compatible string.

    PySR names variables after the DataFrame columns, which matches the dataset
    feature names — exactly what SRBench's ``clean_pred_model`` expects.
    """
    return str(est.sympy())


def method(random_state: int = 0) -> Method:
    return Method(name="pysr", est=make_estimator(random_state), model=sympy_model)
