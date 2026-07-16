import numpy as np
from pysr import PySRRegressor

from symbolic import Expression, OperatorSet

# Finite map from symlab opset names to PySR operator strings.
_OPSET_TO_PYSR: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "sin": "sin",
    "exp": "exp",
}


class PySRModel:
    """PySR (SymbolicRegression.jl) adapter implementing ``SRModel``.

    Parameters
    ----------
    opset:
        Operator set the model searches over. Determines PySR's
        ``binary_operators`` and ``unary_operators`` via name translation.
    niterations:
        PySR search iterations (passed through to ``PySRRegressor``).
    maxsize:
        Max expression size in PySR's internal representation.
    rng:
        Seeded generator for reproducibility.
    """

    def __init__(
        self,
        opset: OperatorSet,
        niterations: int = 40,
        maxsize: int = 20,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.opset = opset
        self.rng = rng if rng is not None else np.random.default_rng()

        binary: list[str] = []
        unary: list[str] = []
        for name, (arity, _) in opset.operators.items():
            pysr_name = _OPSET_TO_PYSR.get(name)
            if pysr_name is None:
                raise ValueError(f"no PySR mapping for operator {name!r}")
            (binary if arity == 2 else unary).append(pysr_name)

        self._binary = binary
        self._unary = unary
        self._niterations = niterations
        self._maxsize = maxsize

    def _make_estimator(self) -> PySRRegressor:
        seed = int(self.rng.integers(0, 2**31))
        return PySRRegressor(
            binary_operators=self._binary,
            unary_operators=self._unary,
            niterations=self._niterations,
            maxsize=self._maxsize,
            deterministic=True,
            parallelism="serial",
            random_state=seed,
            verbosity=0,
            model_selection="accuracy",
        )

    def fit(
        self,
        problems: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[Expression | None]:
        results: list[Expression | None] = []
        for X, y in problems:
            n_inputs = X.shape[1]
            feature_names = [f"x{i}" for i in range(n_inputs)]
            try:
                est = self._make_estimator()
                est.fit(X, y, variable_names=feature_names)
                eq_str = str(est.sympy())
                expr = Expression.from_sympy(
                    eq_str,
                    feature_names=feature_names,
                    opset=self.opset,
                )
                results.append(expr)
            except Exception:
                results.append(None)
        return results
