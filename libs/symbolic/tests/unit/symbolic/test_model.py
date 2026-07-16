"""Tests for the :class:`~symbolic.SRModel` protocol."""

import numpy as np

from symbolic import Expression, OperatorSet, SRModel
from symbolic.expression import ExpressionBuilder


def _identity_expr() -> Expression:
    """``x0`` as a one-input, zero-command expression."""
    builder = ExpressionBuilder(OperatorSet.default(), 1)
    return builder.build(builder.input(0))


class _DummyModel:
    """Minimal concrete SRModel: echoes a fixed expression per problem."""

    def __init__(self, expr: Expression) -> None:
        self._expr = expr

    def fit(
        self,
        problems: list[tuple[np.ndarray, np.ndarray]],
        opset: OperatorSet,
    ) -> list[Expression | None]:
        # one result per problem; ignores the data, returns the fixed expr
        _ = opset
        return [self._expr for _ in problems]


class TestSRModelProtocol:
    def test_dummy_model_is_usable_as_srmodel(self):
        """A class with the matching fit signature is structurally an SRModel.

        ``ty`` enforces the ``model: SRModel`` assignment statically (structural
        subtyping); behaviorally, fit returns one result per problem.
        """
        expr = _identity_expr()
        model: SRModel = _DummyModel(expr)
        problems = [
            (np.zeros((4, 1)), np.zeros(4)),
            (np.ones((4, 1)), np.ones(4)),
        ]
        results = model.fit(problems, OperatorSet.default())
        assert len(results) == len(problems)
        for r in results:
            assert r is expr

    def test_dummy_model_handles_empty_batch(self):
        model: SRModel = _DummyModel(_identity_expr())
        assert model.fit([], OperatorSet.default()) == []

    def test_fit_result_is_usable_expression(self):
        expr = _identity_expr()
        model: SRModel = _DummyModel(expr)
        X = np.linspace(-1, 1, 5).reshape(5, 1)
        y = np.zeros(5)
        (result,) = model.fit([(X, y)], OperatorSet.default())
        assert result is not None
        # the returned expression evaluates without error on the data
        out = result.evaluate(X)
        assert out.shape == (5,)
