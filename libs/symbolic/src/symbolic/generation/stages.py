from dataclasses import dataclass
from typing import Callable

import numpy as np

from symbolic import Expression, OperatorSet
from symbolic.expression import ExpressionBuilder, Ref

from .types import Evaluated, Populated, Skeleton


@dataclass
class _Node:
    """Mutable recursive tree node during skeleton generation (leaves: ``op is None``)."""

    op: str | None
    children: list["_Node"]
    const: bool = False
    var: int = 0


class RandomBinaryTree:
    """Generate uniform random unary-binary expression trees (Lample-Charton model).

    Returns a :class:`Skeleton` with placeholder (``0.0``) constants.

    Parameters
    ----------
    max_ops:
        Upper bound (inclusive) on operator nodes per tree.
    num_vars:
        Inclusive ``(lo, hi)`` range for the number of input variables.
    p_constant:
        Per-leaf probability of being a constant.
    rng:
        Seeded generator; defaults to a fresh one.
    """

    def __init__(
        self,
        opset: OperatorSet,
        max_ops: int = 5,
        num_vars: tuple[int, int] = (1, 2),
        p_constant: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.opset = opset
        self.max_ops = max_ops
        self.num_vars = num_vars
        self.p_constant = p_constant
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, input: None) -> Skeleton:
        n_ops = int(self.rng.integers(1, self.max_ops + 1))
        n_vars = int(self.rng.integers(self.num_vars[0], self.num_vars[1] + 1))

        root = self._grow(n_ops)
        self._label_leaves(root, n_vars)
        expression = self._build(root, n_vars)

        return Skeleton(
            opset=self.opset,
            num_inputs=n_vars,
            num_constants=len(expression.constants),
            expression=expression,
        )

    def _sample_op(self) -> str:
        names = list(self.opset.operators.keys())
        return names[int(self.rng.integers(0, len(names)))]

    def _grow(self, ops: int) -> _Node:
        if ops == 0:
            return _Node(op=None, children=[])

        opname = self._sample_op()
        arity, _ = self.opset[opname]

        if arity == 1:
            return _Node(op=opname, children=[self._grow(ops - 1)])

        remaining = ops - 1
        left_ops = int(self.rng.integers(0, remaining + 1))
        return _Node(
            op=opname,
            children=[self._grow(left_ops), self._grow(remaining - left_ops)],
        )

    def _label_leaves(self, root: _Node, n_vars: int) -> None:
        leaves = self._leaves(root)
        n = len(leaves)
        if n == 0:
            return

        is_const = [bool(self.rng.random() < self.p_constant) for _ in range(n)]

        for node, const in zip(leaves, is_const, strict=True):
            if const:
                node.const = True
            else:
                node.var = int(self.rng.integers(0, n_vars))

    @staticmethod
    def _leaves(node: _Node) -> list[_Node]:
        if node.op is None:
            return [node]
        out: list[_Node] = []
        for kid in node.children:
            out.extend(RandomBinaryTree._leaves(kid))
        return out

    def _build(self, node: _Node, num_inputs: int) -> Expression:
        builder = ExpressionBuilder(self.opset, num_inputs)

        def emit(n: _Node) -> Ref:
            if n.op is None:
                if n.const:
                    return builder.constant(0.0)
                return builder.input(n.var)
            children = [emit(kid) for kid in n.children]
            return builder.apply(n.op, *children)

        return builder.build(emit(node))


class MantissaExponentConstants:
    """Replace placeholder constants with ``sign * mantissa * 10**exponent`` values."""

    def __init__(
        self,
        sign: tuple[float, float] = (-1.0, 1.0),
        mantissa: tuple[float, float] = (0.1, 1.0),
        exponent: tuple[int, int] = (-2, 2),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.sign = sign
        self.mantissa = mantissa
        self.exponent = exponent
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, input: Skeleton) -> Populated:
        src = input.expression
        values = np.array(
            [self._sample_value() for _ in range(len(src.constants))],
            dtype=np.float64,
        )
        expression = Expression(
            opset=input.opset,
            num_inputs=input.num_inputs,
            commands=src.commands,
            constants=values,
            output_index=src.output_index,
        )
        return Populated(
            opset=input.opset,
            num_inputs=input.num_inputs,
            expression=expression,
        )

    def _sample_value(self) -> float:
        sign = self.sign[1] if self.rng.random() < 0.5 else self.sign[0]
        mantissa = float(self.rng.uniform(self.mantissa[0], self.mantissa[1]))
        exponent = int(self.rng.integers(self.exponent[0], self.exponent[1] + 1))
        return sign * mantissa * 10.0**exponent


class UniformSamplePoints:
    """Sample ``X`` uniformly and compute ``y`` by evaluating the expression."""

    def __init__(
        self,
        lo: float = -10.0,
        hi: float = 10.0,
        n: int = 100,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.lo = lo
        self.hi = hi
        self.n = n
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, input: Populated) -> Evaluated:
        X = self.rng.uniform(self.lo, self.hi, size=(self.n, input.num_inputs))
        y = input.expression.evaluate(X)
        return Evaluated(opset=input.opset, expression=input.expression, X=X, y=y)


def is_valid(overflow_threshold: float = 5e4) -> Callable[[Evaluated], bool]:
    """Build a filter that drops NaN, overflow, and trivially constant outputs."""

    def check(entry: Evaluated) -> bool:
        if np.any(np.isnan(entry.y)):
            return False
        if np.max(np.abs(entry.y)) > overflow_threshold:
            return False
        if np.var(entry.y) < 1e-12:
            return False
        return True

    return check


class Simplify:
    """Canonicalize structure via sympy.simplify; destructive to tree topology.

    Operators sympy introduces outside the opset raise ``ValueError``.
    """

    def __call__(self, input: Populated) -> Populated:
        simplified = input.expression.simplify()
        return Populated(
            opset=input.opset,
            num_inputs=input.num_inputs,
            expression=simplified,
        )
