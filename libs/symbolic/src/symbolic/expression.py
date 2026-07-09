from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Tuple, Callable, Dict
from dataclasses import dataclass, field

import numpy.typing as npt
import numpy as np

if TYPE_CHECKING:
    import sympy as sp

UnaryCallable = Callable[[np.float64], np.float64]
BinaryCallable = Callable[[np.float64, np.float64], np.float64]


@dataclass(frozen=True)
class OperatorSet:
    operators: Dict[str, Tuple[int, Callable]]
    _name_to_index: Dict[str, int] = field(init=False, repr=False)
    _index_to_name: Dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        name_to_index = {}
        index_to_name = {}
        for index, opname in enumerate(self.operators.keys()):
            name_to_index[opname] = index
            index_to_name[index] = opname

        object.__setattr__(self, "_name_to_index", name_to_index)
        object.__setattr__(self, "_index_to_name", index_to_name)

    @classmethod
    def default(cls) -> "OperatorSet":
        return cls(
            operators={
                "add": (2, np.add),
                "sub": (2, np.subtract),
                "mul": (2, np.multiply),
                "div": (2, np.divide),
                "sin": (1, np.sin),
                "exp": (1, np.exp),
            }
        )

    def __getitem__(self, key: str) -> Tuple[int, Callable]:
        return self.operators[key]

    def name_to_code(self, opname: str) -> int:
        return self._name_to_index[opname]

    def code_to_name(self, index: int) -> str:
        return self._index_to_name[index]

    def by_index(self, index: int) -> Tuple[int, Callable]:
        opname = self.code_to_name(index)
        return self.operators[opname]


class Kind(Enum):
    input = 1
    const = 2
    cmd = 3


@dataclass(frozen=True)
class Ref:
    kind: Kind
    seq: int


@dataclass(frozen=True)
class ScoreResult:
    """Standard symbolic-regression metrics for an ``Expression`` on data."""

    r2: float
    mse: float
    mae: float
    complexity: int


@dataclass
class Expression:
    opset: OperatorSet
    num_inputs: int
    commands: npt.NDArray[np.uint16]
    constants: npt.NDArray[np.float64]
    output_index: int

    def render(self, feature_names: list[str] | None = None) -> str:
        const_base = self.num_inputs
        cmd_base = const_base + len(self.constants)
        cache: Dict[int, str] = {}

        def _render(idx: int) -> str:
            cached = cache.get(idx)
            if cached is not None:
                return cached
            if idx < const_base:
                s = feature_names[idx] if feature_names is not None else f"x{idx}"
            elif idx < cmd_base:
                s = repr(float(self.constants[idx - const_base]))
            else:
                opcode, p1, p2 = (int(v) for v in self.commands[idx - cmd_base])
                arity, _ = self.opset.by_index(opcode)
                name = self.opset.code_to_name(opcode)
                if arity == 1:
                    s = f"{name}({_render(p1)})"
                else:
                    s = f"{name}({_render(p1)}, {_render(p2)})"
            cache[idx] = s
            return s

        return _render(self.output_index)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"Expression({str(self)})"

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the expression on ``X`` of shape ``(num_samples, num_inputs)``.

        Returns a ``(num_samples,)`` array of predictions.
        """
        num_samples, num_inputs = X.shape
        node_offset = num_inputs + len(self.constants)
        memory = np.empty(
            (node_offset + len(self.commands), num_samples),
            dtype=np.float64,
        )
        memory[:num_inputs] = X.T
        memory[num_inputs:node_offset] = self.constants[:, np.newaxis]

        for i, row in enumerate(self.commands):
            opcode, p1, p2 = int(row[0]), int(row[1]), int(row[2])

            arity, func = self.opset.by_index(opcode)
            if arity == 1:
                result = func(memory[p1])
            else:
                result = func(memory[p1], memory[p2])

            memory[node_offset + i] = result

        return memory[self.output_index]

    def score(self, X: np.ndarray, y: np.ndarray) -> ScoreResult:
        """Compute standard SR metrics for this expression on data.

        ``X`` is ``(num_samples, num_inputs)``; ``y`` is ``(num_samples,)``.
        R² uses the population variance of ``y`` (guarded to ``1e-9`` when
        constant); complexity is the total DAG node count.
        """
        y_pred = self.evaluate(X)
        residual = y - y_pred
        mse = float(np.mean(residual ** 2))
        mae = float(np.mean(np.abs(residual)))
        var_y = float(np.var(y))
        denom = var_y if var_y != 0.0 else 1e-9
        r2 = float(1.0 - mse / denom)
        complexity = len(self.commands) + len(self.constants) + self.num_inputs
        return ScoreResult(r2=r2, mse=mse, mae=mae, complexity=complexity)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Expression:
        """Fit this expression's constants to data via nonlinear least-squares.

        Returns a new ``Expression`` with optimized constants; the original is
        not modified. Optimization starts from this expression's current
        constant values and uses Levenberg-Marquardt with a finite-difference
        Jacobian (computed by scipy).

        ``X`` is ``(num_samples, num_inputs)``; ``y`` is ``(num_samples,)``.
        """
        from scipy.optimize import least_squares

        def residuals(c: np.ndarray) -> np.ndarray:
            fitted = Expression(
                opset=self.opset,
                num_inputs=self.num_inputs,
                commands=self.commands,
                constants=c,
                output_index=self.output_index,
            )
            return fitted.evaluate(X) - y

        # No constants to tune: return an unchanged copy.
        if len(self.constants) == 0:
            return Expression(
                opset=self.opset,
                num_inputs=self.num_inputs,
                commands=self.commands,
                constants=self.constants.copy(),
                output_index=self.output_index,
            )

        result = least_squares(residuals, self.constants.copy(), method="lm")
        return Expression(
            opset=self.opset,
            num_inputs=self.num_inputs,
            commands=self.commands,
            constants=result.x,
            output_index=self.output_index,
        )

    def to_sympy(self, feature_names: list[str]) -> sp.Expr:
        """Render this expression as a sympy ``Expr`` over named variables."""
        from .bridge import to_sympy as _to_sympy

        return _to_sympy(self, feature_names)

    @classmethod
    def from_sympy(cls, source, feature_names, opset=None) -> Expression:
        """Build an ``Expression`` from a sympy expression or model string."""
        from .bridge import from_sympy as _from_sympy

        return _from_sympy(source, feature_names, opset)


class ExpressionBuilder:
    def __init__(self, opset: OperatorSet, num_inputs: int):
        self.opset = opset
        self.num_inputs = num_inputs
        self._commands: list[tuple[int, Ref, Ref]] = []
        self._constants: list[float] = []

    def input(self, i: int) -> Ref:
        if not 0 <= i < self.num_inputs:
            raise IndexError(f"input {i} out of range [0, {self.num_inputs})")
        return Ref(Kind.input, i)

    def constant(self, value: float) -> Ref:
        ref = Ref(Kind.const, len(self._constants))
        self._constants.append(float(value))
        return ref

    def apply(self, opname: str, *refs: Ref) -> Ref:
        arity, _ = self.opset[opname]
        if len(refs) != arity:
            raise ValueError(f"{opname} expects {arity} arg(s), got {len(refs)}")
        opcode = self.opset.name_to_code(opname)
        p1 = refs[0] if arity >= 1 else Ref(Kind.input, 0)
        p2 = refs[1] if arity >= 2 else Ref(Kind.input, 0)
        self._commands.append((opcode, p1, p2))
        return Ref(Kind.cmd, len(self._commands) - 1)

    def build(self, output: Ref) -> Expression:
        const_base = self.num_inputs
        cmd_base = const_base + len(self._constants)

        def resolve(ref: Ref) -> int:
            if ref.kind == Kind.input:
                return ref.seq
            if ref.kind == Kind.const:
                return const_base + ref.seq
            return cmd_base + ref.seq

        commands = np.array(
            [(op, resolve(a), resolve(b)) for op, a, b in self._commands],
            dtype=np.uint16,
        ).reshape(-1, 3)
        constants = np.array(self._constants, dtype=np.float64)
        return Expression(
            self.opset, self.num_inputs, commands, constants, resolve(output)
        )
