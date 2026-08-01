from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable, Iterator
from typing import Dict, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
import sympy as sp

UnaryCallable = Callable[[np.float64], np.float64]
BinaryCallable = Callable[[np.float64, np.float64], np.float64]

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class InputNode:
    slot: int
    index: int


@dataclass(frozen=True, slots=True)
class ConstantNode:
    slot: int
    value: float


@dataclass(frozen=True, slots=True)
class OperatorNode:
    slot: int
    name: str
    operands: tuple[int, ...]


type ExpressionNode = InputNode | ConstantNode | OperatorNode

# name -> sympy builder, keyed by OperatorSet op names. The default opset uses
# exactly these names; custom opsets that reuse the names get the same mapping.
_SYMPY_OPS: dict[str, Callable[..., sp.Expr]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
}

# Reverse map: sympy function *class* -> op name, for the entries above whose
# builder is a class rather than a lambda. ``from_sympy`` dispatches on node
# type via this map; ``sqrt``/``square``/``cube`` (which compile to ``Pow``)
# are handled separately in ``_walk_pow``.
_SYMPY_FUNC_OPS: dict[type, str] = {
    fn: name
    for name, fn in _SYMPY_OPS.items()
    if isinstance(fn, type) and issubclass(fn, sp.Expr)
}


def _default_feature_names(tree: sp.Expr) -> list[str]:
    """Infer ``x0 .. xn`` feature names from a sympy tree's free symbols."""
    indices = []
    for s in tree.free_symbols:
        name = str(s)
        if not (name.startswith("x") and name[1:].isdigit()):
            raise ValueError(
                f"cannot default feature_names: symbol {name!r} is not of the "
                f"form 'x{{int}}'; pass feature_names explicitly"
            )
        indices.append(int(name[1:]))
    n = max(indices) + 1 if indices else 0
    return [f"x{i}" for i in range(n)]


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

    @classmethod
    def comprehensive(cls) -> "OperatorSet":
        """The full opset: the four binary ops plus all unary elementwise ops."""
        return cls(
            operators={
                "add": (2, np.add),
                "sub": (2, np.subtract),
                "mul": (2, np.multiply),
                "div": (2, np.divide),
                "sin": (1, np.sin),
                "cos": (1, np.cos),
                "tan": (1, np.tan),
                "asin": (1, np.arcsin),
                "acos": (1, np.arccos),
                "atan": (1, np.arctan),
                "sinh": (1, np.sinh),
                "cosh": (1, np.cosh),
                "tanh": (1, np.tanh),
                "asinh": (1, np.arcsinh),
                "acosh": (1, np.arccosh),
                "atanh": (1, np.arctanh),
                "exp": (1, np.exp),
                "log": (1, np.log),
                "sqrt": (1, np.sqrt),
                "square": (1, np.square),
                "cube": (1, lambda x: np.power(x, 3)),
                "abs": (1, np.abs),
            }
        )

    def __getitem__(self, key: str) -> Tuple[int, Callable]:
        return self.operators[key]

    def items(self) -> list[Tuple[str, Tuple[int, Callable]]]:
        return list(self.operators.items())

    def keys(self) -> list[str]:
        return list(self.operators.keys())

    def values(self) -> list[Tuple[int, Callable]]:
        return list(self.operators.values())

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


@dataclass
class Expression:
    opset: OperatorSet
    num_inputs: int
    commands: npt.NDArray[np.uint16]
    constants: npt.NDArray[np.float64]
    output_index: int

    def _node_at(self, slot: int) -> ExpressionNode:
        const_base = self.num_inputs
        cmd_base = const_base + len(self.constants)
        if slot < const_base:
            return InputNode(slot=slot, index=slot)
        if slot < cmd_base:
            return ConstantNode(
                slot=slot,
                value=float(self.constants[slot - const_base]),
            )

        opcode, p1, p2 = (int(v) for v in self.commands[slot - cmd_base])
        arity, _ = self.opset.by_index(opcode)
        operands = (p1,) if arity == 1 else (p1, p2)
        return OperatorNode(
            slot=slot,
            name=self.opset.code_to_name(opcode),
            operands=operands,
        )

    def iter_preorder(self) -> Iterator[ExpressionNode]:
        """Yield rooted semantic nodes in parent-first occurrence order."""
        stack = [self.output_index]
        while stack:
            node = self._node_at(stack.pop())
            yield node
            if isinstance(node, OperatorNode):
                stack.extend(reversed(node.operands))

    def _fold(
        self,
        input_fn: Callable[[InputNode], _T],
        constant_fn: Callable[[ConstantNode], _T],
        operator_fn: Callable[[OperatorNode, tuple[_T, ...]], _T],
    ) -> _T:
        cache: dict[int, _T] = {}

        def fold_at(slot: int) -> _T:
            if slot in cache:
                return cache[slot]

            node = self._node_at(slot)
            if isinstance(node, InputNode):
                result = input_fn(node)
            elif isinstance(node, ConstantNode):
                result = constant_fn(node)
            else:
                operands = tuple(fold_at(operand) for operand in node.operands)
                result = operator_fn(node, operands)
            cache[node.slot] = result
            return result

        return fold_at(self.output_index)

    def _render(self, feature_names: list[str] | None = None) -> str:
        def render_operator(node: OperatorNode, operands: tuple[str, ...]) -> str:
            return f"{node.name}({', '.join(operands)})"

        return self._fold(
            input_fn=lambda node: (
                feature_names[node.index]
                if feature_names is not None
                else f"x{node.index}"
            ),
            constant_fn=lambda node: repr(float(node.value)),
            operator_fn=render_operator,
        )

    def __str__(self) -> str:
        return self._render()

    def __repr__(self) -> str:
        return f"Expression({str(self)})"

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate on ``X`` ``(num_samples, num_inputs)`` -> ``(num_samples,)`` predictions.

        ``X`` must have exactly ``self.num_inputs`` columns — the memory
        layout is fixed at construction time. Mathematical edge cases (div
        by zero, log of negative) produce ``inf``/``nan`` rather than raising.
        """
        if X.shape[1] != self.num_inputs:
            raise ValueError(
                f"X has {X.shape[1]} columns; Expression expects {self.num_inputs}"
            )
        num_samples, num_inputs = X.shape
        node_offset = num_inputs + len(self.constants)
        memory = np.empty(
            (node_offset + len(self.commands), num_samples),
            dtype=np.float64,
        )
        memory[:num_inputs] = X.T
        memory[num_inputs:node_offset] = self.constants[:, np.newaxis]

        with np.errstate(divide="ignore", invalid="ignore"):
            for i, row in enumerate(self.commands):
                opcode, p1, p2 = int(row[0]), int(row[1]), int(row[2])

                arity, func = self.opset.by_index(opcode)
                if arity == 1:
                    result = func(memory[p1])
                else:
                    result = func(memory[p1], memory[p2])

                memory[node_offset + i] = result

        return memory[self.output_index]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Expression":
        """Fit constants to ``(X, y)`` via nonlinear least-squares.

        Returns a new ``Expression`` (original unchanged). ``X`` is
        ``(num_samples, num_inputs)``; ``y`` is ``(num_samples,)``.

        Each entry in ``constants`` is one optimizer coordinate. Repeated
        references to the same entry intentionally tie that parameter, while
        equal values stored in separate entries remain independent.
        """

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

    def to_sympy(self, feature_names: list[str] | None = None) -> sp.Expr:
        """Render as a sympy ``Expr``.

        ``feature_names`` defaults to ``x0 .. x{num_inputs-1}``; too few raise
        ``ValueError``.
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.num_inputs)]
        if len(feature_names) < self.num_inputs:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries; "
                f"Expression needs at least {self.num_inputs}"
            )
        syms: list[sp.Expr] = [
            sp.Symbol(name) for name in feature_names[: self.num_inputs]
        ]

        def convert_operator(
            node: OperatorNode, operands: tuple[sp.Expr, ...]
        ) -> sp.Expr:
            fn = _SYMPY_OPS.get(node.name)
            if fn is None:
                raise ValueError(f"no sympy mapping for operator {node.name!r}")
            return fn(*operands)

        return self._fold(
            input_fn=lambda node: syms[node.index],
            constant_fn=lambda node: sp.Float(float(node.value)),
            operator_fn=convert_operator,
        )

    @classmethod
    def from_sympy(
        cls,
        source: str | sp.Expr,
        feature_names: list[str] | None = None,
        opset: OperatorSet | None = None,
    ) -> "Expression":
        """Build an ``Expression`` from a sympy expr or string.

        ``feature_names`` fixes ``num_inputs`` (defaults to ``x0..xn`` inferred
        from free symbols). Operators outside ``opset`` raise ``ValueError``.
        """
        opset = opset or OperatorSet.default()
        if feature_names is None:
            tree = cast(
                sp.Expr, sp.sympify(source) if isinstance(source, str) else source
            )
            feature_names = _default_feature_names(tree)
        else:
            locals_ = {n: sp.Symbol(n) for n in feature_names}
            tree = cast(
                sp.Expr,
                sp.sympify(source, locals=locals_)  # ty: ignore[no-matching-overload]
                if isinstance(source, str)
                else source,
            )
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        b = ExpressionBuilder(opset, len(feature_names))

        def walk(node: sp.Expr) -> Ref:
            if node.is_Symbol:
                key = str(node)
                if key not in name_to_idx:
                    raise ValueError(f"unknown symbol {key!r}; not in feature_names")
                return b.input(name_to_idx[key])
            if node.is_Number:
                return b.constant(float(node))
            # Generic dispatch over any mapped sympy function class (sin, exp,
            # cos, tan, asin/acos/atan, sinh/cosh/tanh, asinh/acosh/atanh, log,
            # abs). sqrt/square/cube compile to Pow and are handled in _walk_pow.
            for sp_cls, opname in _SYMPY_FUNC_OPS.items():
                if isinstance(node, sp_cls) and opname in opset.operators:
                    return b.apply(opname, walk(cast(sp.Expr, node.args[0])))
            if isinstance(node, sp.Pow):
                return _walk_pow(node)
            if isinstance(node, sp.Mul):
                return _walk_mul(node)
            if isinstance(node, sp.Add):
                return _walk_add(node)
            if node == sp.E:
                return b.constant(float(sp.E))
            if node == sp.zoo:
                return b.constant(float("nan"))
            raise ValueError(f"unsupported sympy node: {node} ({type(node).__name__})")

        def _walk_pow(node: sp.Pow) -> Ref:
            base, e = node.args
            if e.is_Integer:
                n = int(e)
                # Prefer a single cube/square op over mul-expansion when the
                # opset provides one (default opset has neither -> unchanged).
                if n == 3 and "cube" in opset.operators:
                    return b.apply("cube", walk(base))
                if n == 2 and "square" in opset.operators:
                    return b.apply("square", walk(base))
                if n >= 1:
                    acc = walk(base)
                    for _ in range(n - 1):
                        acc = b.apply("mul", acc, walk(base))
                    return acc
                if n <= -1:
                    acc = walk(base)
                    for _ in range(-n - 1):
                        acc = b.apply("mul", acc, walk(base))
                    return b.apply("div", b.constant(1.0), acc)
            # x**(1/2) -> sqrt when available (sp.sqrt compiles to this form).
            if e == sp.Rational(1, 2) and "sqrt" in opset.operators:
                return b.apply("sqrt", walk(base))
            raise ValueError(f"unsupported exponent {e} in {node}")

        def _walk_mul(node: sp.Mul) -> Ref:
            coeff, rest = node.as_coeff_Mul()
            num: list[Ref] = []
            den: list[Ref] = []
            for f in sp.Mul.make_args(rest):
                if (
                    isinstance(f, sp.Pow)
                    and f.args[1].is_Integer
                    and int(f.args[1]) < 0
                ):
                    for _ in range(-int(f.args[1])):
                        den.append(walk(f.args[0]))
                else:
                    num.append(walk(f))
            if coeff != 1:
                num.insert(0, b.constant(float(coeff)))
            if not num:
                num.append(b.constant(1.0))
            acc = num[0]
            for r in num[1:]:
                acc = b.apply("mul", acc, r)
            for d in den:
                acc = b.apply("div", acc, d)
            return acc

        def _walk_add(node: sp.Add) -> Ref:
            pos_terms: list[sp.Expr] = []
            neg_terms: list[sp.Expr] = []
            for t in node.args:
                coeff = t if t.is_Number else t.as_coeff_Mul(rational=False)[0]
                (neg_terms if coeff < 0 else pos_terms).append(t)
            if pos_terms:
                acc = walk(pos_terms[0])
                for t in pos_terms[1:]:
                    acc = b.apply("add", acc, walk(t))
            else:
                acc = b.constant(0.0)
            for t in neg_terms:
                acc = b.apply("sub", acc, walk(-t))
            return acc

        return b.build(walk(tree))

    def simplify(self) -> "Expression":
        """Return a mathematically equal expression, normalized via sympy.

        Operators introduced outside this opset raise ``ValueError``; input
        order is preserved.
        """
        names = [f"x{i}" for i in range(self.num_inputs)]
        simplified = sp.simplify(self.to_sympy(names))
        return Expression.from_sympy(simplified, names, self.opset)


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
