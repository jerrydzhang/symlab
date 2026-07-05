from enum import Enum
from typing import Tuple, Callable, Dict
from dataclasses import dataclass, field
import numpy.typing as npt
import numpy as np

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


@dataclass
class Expression:
    opset: OperatorSet
    commands: npt.NDArray[np.uint16]
    constants: npt.NDArray[np.float64]
    output_index: int

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        num_inputs, num_samples = X.shape
        node_offset = num_inputs + len(self.constants)
        memory = np.empty(
            (node_offset + len(self.commands), num_samples),
            dtype=np.float64,
        )
        memory[:num_inputs] = X
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
        return Expression(self.opset, commands, constants, resolve(output))
