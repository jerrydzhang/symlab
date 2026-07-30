from symbolic import (
    Expression,
    OperatorSet,
    InputNode,
    ConstantNode,
    OperatorNode,
)
from symbolic.expression import ExpressionBuilder, Ref

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NUM_ID = 3


class XValsTokenizer:
    def __init__(self, opset: OperatorSet, max_inputs: int) -> None:
        self.opset = opset
        self.max_inputs = max_inputs
        self.vocab: dict[str, int] = self._build_vocab(opset, max_inputs)
        self.id_to_token: list[str] = list(self.vocab.keys())

    def _build_vocab(self, opset: OperatorSet, max_inputs: int) -> dict[str, int]:
        vocab = {}
        # special tokens
        vocab["<PAD>"] = PAD_ID
        vocab["<BOS>"] = BOS_ID
        vocab["<EOS>"] = EOS_ID
        vocab["<NUM>"] = NUM_ID

        for i in range(max_inputs):
            vocab[f"<X{i}>"] = 4 + i

        n_special_tokens = len(vocab)
        for i, name in enumerate(opset.keys()):
            vocab[name] = i + n_special_tokens

        return vocab

    def encode(self, expr: Expression) -> tuple[list[int], list[float]]:
        token_ids = [self.vocab["<BOS>"]]
        numeric_vals = [1.0]
        for node in expr.iter_preorder():
            if isinstance(node, InputNode):
                if node.index >= self.max_inputs:
                    raise ValueError(
                        f"Input index {node.index} exceeds max_inputs {self.max_inputs}"
                    )

                token_ids.append(self.vocab[f"<X{node.index}>"])
                numeric_vals.append(1.0)
            elif isinstance(node, ConstantNode):
                token_ids.append(self.vocab["<NUM>"])
                numeric_vals.append(node.value)
            elif isinstance(node, OperatorNode):
                token_ids.append(self.vocab[node.name])
                numeric_vals.append(1.0)

        token_ids.append(self.vocab["<EOS>"])
        numeric_vals.append(1.0)
        return token_ids, numeric_vals

    def decode(
        self, token_ids: list[int], numeric_vals: list[float]
    ) -> Expression | None:
        if len(token_ids) != len(numeric_vals):
            raise ValueError("token_ids and numeric_vals must have the same length")

        if token_ids[0] != self.vocab["<BOS>"]:
            raise ValueError("Expected <BOS> token at the beginning of the sequence")

        builder = ExpressionBuilder(self.opset, self.max_inputs)
        # op, arity, children
        stack: list[tuple[str, int, list[Ref]]] = []
        root: Ref | None = None
        for token_id, numeric_val in zip(token_ids[1:], numeric_vals[1:]):  # skip <BOS>
            if token_id == self.vocab["<PAD>"]:
                break

            if token_id == self.vocab["<EOS>"]:
                break
            if token_id == self.vocab["<BOS>"]:
                # An untrained model can re-emit <BOS> mid-sequence; treat it
                # as a terminator (like <PAD>/<EOS>) so decode returns None
                # instead of key-erroring on opset lookup.
                break

            token_name = self.id_to_token[token_id]

            if token_name.startswith("<X"):
                current_ref = builder.input(int(token_name[2:-1]))
            elif token_name == "<NUM>":
                current_ref = builder.constant(numeric_val)
            else:
                arity, _ = self.opset[token_name]
                stack.append((token_name, arity, []))
                continue

            while True:
                if not stack:
                    if root:
                        # raise ValueError("Multiple root nodes found")
                        return None
                    root = current_ref
                    break

                frame = stack[-1]
                frame[2].append(current_ref)

                if len(frame[2]) < frame[1]:
                    break

                current_ref = builder.apply(frame[0], *frame[2])
                stack.pop()

        if root is None or stack:
            # raise ValueError("No root node found after processing tokens")
            return None

        return builder.build(root)
