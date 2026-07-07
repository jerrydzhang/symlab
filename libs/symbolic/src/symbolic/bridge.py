"""Bidirectional bridge between our ``Expression`` DAG and sympy.

Only this module pulls sympy into the substrate; ``expression.py`` stays
numpy-only, so ``import symbolic`` costs nothing extra. The operator
vocabulary is whatever an ``OperatorSet`` declares; the default opset
(``add sub mul div sin exp``) maps 1:1 onto sympy, and onto SRBench's
model-string parser.

The conversion is structural, not semantic-preserving under arbitrary
rewrites: ``from_sympy`` folds sympy's normalized n-ary ``Add``/``Mul`` and
integer ``Pow`` back into our binary command DAG, and raises on operators
outside the opset (e.g. ``sqrt``, ``log``) rather than silently inventing
them.
"""
from __future__ import annotations

import sympy as sp

from .expression import Expression, ExpressionBuilder, OperatorSet, Ref

# name -> sympy builder, keyed by OperatorSet op names. The default opset uses
# exactly these names; custom opsets that reuse the names get the same mapping.
_SYMPY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sin": sp.sin,
    "exp": sp.exp,
}


def to_sympy(expr: Expression, feature_names: list[str]) -> sp.Expr:
    """Render an ``Expression`` as a sympy ``Expr`` over named variables."""
    if len(feature_names) != expr.num_inputs:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries; "
            f"Expression expects {expr.num_inputs}"
        )
    syms = [sp.Symbol(n) for n in feature_names]
    const_base = expr.num_inputs
    cmd_base = const_base + len(expr.constants)
    cache: dict[int, sp.Expr] = {}

    def at(idx: int) -> sp.Expr:
        cached = cache.get(idx)
        if cached is not None:
            return cached
        if idx < const_base:
            v: sp.Expr = syms[idx]
        elif idx < cmd_base:
            v = sp.Float(float(expr.constants[idx - const_base]))
        else:
            row = expr.commands[idx - cmd_base]
            opcode, p1, p2 = int(row[0]), int(row[1]), int(row[2])
            arity, _ = expr.opset.by_index(opcode)
            name = expr.opset.code_to_name(opcode)
            fn = _SYMPY_OPS.get(name)
            if fn is None:
                raise ValueError(f"no sympy mapping for operator {name!r}")
            v = fn(at(p1)) if arity == 1 else fn(at(p1), at(p2))
        cache[idx] = v
        return v

    return at(expr.output_index)


def from_sympy(
    source: str | sp.Expr,
    feature_names: list[str],
    opset: OperatorSet | None = None,
) -> Expression:
    """Parse a sympy expression (string or ``sp.Expr``) into an ``Expression``.

    ``feature_names`` fixes both the variable order and the symbol table: only
    those names are treated as variables; everything else must reduce to a
    number. Operators outside ``opset`` raise ``ValueError``.
    """
    opset = opset or OperatorSet.default()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    locals_ = {n: sp.Symbol(n) for n in feature_names}
    tree = sp.sympify(source, locals=locals_) if isinstance(source, str) else source
    b = ExpressionBuilder(opset, len(feature_names))

    def walk(node: sp.Expr) -> Ref:
        if node.is_Symbol:
            key = str(node)
            if key not in name_to_idx:
                raise ValueError(f"unknown symbol {key!r}; not in feature_names")
            return b.input(name_to_idx[key])
        if node.is_Number:
            return b.constant(float(node))
        if isinstance(node, sp.sin) and "sin" in opset.operators:
            return b.apply("sin", walk(node.args[0]))
        if isinstance(node, sp.exp) and "exp" in opset.operators:
            return b.apply("exp", walk(node.args[0]))
        if isinstance(node, sp.Pow):
            return _walk_pow(node)
        if isinstance(node, sp.Mul):
            return _walk_mul(node)
        if isinstance(node, sp.Add):
            return _walk_add(node)
        raise ValueError(f"unsupported sympy node: {node} ({type(node).__name__})")

    def _walk_pow(node: sp.Pow) -> Ref:
        base, e = node.args
        if e.is_Integer:
            n = int(e)
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
        raise ValueError(f"unsupported exponent {e} in {node}")

    def _walk_mul(node: sp.Mul) -> Ref:
        coeff, rest = node.as_coeff_Mul()
        num: list[Ref] = []
        den: list[Ref] = []
        for f in sp.Mul.make_args(rest):
            if isinstance(f, sp.Pow) and f.args[1].is_Integer and int(f.args[1]) < 0:
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
