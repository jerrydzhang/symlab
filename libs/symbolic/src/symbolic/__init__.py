"""Symbolic-regression substrate: a flat command-DAG expression representation.

Core (``expression``) is numpy-only. The sympy bridge (``bridge``) is imported
explicitly by callers that need CAS interop, so ``import symbolic`` itself stays
light.
"""
from .expression import (  # noqa: F401
    Expression,
    ExpressionBuilder,
    Kind,
    OperatorSet,
    Ref,
)
from .metrics import ScoreResult, score  # noqa: F401

__version__ = "0.1.0"
