"""Symbolic-regression substrate: a flat command-DAG expression representation.

Core (``expression``) is numpy-only; CAS interop is exposed as methods on
:class:`Expression` (``to_sympy`` / ``from_sympy``) that import the sympy
bridge lazily, so ``import symbolic`` itself stays light. The data-generation
:mod:`~symbolic.pipeline` and the :mod:`~symbolic.scoring` free functions are
re-exported here for convenience.
"""

from .expression import Expression, OperatorSet  # noqa: F401
from .pipeline import (  # noqa: F401
    Evaluated,
    MantissaExponentConstants,
    Pipeline,
    Populated,
    RandomBinaryTree,
    Skeleton,
    Stage,
    UniformSamplePoints,
    is_valid,
)
from .scoring import complexity, r2  # noqa: F401

__all__ = [
    "Evaluated",
    "Expression",
    "MantissaExponentConstants",
    "OperatorSet",
    "Pipeline",
    "Populated",
    "RandomBinaryTree",
    "Skeleton",
    "Stage",
    "UniformSamplePoints",
    "complexity",
    "is_valid",
    "r2",
]

__version__ = "0.1.0"
