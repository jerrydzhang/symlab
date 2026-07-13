"""Symbolic-regression substrate: a flat command-DAG expression representation.

CAS interop (``to_sympy`` / ``from_sympy`` / ``simplify``) lives directly on
:class:`Expression`. The data-generation :mod:`~symbolic.pipeline` and the
:mod:`~symbolic.scoring` free functions are re-exported here for convenience.
"""

from .expression import Expression, OperatorSet  # noqa: F401
from .model import SRModel  # noqa: F401
from .pipeline import (  # noqa: F401
    Evaluated,
    MantissaExponentConstants,
    Pipeline,
    Populated,
    RandomBinaryTree,
    Simplify,
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
    "SRModel",
    "Simplify",
    "Skeleton",
    "Stage",
    "UniformSamplePoints",
    "complexity",
    "is_valid",
    "r2",
]

__version__ = "0.1.0"
