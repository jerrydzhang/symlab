"""Data generation pipeline for symlab.

Composable, type-safe generation of ``(expression, X, y)`` tuples with known
ground truth. The :class:`Pipeline` builder enforces the
``Skeleton -> Populated -> Evaluated`` progression at type-check time.
"""

from .pipeline import Pipeline, Stage
from .stages import (
    MantissaExponentConstants,
    RandomBinaryTree,
    UniformSamplePoints,
    is_valid,
)
from .types import Evaluated, Populated, Skeleton

__all__ = [
    "Evaluated",
    "MantissaExponentConstants",
    "Pipeline",
    "Populated",
    "RandomBinaryTree",
    "Skeleton",
    "Stage",
    "UniformSamplePoints",
    "is_valid",
]

__version__ = "0.1.0"
