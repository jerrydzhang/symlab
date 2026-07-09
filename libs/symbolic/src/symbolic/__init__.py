"""Symbolic-regression substrate: a flat command-DAG expression representation.

Core (``expression``) is numpy-only; CAS interop is exposed as methods on
:class:`Expression` (``to_sympy`` / ``from_sympy``) that import the sympy
bridge lazily, so ``import symbolic`` itself stays light.
"""
from .expression import Expression, OperatorSet, ScoreResult  # noqa: F401

__version__ = "0.1.0"
