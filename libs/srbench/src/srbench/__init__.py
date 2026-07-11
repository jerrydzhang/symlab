"""Thin driver over the official, vendored SRBench evaluation pipeline.

We do not reimplement SRBench's scoring; we run its official
``evaluate_model`` + ``assess_symbolic_model`` (vendored, pinned) against our
method, so results are comparable to the published benchmark. See
``srbench_upstream/NOTICE`` for provenance and licensing.
"""

from .driver import AssessResult, EvalResult, Method, assess, evaluate

__all__ = ["AssessResult", "EvalResult", "Method", "assess", "evaluate"]
