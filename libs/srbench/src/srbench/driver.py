"""Thin driver over the official, vendored SRBench evaluation pipeline.

We do NOT reimplement SRBench's scoring. We run its official ``evaluate_model``
+ ``assess_symbolic_model`` (vendored, pinned, see ``srbench_upstream/NOTICE``)
against our method, so results are bit-for-bit comparable to the published
benchmark. SRBench is downstream of us: it consumes our sklearn-compatible
estimator; our method code stays under its own license.

Two entry points:

* :func:`evaluate` — run a full official trial of a ``Method`` (fit + score +
  symbolic recovery) on a dataset. The recovered model is also exposed as our
  own :class:`~symbolic.Expression` via ``EvalResult.expression``.
* :func:`assess` — score one of *our* ``Expression``\\ s through the same
  official symbolic checker (no fit). This is the path a learned model takes:
  it hands an ``Expression`` to the yardstick and gets SRBench-comparable
  recovery flags back.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from symbolic import Expression

_UPSTREAM_DIR = Path(__file__).parent / "srbench_upstream"


def _import_official():
    """Import the vendored SRBench modules.

    SRBench's ``experiment/`` scripts are flat modules that import each other
    by bare name (``from utils import jsonify``), so we put their directory on
    ``sys.path`` rather than packaging them as a proper package.
    """
    upstream = str(_UPSTREAM_DIR)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    import assess_symbolic_model  # noqa: F401
    import evaluate_model  # noqa: F401
    return evaluate_model, assess_symbolic_model


def _feature_names(dataset: str | Path) -> list[str]:
    """Feature column names of a PMLB dataset (everything except ``target``)."""
    import pandas as pd

    cols = pd.read_csv(dataset, sep="\t", compression="infer", nrows=0).columns
    return [c for c in cols if c != "target"]


@dataclass
class Method:
    """Our handle for an SRBench-consumable method.

    ``est`` is any sklearn-compatible estimator; ``model(est, X)`` returns the
    fitted best equation as a sympy-compatible string (the contract SRBench's
    ``evaluate_model`` expects). ``eval_kwargs`` forwards to SRBench (e.g.
    ``scale_x``/``scale_y``).
    """

    name: str
    est: Any
    model: Callable[..., str]
    eval_kwargs: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    dataset: str
    algorithm: str
    random_state: int
    symbolic_model: str
    r2_train: float
    r2_test: float
    mse_test: float
    mae_test: float
    model_size: int
    simplicity: float
    time_time: float
    feature_names: list[str] = field(default_factory=list)
    true_model: str | None = None
    simplified_symbolic_model: str | None = None
    simplified_complexity: int | None = None
    # symbolic recovery (populated only when r2_test > 0.5, per SRBench gating)
    symbolic_error_is_zero: bool | None = None
    symbolic_error_is_constant: bool | None = None
    symbolic_fraction_is_constant: bool | None = None

    @property
    def recovered_strict(self) -> bool | None:
        """SRBench strict recovery: ``str(true - pred) == '0'``."""
        return self.symbolic_error_is_zero

    @property
    def recovered(self) -> bool | None:
        """Lenient equivalence: exact, constant diff, or constant fraction."""
        if self.symbolic_error_is_zero is None:
            return None
        return (
            self.symbolic_error_is_zero
            or self.symbolic_error_is_constant
            or self.symbolic_fraction_is_constant
        )

    @property
    def expression(self) -> "Expression | None":
        """The recovered model as our ``Expression`` (None if not recoverable).

        Translates ``simplified_symbolic_model`` via :func:`symbolic.bridge.from_sympy`.
        Raises ``ValueError`` if the model uses an operator outside the default
        opset (e.g. ``sqrt``, ``log``) — extend the opset to round-trip those.
        """
        if not self.simplified_symbolic_model or not self.feature_names:
            return None
        from symbolic.bridge import from_sympy

        return from_sympy(self.simplified_symbolic_model, self.feature_names)


def evaluate(
    *,
    dataset: str | Path,
    method: Method,
    random_state: int = 0,
    target_noise: float = 0.0,
    feature_noise: float = 0.0,
    sym_data: bool = True,
    max_train_samples: int = 0,
    fit_time_limit: int = 3600,
) -> EvalResult:
    """Run one official SRBench trial of ``method`` on ``dataset``.

    Delegates to SRBench's ``evaluate_model`` (numeric + symbolic model) then
    ``assess_symbolic_model`` (gated symbolic recovery), reading back the merged
    JSON. Defaults mirror the official ground-truth experiment.
    """
    evaluate_model, assess_symbolic_model = _import_official()
    dataset = str(Path(dataset).expanduser())
    feature_names = _feature_names(dataset)

    tmpdir = tempfile.mkdtemp(prefix="srbench_")
    try:
        kwargs = dict(method.eval_kwargs)
        if max_train_samples:
            kwargs["max_train_samples"] = max_train_samples

        json_path = evaluate_model.evaluate_model(
            dataset=dataset,
            results_path=tmpdir,
            random_state=random_state,
            est_name=method.name,
            est=method.est,
            model=method.model,
            algorithm=None,
            test=False,
            target_noise=target_noise,
            feature_noise=feature_noise,
            sym_data=sym_data,
            fit_time_limit=fit_time_limit,
            **kwargs,
        )

        # second official step: symbolic equivalence (writes <json_path>.updated)
        assess_symbolic_model.assess_symbolic_model_from_file(json_path, dataset)
        updated = json_path + ".updated"
        data = json.load(open(updated if Path(updated).exists() else json_path))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return EvalResult(
        dataset=data.get("dataset", ""),
        algorithm=data.get("algorithm", method.name),
        random_state=data.get("random_state", random_state),
        symbolic_model=data.get("symbolic_model", ""),
        r2_train=data.get("r2_train", float("nan")),
        r2_test=data.get("r2_test", float("nan")),
        mse_test=data.get("mse_test", float("nan")),
        mae_test=data.get("mae_test", float("nan")),
        model_size=data.get("model_size", -1),
        simplicity=data.get("simplicity", float("nan")),
        time_time=data.get("time_time", float("nan")),
        feature_names=feature_names,
        true_model=data.get("true_model"),
        simplified_symbolic_model=data.get("simplified_symbolic_model"),
        simplified_complexity=data.get("simplified_complexity"),
        symbolic_error_is_zero=data.get("symbolic_error_is_zero"),
        symbolic_error_is_constant=data.get("symbolic_error_is_constant"),
        symbolic_fraction_is_constant=data.get("symbolic_fraction_is_constant"),
    )


@dataclass
class AssessResult:
    """Symbolic-recovery verdict for a single ``Expression``, SRBench-comparable."""

    dataset: str
    symbolic_model: str
    true_model: str | None = None
    simplified_symbolic_model: str | None = None
    simplified_complexity: int | None = None
    symbolic_error: str | None = None
    symbolic_error_is_zero: bool | None = None
    symbolic_error_is_constant: bool | None = None
    symbolic_fraction_is_constant: bool | None = None
    sympy_exception: str | None = None

    @property
    def recovered_strict(self) -> bool | None:
        return self.symbolic_error_is_zero

    @property
    def recovered(self) -> bool | None:
        if self.symbolic_error_is_zero is None:
            return None
        return (
            self.symbolic_error_is_zero
            or self.symbolic_error_is_constant
            or self.symbolic_fraction_is_constant
        )


def assess(
    *,
    expression: "Expression",
    dataset: str | Path,
    feature_names: list[str],
    r2_test: float,
    algorithm: str = "ours",
    random_state: int = 0,
) -> AssessResult:
    """Score one of our ``Expression``\\ s via the OFFICIAL SRBench checker.

    Renders the Expression to a model string and runs SRBench's
    ``assess_symbolic_model_from_file`` — the same ``clean_pred_model`` +
    ``simplify`` + symbolic-diff path the full pipeline uses — so the recovery
    flags are directly comparable to a method that went through ``evaluate``.

    ``r2_test`` is required because SRBench gates symbolic checking on
    ``r2_test > 0.5``; compute it yourself (e.g. via ``Expression.evaluate``
    on a held-out split) and pass it in.
    """
    evaluate_model, assess_symbolic_model = _import_official()
    dataset = str(Path(dataset).expanduser())
    model_str = expression.render(feature_names)

    tmpdir = tempfile.mkdtemp(prefix="srbench_assess_")
    try:
        record = {
            "dataset": Path(dataset).name,
            "algorithm": algorithm,
            "random_state": random_state,
            "symbolic_model": model_str,
            "r2_test": float(r2_test),
        }
        json_path = Path(tmpdir) / f"{record['dataset']}_{algorithm}_{random_state}.json"
        json.dump(record, open(json_path, "w"))
        assess_symbolic_model.assess_symbolic_model_from_file(str(json_path), dataset)
        updated = str(json_path) + ".updated"
        data = json.load(open(updated if Path(updated).exists() else json_path))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return AssessResult(
        dataset=data.get("dataset", record["dataset"]),
        symbolic_model=data.get("symbolic_model", model_str),
        true_model=data.get("true_model"),
        simplified_symbolic_model=data.get("simplified_symbolic_model"),
        simplified_complexity=data.get("simplified_complexity"),
        symbolic_error=data.get("symbolic_error"),
        symbolic_error_is_zero=data.get("symbolic_error_is_zero"),
        symbolic_error_is_constant=data.get("symbolic_error_is_constant"),
        symbolic_fraction_is_constant=data.get("symbolic_fraction_is_constant"),
        sympy_exception=data.get("sympy_exception"),
    )
