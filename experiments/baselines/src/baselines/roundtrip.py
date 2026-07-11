"""Round-trip: run PySR on one SRBench ground-truth problem through the official
pipeline, and report numeric fit + symbolic recovery.

The dataset path is a required argument — the script makes no assumption about
where PMLB lives on your machine. To fetch one problem:

    mkdir -p ~/data/pmlb && cd ~/data/pmlb
    git clone --filter=blob:none --sparse https://github.com/EpistasisLab/pmlb.git
    cd pmlb
    nix shell nixpkgs#git-lfs -c git lfs install
    git sparse-checkout set datasets/feynman_I_12_1
    nix shell nixpkgs#git-lfs -c git lfs pull --include="datasets/feynman_I_12_1/*"

The `git lfs install` step is required before the pull — without it, LFS
reports "Skipping object checkout, Git LFS is not installed for this
repository" and the .tsv.gz stays an LFS pointer file.

Then:

    uv run --package baselines python -m baselines.roundtrip \
        /path/to/pmlb/datasets/feynman_I_12_1/feynman_I_12_1.tsv.gz

Scaling is off by default so PySR recovers the equation in original form (for a
product, SRBench's default scaling introduces cross-terms that mask symbolic
recovery — fine for numeric R², not for this check). Pass --scale for the
faithful SRBench-default run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from srbench import assess, evaluate

from baselines.pysr import method


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "Round-trip PySR baseline"
    )
    ap.add_argument(
        "dataset", type=Path, help="PMLB .tsv.gz with a sibling metadata.yaml"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=2000,
        help="SRBench's subsampling knob (100k points is overkill here)",
    )
    ap.add_argument(
        "--scale",
        action="store_true",
        help="enable SRBench's default X/y scaling (off by default)",
    )
    args = ap.parse_args()

    m = method(random_state=args.seed)
    m.eval_kwargs = {"scale_x": args.scale, "scale_y": args.scale}

    res = evaluate(
        dataset=args.dataset,
        method=m,
        random_state=args.seed,
        max_train_samples=args.max_train_samples,
        fit_time_limit=600,
    )

    print("=" * 60)
    print(f"ROUND TRIP: PySR on {res.dataset}")
    print("=" * 60)
    print(f"r2_train = {res.r2_train:.4f}    r2_test = {res.r2_test:.4f}")
    print(f"model_size = {res.model_size}    fit_time = {res.time_time:.1f}s")
    print(f"symbolic_model = {res.symbolic_model!r}")
    print(f"simplified     = {res.simplified_symbolic_model!r}")
    print(f"true_model     = {res.true_model!r}")
    print(f"recovered_strict = {res.recovered_strict}")
    print(f"recovered        = {res.recovered}")
    # Exercise the substrate bridge: land the recovered model as our own
    # Expression, then re-score it through the official checker to confirm
    # the yardstick agrees with the in-pipeline verdict.
    expr = res.expression
    print("-" * 60)
    if expr is not None:
        print(
            f"expression     = {expr._render(res.feature_names)}   (canonical: {expr})"
        )
        re = assess(
            expression=expr,
            dataset=args.dataset,
            feature_names=res.feature_names,
            r2_test=res.r2_test,
        )
        print(
            f"re-scored recovered = {re.recovered}  (matches official: {re.recovered == res.recovered})"
        )
    else:
        print("expression     = None")


if __name__ == "__main__":
    main()
