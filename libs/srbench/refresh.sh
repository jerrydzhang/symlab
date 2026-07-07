#!/usr/bin/env bash
# Refresh the vendored srbench scoring slice from upstream cavalab/srbench.
#
# Sparse-clones srbench at a given commit, copies the scoring files into
# srbench_upstream/, rewrites PIN, and asserts the vendored set still forms a
# closed import cluster — the invariant srbench.driver's sys.path trick relies
# on: every bare-name import must resolve to a file we ship or a declared dep.
#
# Our own srbench_upstream/{__init__.py,NOTICE} are preserved; only PIN is
# rewritten and only the scoring files are overwritten.
#
# Usage:
#   libs/srbench/refresh.sh            # refresh at the current pin
#   libs/srbench/refresh.sh <sha>      # refresh at a specific commit
#
# Exits non-zero if upstream grew a new intra-srbench import the slice can't
# satisfy on its own. Fix by vendoring the missing module or pinning older.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="$SCRIPT_DIR/src/srbench/srbench_upstream"
PIN_FILE="$UPSTREAM_DIR/PIN"

current_pin() { sed -n 's/^pinned: //p' "$PIN_FILE"; }

NEW_SHA="${1:-$(current_pin)}"
if [[ -z "$NEW_SHA" ]]; then
  echo "refresh.sh: no SHA given and $PIN_FILE has no 'pinned:' line" >&2
  exit 1
fi

# The closed scoring cluster, as paths under upstream experiment/.
UPSTREAM_PATHS=(
  experiment/evaluate_model.py
  experiment/assess_symbolic_model.py
  experiment/symbolic_utils.py
  experiment/read_file.py
  experiment/utils.py
  experiment/metrics/evaluation.py
  experiment/metrics/__init__.py
)

echo "==> sparse-cloning srbench at $NEW_SHA"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/cavalab/srbench "$TMP/srbench"
git -C "$TMP/srbench" checkout --quiet "$NEW_SHA"

echo "==> copying scoring files"
for p in "${UPSTREAM_PATHS[@]}"; do
  dest="$UPSTREAM_DIR/${p#experiment/}"
  mkdir -p "$(dirname "$dest")"
  cp "$TMP/srbench/$p" "$dest"
done

echo "==> updating PIN -> $NEW_SHA"
echo "pinned: $NEW_SHA" > "$PIN_FILE"

echo "==> checking import cluster is still closed"
# Top-level module names the vendored files may import.
ALLOWED=(
  # --- stdlib ---
  argparse ast collections copy datetime enum functools inspect importlib
  io itertools json math operator os pathlib pdb random re shutil signal
  subprocess sys tempfile time typing warnings
  # --- third-party (keep in sync with libs/srbench/pyproject.toml) ---
  joblib numpy pandas sklearn sympy yaml
)
# local modules provided by the vendored files themselves
for p in "${UPSTREAM_PATHS[@]}"; do
  rel="${p#experiment/}"
  if [[ "$rel" == */* ]]; then ALLOWED+=("${rel%%/*}"); else ALLOWED+=("${rel%.py}"); fi
done

violations=()
while IFS= read -r f; do
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    found=0
    for a in "${ALLOWED[@]}"; do [[ "$name" == "$a" ]] && { found=1; break; }; done
    (( found )) || violations+=("$(basename "$f"): imports '$name'")
  done < <(grep -hE '^[[:space:]]*(import|from)[[:space:]]' "$f" \
           | sed -E 's/^[[:space:]]*(import|from)[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*).*/\2/')
done < <(find "$UPSTREAM_DIR" -name '*.py')

if (( ${#violations[@]} )); then
  echo "refresh.sh: vendored slice is no longer self-contained." >&2
  echo "  Undeclared / non-vendored imports:" >&2
  printf '    %s\n' "${violations[@]}" >&2
  echo "  Either vendor the missing module or pin an older commit." >&2
  exit 1
fi

echo "==> done. Run 'git diff libs/srbench' to review."
