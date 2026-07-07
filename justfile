# symlab task runner. Recipes run from the repo root.
# Run `just` with no args to list recipes.

# default: list available recipes
default:
    @just --list

# refresh the vendored srbench scoring slice from upstream
# (default SHA: the current pin in libs/srbench/src/srbench/srbench_upstream/PIN)
update-srbench sha="":
    libs/srbench/refresh.sh {{sha}}

# create / update the virtualenv across all workspace packages
sync:
    uv sync --all-packages

# run the test suite
test:
    uv run pytest

# lint with ruff
lint:
    uv run ruff check
