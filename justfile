default:
    @just --list

# refresh the vendored srbench scoring slice from upstream
# (default SHA: the current pin in libs/srbench/src/srbench/srbench_upstream/PIN)
update-srbench sha="":
    libs/srbench/refresh.sh {{sha}}

sync:
    uv sync --all-packages

test:
    uv run pytest

lint:
    uv run ruff check

marimo:
    marimo edit --watch --headless --host 0.0.0.0
