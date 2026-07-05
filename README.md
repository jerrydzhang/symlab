# symlab
symlab is a symbolic-regression research project.

## Repository layout
- `libs/symbolic` — shared substrate library (data, eval, constants, equivalence). Skeleton.
- `hypotheses/native-constants` — the native-constant experiment. Skeleton.

## Development environment
symlab uses Nix for the dev shell and uv for Python. The shell provides `python3` and `uv`, sets `LD_LIBRARY_PATH` on Linux for native extension loading, and on entry runs `uv sync` and activates `.venv` automatically — so dependencies are installed and the venv is active as soon as you enter the shell.

**With direnv (recommended):**

```
direnv allow
```

Run once; the shell then activates on every `cd` into the repo.

**Without direnv:**

```
nix develop
```

## Workspace
This is a uv workspace (root `package = false`) with two members, both using a `src/` layout and built with hatchling:
- `libs/symbolic` → package `symbolic` (depends on `sympy`)
- `hypotheses/native-constants` → package `native-constants` (depends on `symbolic` via the workspace)
