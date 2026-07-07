# symlab
symlab is a symbolic-regression research project.

## Repository layout
- `libs/symbolic` — shared substrate library.
- `experiments/*` — controlled experiments and baselines; library consumers.

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
