# Jernerics friction log — symlab overnight run (2026-07-31/08-01)

Findings to feed back into jernerics development, ordered by severity.

## CRITICAL — node heterogeneity silently destroys throughput
- QOS `qiy18011a100` implies A100, but jobs freely land on **A30** nodes
  (gpu46/48/49, feature `a30`) which run this workload at **~1.5 s/step vs
  ~0.13 s/step on RTX (gtx21) / A100 (gpu33-35)** — an **11x slowdown**.
- There is no warning. A 10k-step run that should take ~22 min instead projects
  to ~4.2 h, blocking a GPU slot. I lost ~20 min diagnosing this and had to
  cancel+resubmit.
- Workaround applied: added `exclude=gtx18,gpu46,gpu48,gpu49` to every config's
  `backend_overrides`.
- **Ask:** either make the `a100` QOS actually constrain to A100 nodes, or have
  jernerics surface the resolved GPU model / warn when a job lands on a
  markedly slower node.

## HIGH — `jernerics logs` reads the wrong directory
- `jernerics logs -b hpc <id>` looks in `.../jernerics/logs/<id>_*.out` but the
  real files are in `.../jernerics/symlab/logs/<id>_*.out` (with extra
  `checker_<id>.out` siblings). It returns "Log files not found" for every job.
- Workaround: read logs via `ssh ... cat /scratch/.../symlab/logs/<id>_1.out`.

## HIGH — missing-data-file triggers an infinite retry churn
- A `FileNotFoundError` on the pool file made optuna retry the trial repeatedly,
  each retry submitting a fresh GPU job that loads the 35M-param model then
  dies ~2 min later. No backoff, no give-up, no aggregate failure signal — jobs
  just churned, wasting GPU time and confusing `jernerics jobs`.
- **Ask:** fail fast on deterministic setup errors (missing input file) instead
  of retrying; cap trial retries.

## MEDIUM — data pools are outside the project sync
- `pools/` (and `results/`) are gitignored, so `jernerics run`'s project sync
  never transfers them. Transferring 28 MB pools over a slow link took ~15 min
  each (and an interrupted rsync left a truncated file that crashed the trial).
- Useful discovery: pools can be generated directly on the HPC **login node**
  inside the container (`apptainer exec ... python gen_pool_flex.py`), avoiding
  the sync entirely. Worth documenting as the recommended data workflow.
- **Ask:** jernerics should either sync declared data artifacts or clearly
  document that large data must be staged separately.

## LOW — CLI ergonomics
- `jernerics run` requires TWO positional args (`trial_file config_file`); the
  common shorthand shown in docs/tasks (`jernerics run config.py`) is wrong and
  fails with a confusing error.
- `jernerics jobs` / `logs` require `--backend` with no default — fine, but a
  project-config default would remove a repeated flag.
- `jernerics jobs` shows empty `study_name` and generic name "sbatch"; the
  auto-generated `symlab_config_<timestamp>` study names don't map back to the
  experiment `tag`. Hard to tell what is running without reading logs.
  Surfacing the config `tag` (or letting the caller name the study) would help.

## Context (not jernerics' fault, but affects workflow)
- Tracking server is unreachable from compute nodes (events dropped after
  retries). The `results/<tag>.json` + live `.out` SSH-reading workaround is
  solid; val metrics only survive in stdout.
