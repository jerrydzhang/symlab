# HPC GPU Hardware Survey — symlab

**Cluster:** hpc2.storrs.hpc.uconn.edu · **User:** jez21005 · **Account:** qiy18011
**Surveyed:** 2026-08-01 via `sinfo` / `scontrol` / `sacctmgr`

## TL;DR

- **The QOS name `qiy18011a100` is just a label — it does NOT guarantee A100.**
  Its only limits are `GrpTRES = cpu=128, gres/gpu=2` (group-level, shared across
  the whole `qiy18011` account): **max 2 concurrent GPUs and 128 CPUs total.**
  No wall-time cap, no per-user cap, no job-count cap. Jobs freely land on
  **A100, A30, or RTX** nodes depending on SLURM's pick.
- **This is the root cause of the 11× slowdown hit last night** — an `a30` node
  ran at ~1.5 s/step vs ~0.13 s/step on RTX/A100.
- **Fix: request `--constraint=a100`** to pin to A100 nodes (the gold standard).
  Available features include `a100`, `a30`, `l40`, `rtx`, `gtx`, so the
  constraint is honored cluster-wide.

## 1. QOS `qiy18011a100` — what it actually limits

```
Name            MaxWall   GrpTRES                  MaxTRESPU   GrpJobs   MaxJobsPU
qiy18011a100              cpu=128,gres/gpu=2
```

| limit | value | scope |
|---|---|---|
| Concurrent GPUs | **2** (`gres/gpu=2`) | group (all qiy18011 users share this) |
| Concurrent CPUs | **128** | group |
| Max wall time | none | — |
| Max jobs / submit | none | — |

**→ Max concurrent GPU jobs = 2.** This is the hard throughput ceiling,
independent of how many GPUs are free. Plan for 2-at-a-time queuing.

**Account associations** (`sacctmgr show assoc user=jez21005`):
```
qiy18011 | priority-gpu | qos=qiy18011a100,qiy18011gpu
qiy18011 | priority     | qos=qiy18011a100,qiy18011gpu
qiy18011 | general-gpu  | qos=general-gpu
qiy18011 | debug        | qos=general
...
```
We submit to **partition `priority-gpu`** under **QOS `qiy18011a100`**.

## 2. GPU node inventory — `priority-gpu` partition (where our jobs run)

All nodes: 64 CPU (Epyc) except RTX (32 CPU, Skylake); A100/A30 = ~515 GB RAM, RTX = 385 GB.

### A100 nodes — **target these** (`feature=a100`)
| node | GPUs | RAM (MB) | CPU | notes |
|---|---|---|---|---|
| gpu28 | 1 | 515000 | 64 | |
| gpu29 | 3 | 515000 | 64 | |
| gpu30 | 1 | 515000 | 64 | |
| gpu31 | 1 | 515000 | 64 | **state=unknown (DOWN)** — avoid |
| gpu32 | 1 | 515000 | 64 | |
| gpu33 | 1 | 515000 | 64 | |
| gpu34 | 1 | 515000 | 64 | |
| gpu35 | 3 | 515000 | 64 | |
| gpu36 | 3 | 515000 | 64 | |
| gpu37 | 3 | 515000 | 64 | |
| gpu38 | 3 | 515000 | 64 | |
| gpu39 | 3 | 515000 | 64 | |
| gpu40 | 3 | 515000 | 64 | |
| **total** | **27** | | | 13 nodes (12 usable; gpu31 down) |

### A30 nodes — **avoid (slow)** (`feature=a30`)
| node | GPUs | RAM (MB) | CPU | measured speed |
|---|---|---|---|---|
| gpu46 | 3 | 514900 | 64 | **~1.5 s/step** (11× slower) |
| gpu48 | 3 | 514900 | 64 | slow |
| gpu49 | 3 | 514900 | 64 | slow |
| **total** | **9** | | | 3 nodes |

### RTX nodes (`feature=rtx`, Skylake, 8 GPUs/node, 32 CPU)
| node | GPUs | RAM (MB) | state | notes |
|---|---|---|---|---|
| gtx12 | 8 | 385000 | mixed | |
| gtx13 | 8 | 385000 | allocated | |
| gtx14 | 8 | 385000 | **maint** | |
| gtx15 | 8 | 385000 | allocated | |
| gtx16 | 8 | 385000 | **drained** | |
| gtx17 | 8 | 385000 | mixed | |
| gtx18 | 8 | 385000 | mixed | **dead GPU** (CUDA fails to init) — exclude |
| gtx19 | 8 | 385000 | mixed | |
| gtx20 | 8 | 385000 | allocated | |
| gtx21 | 8 | 385000 | idle | measured **~0.13 s/step** (fast) |
| **total** | **80** | | | 10 nodes |

RTX nodes are **fast** (~0.13 s/step) but `constraint=a100` excludes them.
That's fine — A100 is strictly better (40 GB HBM2, strong bf16). Use A100.

### Other partitions (accessible by changing partition/QOS — not used tonight)
- **L40** (`feature=l40`, 4 GPUs/node, 48 GB): gpu41–45, partition `priority-l40` /
  `general-gpu`. Ada Lovelace — also fast, good bf16. Reachable via a different
  partition if ever needed.
- **A100 in general-gpu/debug**: gpu12–27 (many). Same GPU model, different
  partition; reachable if we ever need a partition other than priority-gpu.
- **Old GTX** (`feature=gtx`, gtx02–11): 2–3 GPUs, 20 CPU — legacy, skip.

## 3. SLURM flags to GUARANTEE A100

```
#SBATCH --partition=priority-gpu
#SBATCH --constraint=a100
```

`--constraint=a100` matches the `a100` node feature and rejects A30/RTX/L40.
This is strictly better than an `--exclude` list (which must be maintained as
nodes are added and still allowed slow A30 nodes outside the list).

Verified available features cluster-wide:
```
a100  a30  cpuonly  epyc128  epyc64  gpu  gtx  high-mem  l40  location=local  rtx  skylake
```

## 4. Nodes to exclude (if using exclude instead of constraint)

| node | reason |
|---|---|
| gpu31 | state `unknown` (down) |
| gtx14 | `maint` |
| gtx16 | `drained` |
| gtx18 | one of 8 GPUs is dead → CUDA init fails |
| gpu46, gpu48, gpu49 | A30 — 11× slower |
| gtx13, gtx15, gtx20 | currently `allocated` (wait, don't exclude) |

**Recommendation: use `constraint=a100` — it makes the exclude list unnecessary**
(gtx18/gtx14/gtx16 are RTX, automatically excluded; A30 nodes excluded).

## 5. Concurrency & capacity

- **Max concurrent GPU jobs: 2** (QOS `gres/gpu=2`, group-level).
- **Max concurrent CPUs: 128** (QOS group-level) — at `cpus-per-task=8` that's
  up to 16 concurrent CPU-slots, but the 2-GPU cap binds first.
- **A100 GPUs available in priority-gpu: ~26** (gpu31 down) across 12 nodes —
  far more than the 2 we can use concurrently, so A100 is effectively always
  available to us.
- Total GPUs in priority-gpu across all types: ~116 — capacity is not the
  bottleneck; the **2-GPU QOS cap** is.

## 6. Recommended `backend_overrides` template

```python
backend_overrides = {
    "hpc": {
        "partition": "priority-gpu",
        "qos": "qiy18011a100",
        "account": "qiy18011",
        "constraint": "a100",        # <-- guarantees A100; replaces exclude list
        "time": "2:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
    },
}
```

Notes:
- `constraint=a100` obsoletes the `exclude=gtx18,gpu46,gpu48,gpu49` used last
  night — cleaner and robust to new slow nodes being added.
- If A100 is ever saturated (unlikely, given ~26 free), temporarily add `,rtx`
  via SLURM OR-syntax `--constraint="[a100|rtx]"` to also allow the fast RTX
  nodes. Avoid `a30` and `l40` (l40 is on a different partition anyway).
- No need to set `mem` above 32 G for these runs (35 M-param model + 12800-sample
  pool fits comfortably; nodes have 385–515 GB).

## 7. Caveats observed

- **Node states change.** Re-run `sinfo -p priority-gpu -N -l` before long
  submission bursts to spot newly-drained/maint nodes. `constraint=a100` is
  immune to most state churn (SLURM won't schedule on down nodes regardless).
- **Group QOS is shared.** The 2-GPU cap is shared with other qiy18011 users —
  if someone else is running, you may get <2. `squeue -u jez21005` shows only
  yours; check cluster-wide GPU pressure with `sinfo -p priority-gpu -o "%n %G %C %t"`.
