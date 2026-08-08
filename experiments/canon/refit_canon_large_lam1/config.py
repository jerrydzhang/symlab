# Refit on canon_large_lam1 (lambda=0.1 at scale). Completes the trio
# (refit_canon_large, refit_canon_large_lam01, this) to confirm the constant-fit
# ceiling is ~lambda-independent at scale.

n_trials = 1

base = {
    "seed": 42,
    "tag": "refit_canon_large_lam1",
    "source_tag": "canon_large_lam1",

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "pool_file": "pools/canon_pool_large.pkl",
    "bf16": True,

    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_enc_layers": 3,
    "n_dec_layers": 6,
    "dropout": 0.1,

    "val_batch_size": 32,
    "n_test": 100,
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "priority-gpu",
        "qos": "qiy18011a100",
        "account": "qiy18011",
        "time": "0:20:00",
        "mem": "16G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18,gpu46,gpu48,gpu49",
    },
}
