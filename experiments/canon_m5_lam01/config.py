# canon_m5_lam01: canonicalized, max_ops=5, 12800 samples, lambda=0.01.
# Tests whether the lambda=0.01 constant fix transfers to higher complexity.
# Reuses pools/canon_pool_m5.pkl.

n_trials = 1

base = {
    "seed": 42,
    "tag": "canon_m5_lam01",

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 5,
    "max_seq_len": 32,
    "pool_file": "pools/canon_pool_m5.pkl",
    "bf16": True,

    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_enc_layers": 3,
    "n_dec_layers": 6,
    "dropout": 0.1,

    "n_steps": 10000,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "lambda_": 0.01,

    "val_every": 500,
    "log_every": 50,
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
        "time": "2:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18,gpu46,gpu48,gpu49",
    },
}
