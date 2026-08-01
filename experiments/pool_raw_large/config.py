# Pool-raw-large: CONTROL for pool_canon_large. Same pool size/seed (12800),
# NON-canonicalized. CE(pool_canon_large) vs CE(pool_raw_large) isolates the
# canonicalization effect at scale where memorization is impossible.

n_trials = 1

base = {
    "seed": 42,
    "tag": "pool_raw_large",

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "pool_file": "pools/raw_pool_large.pkl",
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
    "lambda_": 0.0,

    "val_every": 500,
    "log_every": 50,
    "val_batch_size": 32,
    "n_test": 200,
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
        "exclude": "gtx18",
    },
}
