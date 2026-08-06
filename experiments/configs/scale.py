# Scaling sweep: same config, different pool sizes.
# Tests whether metrics plateau with more data or keep improving.
# 50k steps to ensure convergence at each scale.

n_trials = 5

base = {
    "seed": 42,
    "tag": "scale",
    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "bf16": True,
    "d_model": 512, "n_heads": 8, "d_ff": 2048,
    "n_enc_layers": 3, "n_dec_layers": 6, "dropout": 0.1,
    "n_steps": 50000, "batch_size": 64, "lr": 3e-4,
    "weight_decay": 0.01, "warmup_steps": 200,
    "lambda_": 0.01,
    "skeleton_mode": False,
    "canonicalize": True,
    "val_every": 500, "log_every": 50, "val_batch_size": 32,
    "n_test": 200,
}

POOLS = [
    "pools/scale_3200.pkl",
    "pools/scale_6400.pkl",
    "pools/canon_pool_large.pkl",
    "pools/scale_25600.pkl",
    "pools/scale_51200.pkl",
]

def search_space(trial):
    idx = trial.suggest_int("pool_idx", 0, len(POOLS) - 1)
    return {"pool_file": POOLS[idx]}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "priority-gpu",
        "account": "qiy18011", "time": "4:00:00",
        "mem": "32G", "gres": "gpu:1", "cpus-per-task": 8,
        "constraint": "a100",
        "exclude": "gpu14,gpu22,gpu24,gpu26,gtx18",
    },
}
