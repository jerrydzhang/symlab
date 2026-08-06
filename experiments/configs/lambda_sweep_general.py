# Lambda sweep on general-gpu

n_trials = 4

base = {
    "seed": 42,
    "tag": "lambda_gen",
    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "pool_file": "pools/canon_pool_large.pkl",
    "bf16": True,
    "d_model": 512, "n_heads": 8, "d_ff": 2048,
    "n_enc_layers": 3, "n_dec_layers": 6, "dropout": 0.1,
    "n_steps": 10000, "batch_size": 64, "lr": 3e-4,
    "weight_decay": 0.01, "warmup_steps": 200,
    "skeleton_mode": False,
    "canonicalize": True,
    "val_every": 500, "log_every": 50, "val_batch_size": 32,
    "n_test": 200,
}

def search_space(trial):
    lam = trial.suggest_categorical("lambda_", [0.0, 0.001, 0.01, 0.1])
    return {"lambda_": lam}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "general-gpu",
        "account": "qiy18011", "time": "2:00:00",
        "mem": "32G", "gres": "gpu:1", "cpus-per-task": 8,
        "constraint": "a100",
        "exclude": "gpu14,gpu22,gpu24,gpu26,gtx18",
    },
}
