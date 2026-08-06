# Coupled condition: model trains with real constant values (λ=0.01).
# At eval, post-hoc fit() refines constants — so this condition tests whether
# seeing real constants during training improves STRUCTURE prediction.
# Control = experiments/skeleton (identical except skeleton_mode=True).

n_trials = 1

base = {
    "seed": 42,
    "tag": "coupled",
    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "pool_file": "pools/canon_pool_large.pkl",  # reuse existing 12800-sample pool
    "bf16": True,
    "d_model": 512, "n_heads": 8, "d_ff": 2048,
    "n_enc_layers": 3, "n_dec_layers": 6, "dropout": 0.1,
    "n_steps": 10000, "batch_size": 64, "lr": 3e-4,
    "weight_decay": 0.01, "warmup_steps": 200,
    "lambda_": 0.01,  # MSE head active
    "skeleton_mode": False,  # real constants
    "canonicalize": True,  # match the pool's canonicalization
    "val_every": 500, "log_every": 50, "val_batch_size": 32,
    "n_test": 100,
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "priority-gpu",
        "account": "qiy18011", "time": "2:00:00",
        "mem": "32G", "gres": "gpu:1", "cpus-per-task": 8,
        "constraint": "a100",
        "exclude": "gpu14,gpu24,gpu26,gtx18",
    },
}
