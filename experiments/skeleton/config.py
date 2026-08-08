# Skeleton condition: model trains with all constants zeroed to 1.0 (λ=0.01).
# The NUM token is still emitted, but its value is always 1.0 — the model
# learns the same structure vocabulary without ever seeing constant magnitudes.
# At eval, post-hoc fit() refines constants identically to the coupled model,
# so any difference in func_equiv_fit is attributable purely to structure.
# Control = experiments/coupled (identical except skeleton_mode=False).

n_trials = 1

base = {
    "seed": 42,
    "tag": "skeleton",
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
    "skeleton_mode": True,  # constants zeroed to 1.0
    "canonicalize": True,  # match the pool's canonicalization
    "val_every": 500, "log_every": 50, "val_batch_size": 32,
    "n_test": 100,
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "general-gpu", "qos": "general-gpu",
        "account": "qiy18011", "time": "4:00:00",
        "mem": "32G", "gres": "gpu:1", "cpus-per-task": 8,
        "constraint": "a100",
        "exclude": "gpu24,gtx18",
    },
}
