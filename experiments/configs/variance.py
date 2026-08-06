# Multi-seed variance check: coupled vs skeleton with 3 seeds.
# Are the ~2% gaps within noise?

n_trials = 1

base = {
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
    "lambda_": 0.01,
    "canonicalize": True,
    "val_every": 500, "log_every": 50, "val_batch_size": 32,
    "n_test": 200,
}

search_space = {
    "seed": [42, 123, 456],
    "skeleton_mode": [False, True],
    "tag": ["coupled_s42", "coupled_s123", "coupled_s456",
            "skeleton_s42", "skeleton_s123", "skeleton_s456"],
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "priority-gpu", "qos": "qiy18011a100",
        "account": "qiy18011", "time": "2:00:00",
        "mem": "32G", "gres": "gpu:1", "cpus-per-task": 8,
        "constraint": "a100",
        "exclude": "gpu14,gpu24,gpu26,gtx18",
    },
}
