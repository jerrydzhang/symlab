"""
Diagnostic run: simple distribution to isolate architecture from problem difficulty.

If this works (loss decreases, valid_rate improves), the architecture is sound
and the full run failed because the problem was too hard.

If this plateaus too, there's a structural bug to find.
"""

n_trials = 1

base = {
    "seed": 42,

    # SIMPLE distribution — isolate architecture from difficulty
    "opset": "default",          # 6 ops: add, sub, mul, div, sin, exp
    "max_inputs": 2,             # 2 variables only
    "max_ops": 3,                # shallow expressions
    "max_seq_len": 32,           # shorter sequences

    # Same model config as the full run
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_enc_layers": 3,
    "n_dec_layers": 6,
    "dropout": 0.1,

    # Training
    "n_steps": 5000,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "lambda_": 0.01,

    "val_every": 250,
    "log_every": 50,
    "val_batch_size": 32,
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "priority-gpu",
        "qos": "qiy18011a100",
        "account": "qiy18011",
        "time": "1:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
    },
}
