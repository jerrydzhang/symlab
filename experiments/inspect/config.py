# Inspect: diagnostic config with MSE head, then dump 100 generations.

n_trials = 1

base = {
    "seed": 42,
    "tag": "inspect",

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
    "lambda_": 1.0,  # MSE head ON — match the spec

    "val_every": 500,
    "log_every": 50,
    "val_batch_size": 32,
    "n_test": 100,
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "general-gpu",
        "qos": "general-gpu",
        "account": "qiy18011",
        "time": "1:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18",
        "constraint": "a100",  # GPU2 hardware fault -> CUDA fails to init
    },
}
