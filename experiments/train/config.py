n_trials = 1

base = {
    "seed": 42,
    "max_inputs": 10,
    "max_ops": 15,
    "max_seq_len": 48,
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_enc_layers": 3,
    "n_dec_layers": 6,
    "n_steps": 10000,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "lambda_": 0.01,
    "dropout": 0.1,
    "val_every": 500,
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
        "time": "2:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "constraint": "a100",
    },
}
