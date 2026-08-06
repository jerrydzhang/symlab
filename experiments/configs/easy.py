# Easy: lowest-ambiguity distribution (1 op, 1 var). The model should ace this.
# If it does, ambiguity scales with problem complexity. If not, deeper issue.

n_trials = 1

base = {
    "seed": 42,
    "tag": "easy",

    "opset": "default",
    "max_inputs": 1,
    "max_ops": 1,
    "num_vars": [1, 1],
    "max_seq_len": 32,

    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_enc_layers": 3,
    "n_dec_layers": 6,
    "dropout": 0.1,

    "n_steps": 5000,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "lambda_": 0.0,

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
        "account": "qiy18011",
        "time": "1:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18",
        "constraint": "a100",
    },
}
