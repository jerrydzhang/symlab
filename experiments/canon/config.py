# Canon: diagnostic config (lambda=0) WITH sympy canonicalization, fp32.
# Tests whether removing expression ambiguity breaks the CE plateau.
# fp32 + max_const filter: sympy simplify inflates constants to ~1.5e4
# (raw data is bounded to ~100), which explodes activations via
# embedding*num_values -> NaN. Bound constants like the raw distribution.

n_trials = 1

base = {
    "seed": 42,
    "tag": "canon",

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "canonicalize": True,
    "bf16": False,
    "max_const": 100,

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
        "qos": "general-gpu",
        "account": "qiy18011",
        "time": "1:00:00",
        "mem": "32G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18",
        "constraint": "a100",
    },
}
