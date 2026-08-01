n_trials = 1

base = {
    "seed": 42,
    "max_inputs": 2,
    "n_points": 100,
    "max_seq_len": 48,
    "d_model": 64,
    "n_heads": 4,
    "d_ff": 256,
    "n_enc_layers": 2,
    "n_dec_layers": 4,
    "n_steps": 200,
    "warmup_steps": 20,
    "lr": 3e-4,
    "batch_size": 16,
}

objective = lambda results: results["final_loss"]
direction = "minimize"


backend_overrides = {
    "hpc": {
        "partition": "general-gpu",
        "time": "0:10:00",
        "mem": "16G",
        "gres": "gpu:1",
        "account": "qiy18011",
        "qos": "general-gpu",
    },
}
