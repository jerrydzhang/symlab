# Controlled eval at max_ops=5: does the constant-fit conclusion (canon+fit ~
# raw+fit, fit() dominant) hold at higher complexity?
# Loads canon_m5, raw_m5, canon_m5_lam01 checkpoints on a shared m5 test set.

n_trials = 1

base = {
    "seed": 42,
    "tag": "controlled_eval_m5",
    "source_tags": ["canon_m5", "raw_m5", "canon_m5_lam01"],

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 5,
    "max_seq_len": 32,
    "test_pool_file": "pools/shared_test_m5.pkl",
    "bf16": True,

    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_enc_layers": 3,
    "n_dec_layers": 6,
    "dropout": 0.1,

    "val_batch_size": 32,
    "n_test": 100,
}

objective = lambda results: results["final_loss"]
direction = "minimize"

backend_overrides = {
    "hpc": {
        "partition": "priority-gpu",
        "qos": "qiy18011a100",
        "account": "qiy18011",
        "time": "0:30:00",
        "mem": "16G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18,gpu46,gpu48,gpu49",
    },
}
