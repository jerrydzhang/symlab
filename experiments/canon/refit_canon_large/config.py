# Refit analysis on canon_large: load its checkpoint, regenerate on the
# held-out test split, and compare R^2 before/after Expression.fit() (constant
# refinement). Quantifies the "right-structure / wrong-constants" ceiling.
# source_tag must point at a trained model whose results/<tag>.pt exists.

n_trials = 1

base = {
    "seed": 42,
    "tag": "refit_canon_large",
    "source_tag": "canon_large",

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "pool_file": "pools/canon_pool_large.pkl",
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
        "time": "0:20:00",
        "mem": "16G",
        "gres": "gpu:1",
        "cpus-per-task": 8,
        "exclude": "gtx18,gpu46,gpu48,gpu49",
    },
}
