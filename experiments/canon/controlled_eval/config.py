# Controlled functional-recovery eval on a SHARED fresh test set.
# Loads canon_large, raw_large, canon_large_lam01 checkpoints and evaluates
# each on the SAME 100 raw-distribution samples, with and without fit().
# Removes the test-set confound from the canon-vs-raw funcEq comparison.

n_trials = 1

base = {
    "seed": 42,
    "tag": "controlled_eval",
    "source_tags": ["canon_large", "raw_large", "canon_large_lam01"],

    "opset": "default",
    "max_inputs": 2,
    "max_ops": 3,
    "max_seq_len": 32,
    "test_pool_file": "pools/shared_test.pkl",
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
