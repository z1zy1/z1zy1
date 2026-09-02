# Strict Paired CARD/RSACA Matrix

This protocol is the paper-facing `3 datasets x 2 models x 3 seeds` matrix.
It compares scratch-trained CARD with the fixed V1 whole-adapter RSACA model
on LEVIR-CC, LEVIR-MCI, and SECOND-CC.

The runner writes `paired_protocol_lock.json` before work starts. The lock
binds the Git commit, tracked source digest, source-status digest, Python path,
CUDA visibility, worker count, datasets, and seeds. A later invocation fails
if any bound common condition changes. CARD uses `semantic_input_mode=none`;
RSACA alone enables sparse changed-token K/V, the global token, and the
whole-adapter reliability gate. All remaining training settings are shared.

Run the matrix in one pass only after the candidate is frozen:

```bash
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage all
```

For staged recovery, keep the same environment and run:

```bash
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage preflight
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage train
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage test
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage summary
```

`test` evaluates each validation-selected checkpoint once and does not permit
replacement. `--force-select` is only available before a test result exists;
`--reset-incomplete` archives interrupted directories. The summary refuses any
missing arm, selection/test mismatch, or resolved-config difference outside
the semantic-arm allowlist. It reports per-seed paired deltas, their sample
standard deviations, and the pre-registered acceptance result.
