# SECOND-CC optimized transfer flow

This flow refines the validated MCI transfer recipe without using test metrics
for checkpoint or learning-rate selection.

## Experiment matrix

- MCI initialization, semantic cross-attention, partial detach `0.5`.
- Target learning rates: `1.5e-4`, `2.0e-4`, `2.5e-4`.
- Seeds: `1111`, `2222`, `3333`.
- Loaded parameters use `0.1` times the target learning rate; parameters that
  were absent or shape-incompatible in the source checkpoint use the full rate.
- 500-step linear warmup followed by cosine decay to 5% of the initial rate.
- The existing learnable semantic residual gate starts at `0.01` and is clamped
  to `[-0.5, 0.5]`.

Each run saves and evaluates every 1000 steps. Selection requires three
contiguous validation checkpoints, separated by exactly 1000 steps, where all
eight caption metrics strictly exceed the audited CARD validation baseline.

## Commands

First, lock and test the existing `pd=0.5`, `lr=2e-4` three-seed winner:

```bash
bash scripts/run_second_cc_current_locked_test.sh
```

This writes `second_cc_current_mci_test_summary.json`. If it passes the final
acceptance rule, the refinement matrix is optional. If it fails, run the
optimized flow below.

Inspect the complete workflow without writing experiment outputs:

```bash
bash scripts/run_second_cc_optimized_full_flow.sh --dry-run
```

Run the complete matrix, validation lock, locked tests, and summary:

```bash
bash scripts/run_second_cc_optimized_full_flow.sh --stage all
```

Run or resume individual stages:

```bash
bash scripts/run_second_cc_optimized_full_flow.sh --stage train
bash scripts/run_second_cc_optimized_full_flow.sh --stage select
bash scripts/run_second_cc_optimized_full_flow.sh --stage lock
bash scripts/run_second_cc_optimized_full_flow.sh --stage test
bash scripts/run_second_cc_optimized_full_flow.sh --stage summary
```

After reviewing the locked-test summary, generate a snapshot deletion plan:

```bash
bash scripts/run_second_cc_optimized_full_flow.sh --stage prune
```

Apply that exact policy, retaining only the three manifest-locked snapshots:

```bash
bash scripts/run_second_cc_optimized_full_flow.sh --stage prune --confirm-prune
```

Pruning is deliberately excluded from `--stage all`.

Training and selection stages accept `--only_spec` and `--only_seed`. Incomplete
non-empty run directories are refused so a partial run cannot be mistaken for a
completed experiment.

## Outputs

- Runs: `experiments/second_cc_optimized_runs/`
- Immutable validation lock: `experiments/second_cc_optimized_lock.json`
- Locked test summary: `experiments/second_cc_optimized_test_summary.json`
- Per-seed CSV: `experiments/second_cc_optimized_test_summary.csv`
- Prune plan: `experiments/second_cc_optimized_prune_plan.json`

The final acceptance rule requires the three-seed mean to beat the locked CARD
test baseline on all eight metrics and at least two individual seeds to beat it
on all eight metrics. Change and no-change group means are included for failure
analysis but never participate in model selection.
