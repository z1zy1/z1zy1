# Commands and Execution Record

Source preparation:

```bash
git worktree add -b exploratory/norm_context_pre_20261015 \
  /root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/worktree \
  99d1659acf37c6a1099be9fbe10948886b3afb9f
```

Formal stage-A commands executed after resource recovery (the first C0 attempt used the base interpreter and is retained as invalid; the corrected executions explicitly use the registered `card` interpreter):

```bash
cd /root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/worktree
PYTHON=/root/miniconda3/envs/card/bin/python \
EXPLORATORY_OUTPUT_ROOT=/root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/runs_card_env \
scripts/run_exploratory_norm_pair.sh C0 levir_cc 2222 train
PYTHON=/root/miniconda3/envs/card/bin/python \
EXPLORATORY_OUTPUT_ROOT=/root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/runs_card_env \
scripts/run_exploratory_norm_pair.sh N1 levir_cc 2222 train
PYTHON=/root/miniconda3/envs/card/bin/python EXPLORATORY_OUTPUT_ROOT=/root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/runs_card_env \
scripts/run_exploratory_norm_pair.sh C0 levir_cc 2222 select
PYTHON=/root/miniconda3/envs/card/bin/python EXPLORATORY_OUTPUT_ROOT=/root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/runs_card_env \
scripts/run_exploratory_norm_pair.sh N1 levir_cc 2222 select
```

The full matrix and conditional stage rules remain frozen in `EXPLORATORY_PROTOCOL.json`. Stage A completed. Its gate failed, so stage B and C were not run. Full logs, selected checkpoints, validation predictions, scores, and repeated-decode outputs are under `runs_card_env/` and `logs/`. The invalid initial C0 attempt remains under `runs/C0/` with its separate log.

The paired initialization audit used:

```bash
python scripts/compare_exploratory_initialization.py \
  C0_RUN/initial_parameter_summary.json \
  N1_RUN/initial_parameter_summary.json \
  --output paired_initialization.json
```

Selected-checkpoint repeated validation used the saved `resolved_config.yaml` and `test_card_spot.py --split val`, followed by `scripts/sgc_card_metrics.py` against the unchanged CC references. C0 used `norm_C0_levir_cc_seed2222_checkpoint_9000.pt`; N1 used `norm_N1_levir_cc_seed2222_checkpoint_10000.pt`. Their commands and results are preserved in `logs/stageA_*_repeat*` and each run's `repeat_decode/` directory. Mechanism diagnostics used `scripts/diagnose_semantic_controls.py` with a fixed 64-ID sample and seed 20261018; commands and records are in `logs/mechanism_*.log` and `reports/diagnostics/`.

Test logs are `logs/test_exploratory_norm_protocol.log`, `logs/test_wcsg_smoke.log`, and `logs/test_unified_rsaca_manual.log`. No formal test or freeze command is part of this exploratory protocol.
