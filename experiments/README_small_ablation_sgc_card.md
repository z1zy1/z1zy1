# SGC-CARD Small Ablation

This small ablation checks whether the remaining test-set gap comes from feature reweight strength, semantic auxiliary loss strength, or a deeper BLEU/CIDEr vs. SPICE conflict.

## Experiments

| Experiment | Purpose | Key settings |
| --- | --- | --- |
| `lmask003_lsem005_pd05_warmup_no_reweight` | Test whether feature reweight hurts SPICE or balance | `lmask=0.003`, `lsem=0.005`, `partial_detach=0.5`, no feature reweight, aux warmup, semantic final ratio `0.50` |
| `lmask003_lsem005_pd05_rw01_warmup` | Test whether reweight is useful but `alpha=0.2` is too strong | `lmask=0.003`, `lsem=0.005`, `partial_detach=0.5`, `reweight_alpha=0.1`, aux warmup, semantic final ratio `0.50` |
| `lmask003_lsem003_pd05_rw01_warmup` | Test whether semantic loss `0.005` is still too strong | `lmask=0.003`, `lsem=0.003`, `partial_detach=0.5`, `reweight_alpha=0.1`, aux warmup, semantic final ratio `0.25` |

## One-Command Run

```bash
chmod +x scripts/train_lmask003_lsem005_pd05_warmup_no_reweight.sh
chmod +x scripts/train_lmask003_lsem005_pd05_rw01_warmup.sh
chmod +x scripts/train_lmask003_lsem003_pd05_rw01_warmup.sh
chmod +x scripts/run_small_ablation_sgc_card.sh

bash scripts/run_small_ablation_sgc_card.sh
```

The runner is resumable. If `experiments/<exp_name>/snapshots/` already contains checkpoints, training is skipped by default. Use `--overwrite` to recompute all stages.

## Run One Experiment

```bash
bash scripts/train_lmask003_lsem005_pd05_warmup_no_reweight.sh

bash scripts/eval_all_snapshots_sgc_card.sh \
  --exp_dir experiments/lmask003_lsem005_pd05_warmup_no_reweight

python scripts/select_best_snapshot_sgc_card_v2.py \
  --csv experiments/lmask003_lsem005_pd05_warmup_no_reweight/eval_snapshots.csv \
  --exp_dir experiments/lmask003_lsem005_pd05_warmup_no_reweight

bash scripts/test_top_snapshots_sgc_card.sh \
  --exp_dir experiments/lmask003_lsem005_pd05_warmup_no_reweight

python scripts/compare_snapshot_results.py \
  --exp_dir experiments/lmask003_lsem005_pd05_warmup_no_reweight
```

## Outputs

Per experiment:

```text
experiments/<exp_name>/train.log
experiments/<exp_name>/eval_snapshots.csv
experiments/<exp_name>/best_snapshot_v2.json
experiments/<exp_name>/best_balanced_v2.pth
experiments/<exp_name>/test_top_snapshots_summary.csv
experiments/<exp_name>/snapshot_compare_report.txt
experiments/<exp_name>/snapshot_compare_report.csv
```

Overall:

```text
experiments/small_ablation_sgc_card_summary.csv
experiments/small_ablation_sgc_card_report.txt
```

## How To Interpret

- Checkpoint selection issue: validation looks strong but `test_top_snapshots_summary.csv` shows a non-v2 or manually tested checkpoint is better than `best_balanced_v2`.
- Reweight too strong: `rw01` is more balanced than the previous `rw02` run, or `no_reweight` recovers SPICE without a large BLEU-4/CIDEr drop.
- Semantic loss too strong: `lsem003` improves balance over `rw01`, especially with fewer caption metric drops and a smaller SPICE gap.
- Model conflict remains: all three experiments still trade BLEU-4/CIDEr against SPICE, and no tested snapshot exceeds all test baselines.

Use `small_ablation_sgc_card_report.txt` for the final recommendation and next-step conclusion.
