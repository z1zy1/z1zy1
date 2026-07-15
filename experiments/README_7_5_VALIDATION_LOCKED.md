# 7.5 validation-locked follow-up

This flow fixes checkpoint selection before any new test evaluation. Known test-set results are used only to form hypotheses; they are never passed to the 7.5 selector.

- `levir_mci_card_mask_semantic` step 5000 remains the current all-metric-leading LEVIR-MCI result. The later `lr004/step70` semantic trade-off must not silently replace it.
- SECOND-CC `second_cc_card_semantic_crossattn` favors BLEU/CIDEr, while `pd08` favors SPICE. The new gamma-limited candidates test a conservative residual.
- LEVIR-CC protects Bleu_1/2/3/4, METEOR, ROUGE_L, and CIDEr against the CARD validation baseline before maximizing SPICE.

## Candidate matrix

- LEVIR-CC: caption-only 30-step fine-tuning from a validation-locked old SGC checkpoint, `lr=2e-6`, save/eval every 10 steps, normalized content-word weights `1.03` and `1.05`.
- LEVIR-MCI: the same two caption-only candidates from the validation-locked `levir_mci_card_mask_semantic` checkpoint. The unmodified source remains in the selection pool as the safe fallback.
- SECOND-CC: 100-step fine-tuning from validation-locked `pd08`, `lr=2e-6`, save/eval every 20 steps, `partial_detach=0.8`, `lsem=0`, with semantic residual bounds `gamma_max=0.05` and `0.10`. The pool also contains unmodified `pd08`, `second_cc_card_semantic_crossattn`, and the CARD RGB baseline.

Source checkpoints are resolved only from an explicit `*_SOURCE_CHECKPOINT` environment variable or an existing `best_checkpoint`/`best_snapshot` selection artifact. No step number is hard-coded from test results.

## Required environment

```bash
export LEVIR_CC_ROOT=/path/to/Levir-CC
export LEVIR_MCI_ROOT=/path/to/LEVIR-MCI-dataset
export SECOND_CC_ROOT=/path/to/SECOND-CC-AUG
export LEVIR_CC_BASELINE_VAL_METRICS=/path/to/levir_cc_card_validation_metrics.json
export LEVIR_MCI_BASELINE_VAL_METRICS=/path/to/levir_mci_card_validation_metrics.json
export SECOND_CC_BASELINE_VAL_METRICS=/path/to/second_cc_card_validation_metrics.json
```

Each baseline file must contain validation metrics at the top level or under `metrics`, `selected_val_metrics`, or `selected_metrics`. It must contain SPICE and all protected metrics. `val_baseline_pareto` never falls back to the paper summary, built-in test thresholds, nearest checkpoints, or relaxed constraints.

Experiment directory names can be overridden when remote names differ:

```bash
export LEVIR_CC_SOURCE_EXP=sgc_card_lm003_ls005_pd05_rw02_warmup
export LEVIR_CC_CARD_BASELINE_EXP=card_levir_cc_baseline
export LEVIR_MCI_SOURCE_EXP=levir_mci_card_mask_semantic
export LEVIR_MCI_CARD_BASELINE_EXP=levir_mci_card_baseline
export SECOND_CC_SOURCE_EXP=second_cc_crossattn_pd08_lsem0000
export SECOND_CC_TRADITIONAL_EXP=second_cc_card_semantic_crossattn
export SECOND_CC_CARD_BASELINE_EXP=second_cc_card_rgb_baseline
```

## Execution protocol

Local machines should run only syntax, unit, compile, and dry-run checks:

```bash
bash scripts/run_7_5_followup_train.sh --dry_run
bash scripts/run_7_5_followup_reselect.sh --dry_run
bash scripts/run_7_5_followup_all.sh --dry_run
bash scripts/run_7_5_followup_test_locked.sh --dry_run
```

Run full training remotely in tmux. The default all-flow trains candidates and performs validation-only selection, then stops:

```bash
bash scripts/run_7_5_followup_all.sh
```

It writes `experiments/7_5_locked_manifest.json`. Only after reviewing and freezing that manifest, run the test set once for the single locked checkpoint per dataset:

```bash
bash scripts/run_7_5_followup_test_locked.sh
```

The locked test restores the selected source experiment's resolved model/data configuration, including `model.semantic_fusion_gamma_max`; content-word weighting is training-only. It does not test every candidate. Existing results are skipped unless `--force` is supplied. Dedicated 7.5 failure logs are append-only, so older user logs are not overwritten.
