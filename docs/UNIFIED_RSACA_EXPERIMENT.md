# Unified CARD + RSACA experiment

This experiment uses one architecture for LEVIR-CC, LEVIR-MCI, and SECOND-CC:

* `model.semantic_input_mode=cross_attention`
* residual semantic fusion with `gamma_init=0.01`, `gamma_max=0.5`
* locked runs use the legacy post-residual LayerNorm for exact reproducibility
* `use_semantic_partial_detach=true`, `semantic_detach_ratio=0.5`
* no auxiliary mask loss, semantic caption loss, hard gate, or feature reweighting
* seven shared semantic classes plus the unchanged token used by the diff encoding

SECOND-CC supplies paired semantic maps (`sem/A`, `sem/B`). LEVIR-CC supplies
binary pseudo change maps, mapped to shared class 6; LEVIR-MCI supplies its
multiclass `images/<split>/label` maps. These are intentionally marked as
`semantic_diff_only` in the resolved config, so the report distinguishes paired
semantic input from difference-only input.

Run the complete matrix after feature extraction:

```bash
bash scripts/run_unified_rsaca_experiments.sh --stage all
```

Useful stages are `preflight`, `train`, `select`, `test`, and `summary`. Use
`--dataset`, `--seed`, and `--dry-run` to narrow or inspect a run. Checkpoint
selection reads validation metrics only; the summary verifies that each locked
test result uses the selected checkpoint and compares seed means to
`experiments/card_baseline_test_summary.json`.

The locked runs revealed that `legacy_post_norm` applies LayerNorm even when
the residual gate is zero. An exploratory, validation-only ablation can test
the identity-preserving `context_pre_norm` variant without overwriting locked
artifacts:

```bash
RUN_ROOT=./experiments/unified_rsaca_identity_ablation \
SEMANTIC_FUSION_NORM_MODE=context_pre_norm \
bash scripts/run_unified_rsaca_experiments.sh --stage train --dataset levir_cc --seed 1111
```

This candidate is not part of the locked paper claim until it passes validation
screening and the complete scratch multi-seed protocol.
