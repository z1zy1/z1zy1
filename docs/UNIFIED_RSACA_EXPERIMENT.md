# Unified CARD + RSACA experiment

This experiment uses one architecture for LEVIR-CC, LEVIR-MCI, and SECOND-CC:

* `model.semantic_input_mode=cross_attention`
* residual semantic fusion with `gamma_init=0.01`, `gamma_max=0.5`
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
