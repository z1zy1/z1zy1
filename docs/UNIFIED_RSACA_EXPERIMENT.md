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

## Strict Paired CARD/RSACA Matrix

The historical unified runner has an RSACA arm only and compares it with a
single CARD audit checkpoint. The paper-facing matrix must instead use
`scripts/run_paired_card_rsaca_matrix.sh`, which trains CARD and the
whole-adapter RSACA candidate from scratch for the same three datasets and
three seeds. It writes `paired_protocol_lock.json` before training and rejects
later stages if the Git commit, tracked source digest, Python, CUDA visibility,
worker count, dataset list, or seed list changes. The only permitted arm
difference is the RSACA semantic-fusion path.

Run all 18 jobs only after the candidate is frozen:

```bash
PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage all
```

For a recoverable staged execution, run `preflight`, `train`, `select`,
`test`, then `summary` with the same complete `PAIR_ROOT`, environment, and
default seeds. `test` is immutable; `--force-select` only refreshes a selection
before a test exists; `--reset-incomplete` archives rather than overwrites an
interrupted training directory. The summary requires all 18 artifacts, verifies
that each test used its validation-selected checkpoint, compares resolved
configs outside an explicit semantic-arm allowlist, and reports paired deltas.

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

## Reliability-gated sparse V1 candidate

`scripts/run_reliability_sparse_rsaca_v1.sh` is a separate, full-process
candidate runner. It preserves the locked RSACA training schedule, semantic
sources, normalization mode, residual scale, and partial detach setting, while
adding two shared mechanisms:

* only changed semantic locations are available as cross-attention K/V tokens;
  a learned fallback token handles samples without a valid changed location;
* a learned continuous reliability gate uses visual change summary, sparse
  semantic summary, and semantic change coverage to scale the residual.

For paired inputs, a change is `before != after`; for diff-only inputs it is a
non-zero diff class. This is a single rule across all datasets, not a
dataset-specific module. The full scratch matrix is:

```bash
bash scripts/run_reliability_sparse_rsaca_v1.sh --stage all
```

Start with `--stage preflight` or one validation-only training run; do not run
the locked test stage while choosing V1 hyperparameters.

## Whole-adapter gate follow-up

After the first sparse V1 run, the follow-up
`scripts/run_reliability_sparse_rsaca_v1_whole_gate.sh` applies the learned
reliability coefficient to the complete RSACA adapter output rather than only
the semantic residual. This makes an unreliable sample able to return to the
original CARD visual representation. It also appends an input-conditioned
global semantic token to the changed-token K/V set, preserving contextual
information discarded by sparse masking.

Its outputs are isolated under
`experiments/reliability_sparse_rsaca_v1_whole_gate`; the candidate still must
be screened on validation metrics before any locked test stage.

## Pre-norm changed-token follow-up

The next isolated candidate is
`scripts/run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh`. It keeps
the whole-adapter reliability gate, switches to `context_pre_norm`, limits
`gamma_max` to `0.1`, initializes the reliability gate conservatively, and
uses a global token pooled from changed semantic locations only. The runner
also records a sampled semantic-input audit before training and passes the
explicit validation selection metric to the selector.

The noise-aware follow-up also supports an optional
`data.semantic_diff_confidence_root` containing source-model probability maps.
When present, changed tokens and the `changed_mean` token are confidence
weighted. The gate receives semantic quality and visual-semantic agreement,
and a capped visual fallback remains available when a pseudo-mask misses a
change. `context_pre_norm` fusion can be linearly warmed up with
`model.semantic_fusion_warmup_steps`. The recommended candidate uses
`paper_balanced_no_spice` for validation checkpoint selection; SPICE is still
reported at test time but does not select the checkpoint.

For LEVIR-CC pseudo-mask regeneration, use
`scripts/generate_levir_ensemble_masks_direct.sh` with official LEVIR
checkpoints from ChangeFormerV6 and BIT. The disk-constrained entrypoint
validates checkpoint, dependency, and CUDA availability, deletes the active
`pseudo_*` trees before inference, writes 8-bit probability maps, then
produces a consensus `pseudo_masks`, agreement-aware `pseudo_confidence`, and
`pseudo_uncertainty` tree. It validates one output per input image before
installation. The resulting ensemble has completed its locked seed matrix but
remains an exploratory semantic-input change because that matrix did not pass
the CARD-over-baseline acceptance criterion.

Its default output root is
`experiments/reliability_sparse_rsaca_v1_prenorm_changed_global`.

## Detached-gate V2 candidate

`scripts/run_reliability_sparse_rsaca_v2_detached_gate.sh` is the LEVIR-CC
stability candidate derived from the strongest pre-norm changed-token RSACA
core. It keeps sparse changed-location K/V, `changed_mean`, the whole-adapter
continuous gate, `context_pre_norm`, `gamma_init=0.01`, `gamma_max=0.1`, gate
bias `-2.5`, and partial detach `0.5`. It adds only
`model.semantic_fusion_detach_reliability_inputs=true`: visual and semantic
summaries are detached before entering the reliability MLP, while the adapter
and the MLP remain trainable.

V2 explicitly disables confidence maps, visual-consistency gating, visual
fallback, and fusion warmup. It uses the active LEVIR-CC `pseudo_masks`; this
is a new model candidate, not evidence that the current masks are better.

Screen the difficult seeds on validation only first. Do not run `test` or
`all` during this decision:

```bash
RUN_ROOT=/root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_detached_gate NUM_WORKERS=8 SEEDS="3333 1111" bash scripts/run_reliability_sparse_rsaca_v2_detached_gate.sh --stage preflight --dataset levir_cc
RUN_ROOT=/root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_detached_gate NUM_WORKERS=8 SEEDS="3333 1111" bash scripts/run_reliability_sparse_rsaca_v2_detached_gate.sh --stage train --dataset levir_cc
RUN_ROOT=/root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_detached_gate NUM_WORKERS=8 SEEDS="3333 1111" bash scripts/run_reliability_sparse_rsaca_v2_detached_gate.sh --stage select --dataset levir_cc
```

Only after pre-registering V2 as the sole candidate and completing its paired
CARD control should the locked three-seed process run once:

```bash
RUN_ROOT=/root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_detached_gate NUM_WORKERS=8 SEEDS="1111 2222 3333" bash scripts/run_reliability_sparse_rsaca_v2_detached_gate.sh --stage all --dataset levir_cc
```

V2 requires that the selected `PYTHON` interpreter reports CUDA available
before it creates a training directory. If a run was interrupted before its
final checkpoint, preserve it for audit and restart only that run with
`--reset-incomplete`; selection rejects empty metric rows and non-file
snapshots, so interrupted runs cannot become test candidates.
Use `--force-select` only to refresh a pre-test validation selection after a
selector repair; it refuses to modify any run that already has a locked test.

The detached-gate change must be compared with the paired control below before
any V2 test. It uses the same active masks, CUDA environment, worker count,
seeds, schedule, and validation selector, changing only
`semantic_fusion_detach_reliability_inputs` from `true` to `false`:

```bash
RUN_ROOT=/root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_paired_control NUM_WORKERS=8 OMP_NUM_THREADS=1 SEEDS="3333 1111" PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_reliability_sparse_rsaca_v2_paired_control.sh --stage train --dataset levir_cc
RUN_ROOT=/root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_paired_control NUM_WORKERS=8 OMP_NUM_THREADS=1 SEEDS="3333 1111" PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_reliability_sparse_rsaca_v2_paired_control.sh --stage select --dataset levir_cc
/root/miniconda3/envs/card/bin/python scripts/compare_rsaca_validation_pairs.py --candidate-root /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_detached_gate --control-root /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_paired_control --seeds "3333 1111" --output /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v2_paired_comparison.json
```

Do not run `test` for either arm unless both V2 seeds consistently improve the
paired control on validation. A successful screen must then add seed `2222`
and a same-environment CARD control before a one-time locked test.

The completed new-mask run is isolated under
`experiments/reliability_sparse_rsaca_v1_new_masks_20260821`. Its three-dataset
three-seed locked summary does not meet the CARD-over-baseline acceptance
criterion; this remains true when SPICE is excluded from the acceptance
comparison. See
`docs/EXPERIMENT_CONFIG_RESULTS.md` for the resolved configuration, per-seed
checkpoints, and complete result interpretation.

To isolate the effect of the active new LEVIR-CC masks from the later
confidence and visual-gating additions, use
`scripts/run_levir_cc_new_masks_matched_control.sh`. It keeps the dataset-scoped
confidence isolation, but deliberately disables confidence, visual-consistency
gating, visual fallback, and fusion warmup, and restores `paper_balanced`
validation selection. Its results are written to
`experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control`.
The unified runner propagates `--dataset` and `--seed` to its summary stage, so
the complete LEVIR-CC matched-control matrix can be run independently.
The matched-control entrypoint requires CUDA and defaults to `NUM_WORKERS=0`
for constrained containers; set a larger explicit worker count only on a host
with sufficient cgroup memory.
