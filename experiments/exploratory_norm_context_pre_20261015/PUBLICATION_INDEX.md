# RSACA Norm-Placement Exploration: Published Evidence

This directory publishes the registered protocol and decision-relevant evidence for the exploratory comparison of `legacy_post_norm` (C0) and `context_pre_norm` (N1).

## Result

LEVIR-CC seed 2222 completed the registered stage-A screen with matched initialization and 10,000 updates per arm. The N1-minus-C0 deltas were BLEU-4 `-0.0244497721`, METEOR `-0.0072199856`, ROUGE-L `-0.0073314848`, CIDEr `-0.0079860177`, and SPICE `+0.0069635953`. The predeclared gate failed on CIDEr and BLEU-4, so stages B and C were not entered. This one-seed screen is exploratory and does not change historical admission or establish superiority to CARD.

## Contents

- `EXPLORATORY_PROTOCOL.json`, `PROTOCOL_AMENDMENTS.json`, and `status.json`: original registration, append-only operational amendments, and actual outcome.
- `configs/`: resolved C0/N1 configs; their only scientific difference is normalization mode.
- `stageA_results.json`: exact selected metrics, repeat metrics, initialization match, validity gate, and checkpoint identities.
- `reports/`: baseline audit, implementation/tests, resources, paired metrics, mechanism/annotation diagnostics, and next decision.
- `CODE_DIFF_POST_AMENDMENT.patch`: source patch used for this experiment, including the registered-interpreter launcher fix.
- `POST_RUN_ARTIFACT_MANIFEST.json`: checksums and identities of the full local artifact set.
- `COMMANDS.md`: executed command record.

## Artifact Boundary

The full experiment archive remains on the training server at `/root/autodl-tmp/v1_exploratory_norm_20261015T130107Z/`. Checkpoints, full training logs, per-checkpoint validation predictions/scores, repeated decode outputs, per-sample mechanism records, and new caption packets are not included in this V1 publication. The post-run manifest records their checksums, but does not make those files available in a clone. The reports preserve the complete validation aggregates and state the limitations of the omitted raw artifacts.

The reports were copied from the server archive. Relative references such as `../runs_card_env/` refer to that original archive layout and are not paths in a V1 clone.

Annotation provenance is reported as single-rater `rater01` with AI-assisted review. This is not independent double-blind annotation; no inter-rater kappa is claimed. New predictions have not received human or AI review, and historic labels were not transferred to them.
