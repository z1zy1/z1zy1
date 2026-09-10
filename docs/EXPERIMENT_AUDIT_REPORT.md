# Experiment Audit Report

Audit time: `2026-09-08 07:44:20 UTC`  
Code inspected: `64be6e134fc3ecf6c251333cecfa26712b9af84d`

This is a partial audit of locally accessible artifacts. It does not treat an experiment name, summary file, checkpoint, or test-output directory as proof that a complete comparable experiment exists.

## Evidence Coverage

- resolved training configurations: 165
- test-result records: 243
- selection-and-test records with matching checkpoint paths: 61
- missing/invalid evidence findings: 293

## Comparison Groups

| Group | Protocol | observed/expected | Verified main table |
|---|---|---:|---|
| strict_paired_whole_gate_v1 | strict_paired | 1/9 | no |
| unified_rsaca_vs_single_audit_card | exploratory_vs_single_audit_baseline | 9/9 | no |
| whole_gate_retry_vs_single_audit_card | exploratory_vs_single_audit_baseline | 9/9 | no |
| second_cc_mci_transfer | transfer_not_ablation | 18/N/A | no |
| second_cc_scratch_protocol_undetermined | protocol_undetermined | 3/N/A | no |

The strict paired matrix is incomplete: only 1 of 9 required seed-pairs (2/18 arms) currently have both validation selection and locked-test records. It must not be summarized as a completed `3 x 2 x 3` main experiment.

## Verified Evaluation Records

| Run | Evaluation | Checkpoint | Metric count | Scale |
|---|---|---|---:|---|
| run-52ae404f7b83697a | experiments/levir_mci_card_mask_semantic/test_change_result.json | /root/autodl-tmp/z1zy1/experiments/levir_mci_card_mask_semantic/snapshots/levir_mci_card_mask_semantic_checkpoint_5000.pt | 8 | raw_fraction_or_cider |
| run-52ae404f7b83697a | experiments/levir_mci_card_mask_semantic/test_no-change_result.json | /root/autodl-tmp/z1zy1/experiments/levir_mci_card_mask_semantic/snapshots/levir_mci_card_mask_semantic_checkpoint_5000.pt | 8 | raw_fraction_or_cider |
| run-52ae404f7b83697a | experiments/levir_mci_card_mask_semantic/test_overall_result.json | /root/autodl-tmp/z1zy1/experiments/levir_mci_card_mask_semantic/snapshots/levir_mci_card_mask_semantic_checkpoint_5000.pt | 8 | raw_fraction_or_cider |
| run-52ae404f7b83697a | experiments/levir_mci_card_mask_semantic/test_paper_best_result.json | /root/autodl-tmp/z1zy1/experiments/levir_mci_card_mask_semantic/snapshots/levir_mci_card_mask_semantic_checkpoint_5000.pt | 8 | raw_fraction_or_cider |
| run-23220e1bbbeaa63a | experiments/paired_card_rsaca_whole_gate_v1/card/card_levir_mci_seed3333/test_paired_locked_result.json | /root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1/card/card_levir_mci_seed3333/snapshots/card_levir_mci_seed3333_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-584abd764f127e43 | experiments/paired_card_rsaca_whole_gate_v1/rsaca/rsaca_levir_mci_seed3333/test_paired_locked_result.json | /root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1/rsaca/rsaca_levir_mci_seed3333/snapshots/rsaca_levir_mci_seed3333_checkpoint_6000.pt | 8 | raw_fraction_or_cider |
| run-661d48e996faefcd | experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_cc_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_cc_seed1111/snapshots/unified_rsaca_levir_cc_seed1111_checkpoint_7000.pt | 8 | raw_fraction_or_cider |
| run-b28061c642e20dab | experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_cc_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_cc_seed2222/snapshots/unified_rsaca_levir_cc_seed2222_checkpoint_8000.pt | 8 | raw_fraction_or_cider |
| run-b6fc1f461fd1f65a | experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_cc_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_cc_seed3333/snapshots/unified_rsaca_levir_cc_seed3333_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-eead873c9873297b | experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_mci_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_mci_seed1111/snapshots/unified_rsaca_levir_mci_seed1111_checkpoint_7000.pt | 8 | raw_fraction_or_cider |
| run-d2c68bfc41ce8508 | experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_mci_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_mci_seed2222/snapshots/unified_rsaca_levir_mci_seed2222_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-ec8ff65baf753c1b | experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_mci_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_levir_mci_seed3333/snapshots/unified_rsaca_levir_mci_seed3333_checkpoint_8000.pt | 8 | raw_fraction_or_cider |
| run-e109152cf767ccf4 | experiments/reliability_sparse_rsaca_v1/unified_rsaca_second_cc_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_second_cc_seed1111/snapshots/unified_rsaca_second_cc_seed1111_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-c43ed755647f4da6 | experiments/reliability_sparse_rsaca_v1/unified_rsaca_second_cc_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_second_cc_seed2222/snapshots/unified_rsaca_second_cc_seed2222_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-7338d7c86d5de192 | experiments/reliability_sparse_rsaca_v1/unified_rsaca_second_cc_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1/unified_rsaca_second_cc_seed3333/snapshots/unified_rsaca_second_cc_seed3333_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-9132ed9f9419de1f | experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/unified_rsaca_levir_cc_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/unified_rsaca_levir_cc_seed1111/snapshots/unified_rsaca_levir_cc_seed1111_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-5f668cf178e6e6c5 | experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/unified_rsaca_levir_cc_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/unified_rsaca_levir_cc_seed2222/snapshots/unified_rsaca_levir_cc_seed2222_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-e625cecfe4e44823 | experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/unified_rsaca_levir_cc_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_levir_cc_new_masks_matched_control_retry_20260826/unified_rsaca_levir_cc_seed3333/snapshots/unified_rsaca_levir_cc_seed3333_checkpoint_8000.pt | 8 | raw_fraction_or_cider |
| run-5509087e52c0bf02 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_cc_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_cc_seed1111/snapshots/unified_rsaca_levir_cc_seed1111_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-4b3b8a2ba6cdf224 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_cc_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_cc_seed2222/snapshots/unified_rsaca_levir_cc_seed2222_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-d06cfe6e9e2d5d24 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_cc_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_cc_seed3333/snapshots/unified_rsaca_levir_cc_seed3333_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-eba502db5ba74f4f | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_mci_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_mci_seed1111/snapshots/unified_rsaca_levir_mci_seed1111_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-bcf0f042134b7943 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_mci_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_mci_seed2222/snapshots/unified_rsaca_levir_mci_seed2222_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-a387d78e8e773923 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_mci_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_levir_mci_seed3333/snapshots/unified_rsaca_levir_mci_seed3333_checkpoint_9000.pt | 8 | raw_fraction_or_cider |
| run-e9342bbd6240c512 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_second_cc_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_second_cc_seed1111/snapshots/unified_rsaca_second_cc_seed1111_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-ad7ed7a102e9bbeb | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_second_cc_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_second_cc_seed2222/snapshots/unified_rsaca_second_cc_seed2222_checkpoint_8000.pt | 8 | raw_fraction_or_cider |
| run-b2d3e1817cec70f7 | experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_second_cc_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_new_masks_20260821/unified_rsaca_second_cc_seed3333/snapshots/unified_rsaca_second_cc_seed3333_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-bd401e406bbcee67 | experiments/reliability_sparse_rsaca_v1_prenorm_changed_global/unified_rsaca_levir_cc_seed1111/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_prenorm_changed_global/unified_rsaca_levir_cc_seed1111/snapshots/unified_rsaca_levir_cc_seed1111_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-713f09d358961bcb | experiments/reliability_sparse_rsaca_v1_prenorm_changed_global/unified_rsaca_levir_cc_seed2222/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_prenorm_changed_global/unified_rsaca_levir_cc_seed2222/snapshots/unified_rsaca_levir_cc_seed2222_checkpoint_10000.pt | 8 | raw_fraction_or_cider |
| run-41220a76fea65732 | experiments/reliability_sparse_rsaca_v1_prenorm_changed_global/unified_rsaca_levir_cc_seed3333/test_unified_locked_result.json | /root/autodl-tmp/z1zy1/experiments/reliability_sparse_rsaca_v1_prenorm_changed_global/unified_rsaca_levir_cc_seed3333/snapshots/unified_rsaca_levir_cc_seed3333_checkpoint_9000.pt | 8 | raw_fraction_or_cider |

## Dataset Conditions

All locally verifiable `dataset=second_cc` runs whose resolved config names a data root point to `SECOND-CC-AUG`. The relevant inputs are `second_cc_aug_captions_reformat.json`, `splits.json`, augmented vocabulary/H5 labels, and paired semantic maps. A separate non-AUG `SECOND-CC` root was not found; claims must therefore say `SECOND-CC-AUG` or record the condition as unknown.

## Audit Boundaries

Checkpoint equality is currently path-level evidence only: historical file content at test time cannot be reconstructed without a contemporaneous checksum. Prediction coverage, duplicate IDs, empty captions, scorer version, and reference-ID alignment remain unverified where the underlying prediction/reference files are unavailable or not linked from the result wrapper.
