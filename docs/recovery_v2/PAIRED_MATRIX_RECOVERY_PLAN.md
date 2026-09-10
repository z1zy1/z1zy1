# Paired Matrix Recovery Plan

Generated: `2026-09-09T12:36:14.591532+00:00`  
Repository: `/root/autodl-tmp/z1zy1`  
Pair root: `/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_recovery_v2`  
Protocol ID: `paired_card_rsaca_whole_gate_v1_locked`

This plan does not overwrite historical artifacts. Recovery actions may have been executed in a separate protocol root; all such actions are recorded separately from historical results.

## Mutually Exclusive Primary States

| State | Count | Meaning |
|---|---:|---|
| VERIFIED_COMPLETE | 18 | selection and complete locked test are present |
| REGISTER_EXISTING | 0 | existing evidence only needs registration |
| RECONSTRUCT_SELECTION | 0 | training and full validation range exist; selection record absent |
| NEED_VALIDATION | 0 | checkpoint exists but validation evidence is incomplete |
| NEED_TEST_OR_RESCORING | 0 | selection exists but complete test record does not |
| RESUME_OR_RETRAIN | 0 | training failed/incomplete; continuation equivalence is not established |
| INCOMPATIBLE_OR_UNVERIFIABLE | 0 | target-root evidence is absent |

## Pair status

| Dataset | Seed | CARD | RSACA | Status | Complete | Qualified |
|---|---:|---|---|---|---|---|
| LEVIR-CC | 1111 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| LEVIR-CC | 2222 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| LEVIR-CC | 3333 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| LEVIR-MCI | 1111 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| LEVIR-MCI | 2222 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| LEVIR-MCI | 3333 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| SECOND-CC-AUG | 1111 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| SECOND-CC-AUG | 2222 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |
| SECOND-CC-AUG | 3333 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION | yes | no |

## 18-arm ledger

| Dataset | Arm | Seed | State | Train step | Val range | Selection | Test | Reusable | Final comparison |
|---|---|---:|---|---:|---|---|---|---|---|
| LEVIR-CC | card | 1111 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-CC | card | 2222 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-CC | card | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-CC | rsaca | 1111 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-CC | rsaca | 2222 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-CC | rsaca | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-MCI | card | 1111 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-MCI | card | 2222 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-MCI | card | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-MCI | rsaca | 1111 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-MCI | rsaca | 2222 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| LEVIR-MCI | rsaca | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| SECOND-CC-AUG | card | 1111 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| SECOND-CC-AUG | card | 2222 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| SECOND-CC-AUG | card | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| SECOND-CC-AUG | rsaca | 1111 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| SECOND-CC-AUG | rsaca | 2222 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |
| SECOND-CC-AUG | rsaca | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | no |

## First actions

1. Run the read-only preflight and review this ledger.
2. Reconstruct validation selection for the 11 complete-but-unselected arms using the locked `paper_balanced_no_spice` rule; mark each output `reconstructed=true` and do not call it historical.
3. Run one locked test per reconstructed selection only after review; keep outputs in the existing arm directory only if the runner confirms no test result exists.
4. Do not resume the failed SECOND-CC-AUG RSACA seed 1111 checkpoint as equivalent: the 7000 file is corrupted and the checkpoint payload lacks optimizer/scheduler state. Archive/new-directory retraining requires a new protocol confirmation.
5. Locate the four missing SECOND-CC-AUG arm roots before declaring them retraining targets.

## Dry-run commands


## Protocol conflicts and evidence boundary

The lock records `validation_only: paper_balanced_no_spice`, commit `64be6e…`, Python `/root/miniconda3/envs/card/bin/python3.8`, NUM_WORKERS=8 and SECOND-CC-AUG through the runner's resolved data root. The current target configs agree on the selection rule and 10000-step budget. Training `git_info.txt` records commit/dirty status separately; current SHA-256 values are not historical test-time content proofs.

| Lock check | Value |
|---|---|
| Git commit matches | True |
| Source digest matches | True |
| Source-status digest matches | True |

The source/source-status digests are checked using the runner's lock algorithm. If either is false, all listed commands remain dry-runs: use a new protocol root or restore the locked source state; do not rewrite the historical lock.

Prediction integrity reports cover the selected test tag `paired_recovery_seeded`; current SHA-256 values prove current files only and do not rewrite historical evaluation records.
