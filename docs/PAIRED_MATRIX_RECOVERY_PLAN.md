# Paired Matrix Recovery Plan

Generated: `2026-09-09T08:12:06.491925+00:00`  
Repository: `/root/autodl-tmp/z1zy1`  
Pair root: `/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1`  
Protocol ID: `paired_card_rsaca_whole_gate_v1_locked`

This plan is read-only with respect to historical artifacts. It does not create selections, predictions, tests, or checkpoints.

## Mutually Exclusive Primary States

| State | Count | Meaning |
|---|---:|---|
| VERIFIED_COMPLETE | 2 | selection and complete locked test are present |
| REGISTER_EXISTING | 0 | existing evidence only needs registration |
| RECONSTRUCT_SELECTION | 11 | training and full validation range exist; selection record absent |
| NEED_VALIDATION | 0 | checkpoint exists but validation evidence is incomplete |
| NEED_TEST_OR_RESCORING | 0 | selection exists but complete test record does not |
| RESUME_OR_RETRAIN | 1 | training failed/incomplete; continuation equivalence is not established |
| INCOMPATIBLE_OR_UNVERIFIABLE | 4 | target-root evidence is absent |

## Pair status

| Dataset | Seed | CARD | RSACA | Status | Complete | Qualified |
|---|---:|---|---|---|---|---|
| LEVIR-CC | 1111 | RECONSTRUCT_SELECTION | RECONSTRUCT_SELECTION | PENDING_SELECTION_AND_TEST | no | no |
| LEVIR-CC | 2222 | RECONSTRUCT_SELECTION | RECONSTRUCT_SELECTION | PENDING_SELECTION_AND_TEST | no | no |
| LEVIR-CC | 3333 | RECONSTRUCT_SELECTION | RECONSTRUCT_SELECTION | PENDING_SELECTION_AND_TEST | no | no |
| LEVIR-MCI | 1111 | RECONSTRUCT_SELECTION | RECONSTRUCT_SELECTION | PENDING_SELECTION_AND_TEST | no | no |
| LEVIR-MCI | 2222 | RECONSTRUCT_SELECTION | RECONSTRUCT_SELECTION | PENDING_SELECTION_AND_TEST | no | no |
| LEVIR-MCI | 3333 | VERIFIED_COMPLETE | VERIFIED_COMPLETE | QUALIFIED_COMPLETE | yes | yes |
| SECOND-CC-AUG | 1111 | RECONSTRUCT_SELECTION | RESUME_OR_RETRAIN | BLOCKED_BY_RESUME_OR_RETRAIN | no | no |
| SECOND-CC-AUG | 2222 | INCOMPATIBLE_OR_UNVERIFIABLE | INCOMPATIBLE_OR_UNVERIFIABLE | BLOCKED_BY_MISSING_OR_UNVERIFIABLE_EVIDENCE | no | no |
| SECOND-CC-AUG | 3333 | INCOMPATIBLE_OR_UNVERIFIABLE | INCOMPATIBLE_OR_UNVERIFIABLE | BLOCKED_BY_MISSING_OR_UNVERIFIABLE_EVIDENCE | no | no |

## 18-arm ledger

| Dataset | Arm | Seed | State | Train step | Val range | Selection | Test | Reusable | Final comparison |
|---|---|---:|---|---:|---|---|---|---|---|
| LEVIR-CC | card | 1111 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-CC | card | 2222 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-CC | card | 3333 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-CC | rsaca | 1111 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-CC | rsaca | 2222 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-CC | rsaca | 3333 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-MCI | card | 1111 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-MCI | card | 2222 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-MCI | card | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | yes |
| LEVIR-MCI | rsaca | 1111 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-MCI | rsaca | 2222 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| LEVIR-MCI | rsaca | 3333 | VERIFIED_COMPLETE | 10000 | yes | yes | yes | yes | yes |
| SECOND-CC-AUG | card | 1111 | RECONSTRUCT_SELECTION | 10000 | yes | no | no | yes | no |
| SECOND-CC-AUG | card | 2222 | INCOMPATIBLE_OR_UNVERIFIABLE | - | no | no | no | no | no |
| SECOND-CC-AUG | card | 3333 | INCOMPATIBLE_OR_UNVERIFIABLE | - | no | no | no | no | no |
| SECOND-CC-AUG | rsaca | 1111 | RESUME_OR_RETRAIN | - | no | no | no | no | no |
| SECOND-CC-AUG | rsaca | 2222 | INCOMPATIBLE_OR_UNVERIFIABLE | - | no | no | no | no | no |
| SECOND-CC-AUG | rsaca | 3333 | INCOMPATIBLE_OR_UNVERIFIABLE | - | no | no | no | no | no |

## First actions

1. Run the read-only preflight and review this ledger.
2. Reconstruct validation selection for the 11 complete-but-unselected arms using the locked `paper_balanced_no_spice` rule; mark each output `reconstructed=true` and do not call it historical.
3. Run one locked test per reconstructed selection only after review; keep outputs in the existing arm directory only if the runner confirms no test result exists.
4. Do not resume the failed SECOND-CC-AUG RSACA seed 1111 checkpoint as equivalent: the 7000 file is corrupted and the checkpoint payload lacks optimizer/scheduler state. Archive/new-directory retraining requires a new protocol confirmation.
5. Locate the four missing SECOND-CC-AUG arm roots before declaring them retraining targets.

## Dry-run commands

### LEVIR-CC / card / seed 1111
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select --dataset levir_cc --seed 1111 --dry-run
```
### LEVIR-CC / card / seed 2222
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select --dataset levir_cc --seed 2222 --dry-run
```
### LEVIR-CC / card / seed 3333
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select --dataset levir_cc --seed 3333 --dry-run
```
### LEVIR-MCI / card / seed 1111
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select --dataset levir_mci --seed 1111 --dry-run
```
### LEVIR-MCI / card / seed 2222
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select --dataset levir_mci --seed 2222 --dry-run
```
### SECOND-CC-AUG / card / seed 1111
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage select --dataset second_cc --seed 1111 --dry-run
```
### SECOND-CC-AUG / rsaca / seed 1111
```bash
cd /root/autodl-tmp/z1zy1 && PAIR_ROOT=/root/autodl-tmp/z1zy1/experiments/paired_card_rsaca_whole_gate_v1 NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python bash scripts/run_paired_card_rsaca_matrix.sh --stage train --dataset second_cc --seed 1111 --dry-run  # dry-run only; existing failed directory must be archived/new protocol confirmed
```

## Protocol conflicts and evidence boundary

The lock records `validation_only: paper_balanced_no_spice`, commit `64be6e…`, Python `/root/miniconda3/envs/card/bin/python3.8`, NUM_WORKERS=8 and SECOND-CC-AUG through the runner's resolved data root. The current target configs agree on the selection rule and 10000-step budget. Training `git_info.txt` records commit/dirty status separately; current SHA-256 values are not historical test-time content proofs.

| Lock check | Value |
|---|---|
| Git commit matches | True |
| Source digest matches | False |
| Source-status digest matches | False |

The source/source-status digests are checked using the runner's lock algorithm. If either is false, all listed commands remain dry-runs: use a new protocol root or restore the locked source state; do not rewrite the historical lock.

Prediction integrity reports are limited to the two existing LEVIR-MCI seed-3333 test outputs. No new predictions or scores were generated.
