# Current Verified Results

Audit time: `2026-09-08 07:44:20 UTC`  
Evidence status: **partial audit**. Values are not promoted to a paper claim unless their comparison group is marked verified.

## Main-Table Status

No group is currently eligible for a verified paper main table. The strict paired CARD/whole-gate RSACA matrix requires 9 seed-pairs / 18 selected-and-tested arms; local evidence covers 1/9 pairs (2/18 arms).

## What Can Be Written Now

- The project contains test records and validation-selection records for several CARD/RSACA runs.
- `SECOND-CC` local runs use the `SECOND-CC-AUG` data condition when a resolved configuration is available.
- Existing single-CARD versus three-seed RSACA summaries are exploratory comparisons against a single audit baseline, not three-seed paired gains.

## Audited Descriptive Summaries

These values are recomputed from the locally accessible selected-test result wrappers. They use the raw scorer scale and must not be treated as a paired main-table result.

| Group | Dataset | observed/expected seeds | B4 mean +/- sample sd | B4 delta | CIDEr delta | SPICE delta |
|---|---|---:|---:|---:|---:|---:|
| unified_rsaca_vs_single_audit_card | levir_cc | 3/3 | 0.5541 +/- 0.0090 | -0.0585 | -0.0026 | 0.0612 |
| unified_rsaca_vs_single_audit_card | levir_mci | 3/3 | 0.5762 +/- 0.0237 | 0.0141 | 0.0453 | 0.0091 |
| unified_rsaca_vs_single_audit_card | second_cc | 3/3 | 0.3065 +/- 0.0209 | 0.0113 | 0.0329 | -0.0005 |
| whole_gate_retry_vs_single_audit_card | levir_cc | 3/3 | 0.5926 +/- 0.0271 | -0.0200 | -0.0004 | 0.0215 |
| whole_gate_retry_vs_single_audit_card | levir_mci | 3/3 | 0.5661 +/- 0.0339 | 0.0039 | 0.0362 | 0.0088 |
| whole_gate_retry_vs_single_audit_card | second_cc | 3/3 | 0.3168 +/- 0.0038 | 0.0215 | 0.0536 | 0.0048 |

The only currently complete strict seed-pair is LEVIR-MCI seed 3333: B4=+0.024005, CIDEr=+0.056049, SPICE=-0.008365; paired sample standard deviations are N/A because observed_n=1.

## What Cannot Be Written Now

- That CARD/RSACA completed a strict `3 x 2 x 3` paired matrix.
- A paired mean delta, confidence interval, significance claim, or zero standard deviation for any single-seed comparison.
- That a historical tested checkpoint has identical content to the validation-selected file based solely on matching paths.
- That SECOND-CC non-AUG conditions were evaluated.

## Minimal Next Experiments

1. Complete the remaining 16 validation-selected, locked-test records in the existing paired matrix without changing its lock.
2. Persist a SHA-256 checksum at selection and test time, plus scorer commit/version.
3. Retain/link prediction and reference IDs, then audit test coverage and duplicate/empty predictions.
4. Run a separately registered non-AUG SECOND-CC condition only if it is a paper target; never pool it with SECOND-CC-AUG.

## Paired Matrix Recovery Audit (2026-09-09T08:12:06.491925+00:00)

Evidence/recovery update only: no training, test prediction, or formal rescoring was run. The 18 arms are VERIFIED_COMPLETE=2, RECONSTRUCT_SELECTION=11, RESUME_OR_RETRAIN=1, INCOMPATIBLE_OR_UNVERIFIABLE=4. Of 9 expected pairs, 1 have both historical selection/test records and 1 currently qualify after the available prediction-integrity checks. Current-source lock consistency is commit=True, source=False, status=False; when a digest is false, recovery commands remain dry-runs until source isolation is resolved. See `docs/PAIRED_MATRIX_RECOVERY_PLAN.md` and `experiments/audit/paired_matrix_recovery.json`.

## Recovery v2 Update (2026-09-09)

The independent `experiments/paired_card_rsaca_recovery_v2` protocol root was created after fixing test-time seed propagation. It contains new validation-only selections and the independent `paired_recovery_seeded` test results for SECOND-CC-AUG seeds 1111 and 3333. Both pairs cover all 1,227 test samples with no duplicate, missing, extra, or empty-caption IDs. Seed 1111 RSACA was retrained from scratch because the historical run was corrupted and lacked optimizer state; seed 3333 RSACA was also trained in this recovery root. These are recovery/re-evaluation records, not historical lock replacements. Both pairs remain excluded from the verified main table because their training `git_info.txt` records `dirty=True` and the exact historical training source cannot be reconstructed. See `docs/recovery_v2/PAIRED_MATRIX_RECOVERY_PLAN.md` and `experiments/audit/recovery_v2/paired_matrix_recovery.json`.

The final recovery-v2 audit now contains all 18 arms and all 9 expected pairs (`observed_pairs=9`, `qualified_pairs=0`). LEVIR-CC, LEVIR-MCI, and SECOND-CC-AUG each have three CARD/RSACA pairs with prediction integrity status `PASS_IDS_AND_CAPTIONS`; these results are suitable for exploratory/recovery tables only until a clean, source-matched training protocol is rerun or the historical source state is recovered.

Recovery-v2 exploratory paired statistics (raw scorer scale, three seeds, sample standard deviation; not a verified main-table result):

| Dataset | CARD B4 mean +/- sd | RSACA B4 mean +/- sd | paired B4 delta mean +/- sd | paired CIDEr delta mean +/- sd | paired SPICE delta mean +/- sd |
|---|---:|---:|---:|---:|---:|
| LEVIR-CC | 0.572697 +/- 0.012601 | 0.560247 +/- 0.014113 | -0.012450 +/- 0.026610 | +0.013763 +/- 0.013086 | +0.023324 +/- 0.030215 |
| LEVIR-MCI | 0.584544 +/- 0.018534 | 0.610764 +/- 0.005447 | +0.026219 +/- 0.013414 | +0.050613 +/- 0.014631 | -0.008662 +/- 0.008801 |
| SECOND-CC-AUG | 0.292357 +/- 0.009340 | 0.307620 +/- 0.005681 | +0.015264 +/- 0.014251 | +0.042680 +/- 0.015673 | +0.010805 +/- 0.017042 |

These means use the new `paired_recovery_seeded` evaluations and must not be mixed with the old `paired_locked` records. No confidence interval or significance test was calculated.
