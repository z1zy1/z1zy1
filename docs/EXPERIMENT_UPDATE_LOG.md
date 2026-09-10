# Experiment Update Log

## 2026-09-08 audit correction

- Old behavior: `summarize_unified_rsaca.py` and `summarize_paired_card_rsaca_matrix.py` emitted `0.0` sample standard deviation when only one seed was present.
- New behavior: one-seed sample standard deviation is `null` / `N/A`; it is not numerical evidence of zero training variation.
- Old behavior: checkpoint agreement was inferred from path equality only and not surfaced as an evidence limitation.
- New behavior: the registry records `path_match_only` separately from a content checksum and reports missing historical checksum evidence.
- Result provenance: original result wrappers and raw scorer JSON are retained unchanged. This update is a registry/summary correction, not a re-score.

## Paired Matrix Recovery Audit (2026-09-09T08:12:06.491925+00:00)

Evidence/recovery update only: no training, test prediction, or formal rescoring was run. The 18 arms are VERIFIED_COMPLETE=2, RECONSTRUCT_SELECTION=11, RESUME_OR_RETRAIN=1, INCOMPATIBLE_OR_UNVERIFIABLE=4. Of 9 expected pairs, 1 have both historical selection/test records and 1 currently qualify after the available prediction-integrity checks. Current-source lock consistency is commit=True, source=False, status=False; when a digest is false, recovery commands remain dry-runs until source isolation is resolved. See `docs/PAIRED_MATRIX_RECOVERY_PLAN.md` and `experiments/audit/paired_matrix_recovery.json`.

## 2026-09-09 recovery v2 execution

- Old state: SECOND-CC-AUG seeds 1111 and 3333 were missing complete recovery test evidence; seed 1111 also had a corrupted historical RSACA run.
- New state: independent `recovery_v2` contains complete CARD/RSACA validation selections and `paired_recovery_seeded` tests for both pairs, each covering 1,227/1,227 test samples.
- Seed 1111 RSACA action: historical directory was preserved/archived; optimizer state was unavailable, so a from-scratch retrain was run. This is not an equivalent resume.
- Seed 3333 RSACA action: new 10,000-step recovery training was completed; selection used validation-only `paper_balanced_no_spice`.
- Test wrapper correction: `SEED` is now passed explicitly to test-time overrides; old `paired_locked` results remain untouched. New scores are independent recovery evaluations, not replacements or additional training seeds.
- Evidence boundary: both completed pairs are excluded from the verified main table because `git_info.txt` reports `dirty=True`; this is a protocol/evidence limitation, not a claim about model quality.

## 2026-09-09 recovery v2 final audit

- Final state: all 18 arms and all 9 expected pairs have complete recovery-v2 selection and `paired_recovery_seeded` test records.
- Observed/qualified: `observed_pairs=9`, `qualified_pairs=0`; every prediction integrity report passed ID, duplicate, extra, missing, and empty-caption checks.
- Qualification remains blocked by dirty training source evidence, not by missing metrics.
- Recovery-v2 paired statistics are recorded in `docs/CURRENT_VERIFIED_RESULTS.md`; no significance test or confidence interval was calculated.
