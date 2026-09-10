
## Paired Matrix Recovery Audit (2026-09-09T11:21:33.731864+00:00)

Evidence/recovery update only: no training, test prediction, or formal rescoring was run. The 18 arms are VERIFIED_COMPLETE=2, RECONSTRUCT_SELECTION=0, RESUME_OR_RETRAIN=1, INCOMPATIBLE_OR_UNVERIFIABLE=0. Of 9 expected pairs, 1 have both historical selection/test records and 1 currently qualify after the available prediction-integrity checks. Current-source lock consistency is commit=True, source=True, status=True; when a digest is false, recovery commands remain dry-runs until source isolation is resolved. See `docs/PAIRED_MATRIX_RECOVERY_PLAN.md` and `experiments/audit/paired_matrix_recovery.json`.
