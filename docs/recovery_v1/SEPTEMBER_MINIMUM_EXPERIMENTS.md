# September Minimum Experiments

This schedule is derived from the frozen V1 whole-adapter protocol and the recovery ledger. It does not authorize training in this audit turn.

## P0 — recover existing evidence

1. Register the 11 completed training runs with full validation rows and reconstruct selection using only the locked `paper_balanced_no_spice` validation rule. Keep `reconstructed=true` and a timestamp.
2. Review the two existing LEVIR-MCI seed-3333 test predictions; they pass ID/empty-caption checks, but current hashes do not prove historical test-time bytes.
3. Test only reconstructed selections, one per arm, after confirming no previous locked test exists.
4. Resolve the four missing SECOND-CC-AUG arm roots and the corrupted RSACA seed-1111 checkpoint. Do not change the existing protocol lock.

## P1 — only after P0

- Complete the scratch CARD/whole-adapter paired matrix on SECOND-CC-AUG with the same semantic files, split, selection rule, worker count and scorer.
- Run gate-off / sparse-token / semantic-input diagnostics one variable at a time; diagnostic inference with a module disabled is not a trained ablation.
- Freeze any controlled perturbation on validation data before test use; synthetic noise is not a substitute for a real predicted prior.

## P2 / deferred

Global-token-only, sparse-only or normalization-only studies can follow once the main protocol is complete. P/V/PV combinations, new backbones and large prior-generator training remain deferred.

## Budget accounting

Known existing training time is recorded per `run_summary.json`; no GPU-hour estimate is invented here. The immediate work is primarily registration and validation-only selection, not 16 new trainings. New training is conditional on locating the missing roots or proving the failed checkpoint cannot be safely resumed.
