# Mechanism Diagnostics

Diagnostics used the selector-chosen C0 step-9000 and N1 step-10000 checkpoints on the same fixed 64 CC validation IDs, sampled with seed 20261018. The existing forward hook ran in eval mode with inference RNG restored; before/after RNG fingerprints match for both arms. Raw per-sample records and predictions are in `../reports/diagnostics/mechanism_C0_fixed64.json` and `mechanism_N1_fixed64.json`.

| Measure | C0 | N1 |
|---|---:|---:|
| Learned gamma | 0.03663923963904381 | 0.05906827747821808 |
| Mean reliability gate | 0.19815282931085676 | 0.46581534296274185 |
| Gate range | 0 to 0.8420413732528687 | 0 to 0.9999912977218628 |
| Mean `||fused-query|| / (||query|| + eps)` | 0.24024931085295975 | 0.054290592670440674 |
| Median relative residual | 0 | 0 |
| Empty semantic-difference map | 34 / 64 | 34 / 64 |
| Coverage below 0.1 | 54 / 64 | 54 / 64 |
| Coverage range | 0 to 0.2857142686843872 | 0 to 0.2857142686843872 |
| Effective changed semantic tokens (196 query cells) | mean 7.78125, median 0, range 0-56 | mean 7.78125, median 0, range 0-56 |

Coverage is taken from the model's `last_change_coverage` hook for the configured binary `semantic_diff` input sourced from LEVIR-CC `pseudo_masks`; the normal 7-to-8 runtime class derivation remains active. The module averages its change mask over the 196 query locations; the reported token count is the corresponding exact integer mask sum. Empty means the supplied pseudo-semantic map contains no usable change tokens. It does not mean the image pair has no real change. Gate/residual summaries are descriptive over this fixed 64-item validation subset, not inferential evidence.

For the 64-sample diagnostic subset, corpus scorers were recomputed over the whole empty-map group (34 IDs) and low-coverage group (54 IDs), not averaged over sentences. Empty-map C0 vs N1: BLEU-4 0.7516469341 vs 0.7577877314, METEOR 0.5031147929 vs 0.4899905850, ROUGE-L 0.8594888684 vs 0.8472954185, CIDEr 1.6611668558 vs 1.6577305527, SPICE 0.3950711702 vs 0.3962344029. Low-coverage C0 vs N1: BLEU-4 0.5443036224 vs 0.5523027486, METEOR 0.4092729291 vs 0.4042686318, ROUGE-L 0.7636539399 vs 0.7776016868, CIDEr 1.4022809691 vs 1.4602186827, SPICE 0.3500906430 vs 0.3535303080. These overlapping groups are selected from one fixed validation subset and are diagnostic only. The exact prediction subsets and scorer JSON are under `../runs_card_env/mechanism_subgroups/`.

The mechanism observations are consistent with a smaller fused-query residual under N1 on this subset, but do not establish why the full validation metrics fell or confirm a historical root cause. They do not show that spatial relation errors are solved.
