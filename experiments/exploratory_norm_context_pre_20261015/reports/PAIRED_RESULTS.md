# Paired Results

The valid stage-A pair is LEVIR-CC seed 2222. C0 and N1 each completed 10,000 updates in the registered `card` environment and used the registered validation grid and selector. C0's first 10,000-step attempt used the base Python environment because the launcher did not export `PYTHON`; that complete attempt is retained under `../runs/C0/` and is invalid for selection or comparison. The runner was amended to export the registered Python executable; the valid pair is stored under `../runs_card_env/`.

| Arm | Selected step | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 9000 | 0.560769241524766 | 0.3956319094121805 | 0.7447305684578187 | 1.3576223988287586 | 0.337389545129031 |
| N1 | 10000 | 0.5363194694388981 | 0.3884119238284123 | 0.7373990836567516 | 1.3496363810987784 | 0.3443531404175354 |
| N1 - C0 | -0.02444977208586796 | -0.00721998558376824 | -0.007331484801067156 | -0.0079860177299802 | +0.006963595288504376 |

Stage-A gate: `ΔCIDEr > 0` failed; `ΔBLEU-4 >= 0` failed; `ΔSPICE >= 0` passed. The pair and full-validation outputs otherwise passed the registered validity checks. Stage B was not entered and stage C was not entered. This is one screening seed, not a seed-level uncertainty estimate; no mean, sample standard deviation, or seed sign count is applicable. The remaining three metrics declined while SPICE increased.

The independent selected-checkpoint full-validation decode matched N1's grid predictions exactly. C0's repeated decoder differed on 2/1,333 captions relative to its step-9000 in-training predictions: `val_000144.png` changed from “some trees in the lower-right corner of the scene disappear” to “the two scenes seem identical”, and `val_001435.png` changed from “a parking lot is built in the open space” to “the vegetation has been removed and a road is built on the bareland”. A second independent repeat exactly matched the first repeat. The cause of the training-evaluation versus independent-evaluation discrepancy was not established; it is retained for diagnosis, with no threshold relaxation. The repeated scores were BLEU-4 0.560524353446705, METEOR 0.39552614541617037, ROUGE-L 0.7445054483511463, CIDEr 1.356927231954079, SPICE 0.33718819609047274. These scores do not replace the registered grid metrics used for the gate.

No additional seeds, MCI runs, SECOND runs, additional CARD training, formal test, or freeze were run. Historical CARD scores are not used as the direct comparison.
