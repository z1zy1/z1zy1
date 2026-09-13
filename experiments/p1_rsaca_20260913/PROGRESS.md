# P1 RSACA execution progress

- Protocol: `p1_rsaca_20260913`
- Status: P1 preflight passed; first launch failed before training, boolean config fix committed, and all 18 rerun trainings completed
- Source at implementation freeze: `b4fb7a81f2ccb4ec774d1196bde3c3b296b25ea5`
- Selection: validation-only `five_metric_equal_weight_log`, five metrics, earlier-step tie break
- Reference: `configs/protocols/p1_rsaca_20260913.json`
- Existing recovery-v2 results remain P0 exploratory evidence and are not mixed into P1.
- Launch failure: `train.validation_greedy` received integer `1` instead of a boolean; no GPU work or result was produced.
- Training: tmux `p1_rsaca_train_r1` completed all 18 runs; no training process is currently running. Next action is validation selection, then immutable test.

The working tree contains unrelated historical experiment artifacts and user deletions; they are intentionally preserved.
