# P1 RSACA execution progress

- Protocol: `p1_rsaca_20260913`
- Status: P1 preflight passed; first launch failed during boolean config parsing before training; fix committed and rerun queued in a new root
- Source at implementation freeze: `1bcc909b16a1f856d3f6543c06125f1f11df2d07`
- Selection: validation-only `five_metric_equal_weight_log`, five metrics, earlier-step tie break
- Reference: `configs/protocols/p1_rsaca_20260913.json`
- Existing recovery-v2 results remain P0 exploratory evidence and are not mixed into P1.
- Launch failure: `train.validation_greedy` received integer `1` instead of a boolean; no GPU work or result was produced.
- Next executable action: monitor the rerun queue; after all runs complete, run validation selection, then immutable test.

The working tree contains unrelated historical experiment artifacts and user deletions; they are intentionally preserved.
