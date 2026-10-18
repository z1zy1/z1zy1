# Artifact Index

The original `artifact_manifest.json` is the registration-time preflight manifest and remains unchanged. The post-run manifest `POST_RUN_ARTIFACT_MANIFEST.json` lists hashes for final reports, protocol and amendment records, code diffs, test logs, valid and invalid run directories, checkpoints, validation predictions, scorer outputs, repeat decodes, mechanism diagnostics, annotation packets, and execution logs.

Valid stage-A runs are under `runs_card_env/runs/C0/` and `runs_card_env/runs/N1/`. The excluded base-Python C0 attempt is retained under `runs/C0/` with `INVALID_ATTEMPT_NOTE.md`. Paired metrics and gate decisions are in `runs_card_env/stageA_results.json`; new CC caption packets are in `runs_card_env/annotation_diagnostics/`.

The registered protocol is `EXPLORATORY_PROTOCOL.json` with SHA-256 `f9bf67e052e6e6510003079a3dbcc9f3e727981ee4a15c9f8e89f9bc96f92469`. Post-registration operational changes are recorded append-only in `PROTOCOL_AMENDMENTS.json`. This filesystem record is not a third-party trusted timestamp.
