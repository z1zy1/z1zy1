#!/usr/bin/env python3
"""Reproducible, non-destructive diagnostics for the B/D/C0/G controls.

The command has six explicit stages: ``inventory``, ``curves``, ``score``,
``infer``, ``content`` and ``report``.  The original ``--cfg ... --checkpoint
... --output ...`` invocation remains supported for the fixed validation
sample forward diagnostic used by the historical regression tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import copy
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, Iterable, Mapping, Optional

PROJECT = Path(os.environ.get("PROJECT_DIR") or Path(__file__).resolve().parents[1]).resolve()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.semantic_controls import ARMS, DATASETS, SEEDS, PROTOCOL
from utils.semantic_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    analyze_content,
    analyze_fusion,
    capture_runtime_identity,
    curve_report,
    inventory_artifacts,
    jsonable,
    read_json,
    render_markdown,
    reproduce_scores,
    source_identity,
)


def _parse_ints(values: Optional[Iterable[str]], allowed: Iterable[int]) -> Optional[list[int]]:
    if values is None:
        return None
    result = [int(value) for value in values]
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise ValueError("unknown integer filter(s): %s" % unknown)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_outputs(output_dir: Path, payload: Mapping[str, Any], *, markdown: Optional[str] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "diagnostic_report.json", payload)
    if markdown is not None:
        with (output_dir / "diagnostic_report.md").open("x", encoding="utf-8") as handle:
            handle.write(markdown)


def _guard_output(args):
    output = Path(getattr(args, "output_dir", None) or args.output).resolve()
    root = getattr(args, "experiment_root", None)
    protected = [Path(root).resolve()] if root else []
    for key in ("checkpoint", "prediction", "reference", "cfg", "original_score"):
        value = getattr(args, key, None)
        for item in (value if isinstance(value, list) else [value]):
            if not item:
                continue
            source = Path(str(item).split("=", 1)[-1]).resolve()
            if source == output:
                raise ValueError("output collides with an input file")
            for parent in source.parents:
                if (parent / "protocol.json").exists() or parent.name == PROTOCOL:
                    protected.append(parent)
    for historical in protected:
        if output == historical or historical in output.parents:
            raise ValueError("diagnostic output must be outside historical experiment root")
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise ValueError("diagnostic output already exists and is nonempty")


def _filters(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "arms": list(args.arms) if getattr(args, "arms", None) else None,
        "datasets": list(args.datasets) if getattr(args, "datasets", None) else None,
        "seeds": _parse_ints(getattr(args, "seeds", None), SEEDS),
        "steps": _parse_ints(getattr(args, "steps", None), tuple(range(1000, 10001, 1000))),
    }


def _run_inventory(args: argparse.Namespace) -> Dict[str, Any]:
    return inventory_artifacts(args.experiment_root, project=getattr(args, "project", PROJECT), **_filters(args))


def _run_curves(args: argparse.Namespace) -> Dict[str, Any]:
    inventory = _run_inventory(args)
    return {"inventory": inventory,
            "curves": curve_report(inventory, require_checkpoints=not getattr(args, "no_checkpoint_validation", False)),
            "diagnostic_source": source_identity(getattr(args, "project", PROJECT))}


def _parse_key_value_paths(values: Optional[Iterable[str]]) -> Dict[str, str]:
    aliases = {"B": "card", "D": "plain_fusion", "C0": "rsaca", "G": "fixed_gate"}
    result: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("expected ARM=PATH, got %s" % value)
        arm, path = value.split("=", 1)
        if arm not in ARMS and arm not in aliases:
            raise ValueError("unknown arm %s" % arm)
        result[aliases.get(arm, arm)] = path
    return result


def _run_score(args: argparse.Namespace, output_dir: Optional[Path]) -> Dict[str, Any]:
    expected_ids = None
    if args.expected_ids_file:
        payload = read_json(args.expected_ids_file)
        expected_ids = payload.get("sample_ids", payload) if isinstance(payload, Mapping) else payload
    return reproduce_scores(
        args.prediction, args.reference, expected_ids=expected_ids,
        scorer_command=args.scorer_command, original_score_path=args.original_score,
        output_dir=output_dir, tolerance=args.tolerance, spice=args.spice,
        dry_run=args.dry_run,
    )


def _run_content(args: argparse.Namespace) -> Dict[str, Any]:
    predictions = _parse_key_value_paths(args.prediction)
    result = analyze_content(predictions, sample_seed=args.sample_seed, sample_count=args.sample_count,
                             failure_ids=getattr(args, "failure_ids", None))
    if args.fusion:
        result["fusion"] = analyze_fusion(_parse_key_value_paths(args.fusion))
    return result


def _load_checkpoint_for_diagnostic(path: Path, *, require_integrity: bool) -> Dict[str, Any]:
    from utils.checkpoint_integrity import sha256_file, validate_checkpoint_file
    from utils.checkpointing import load_checkpoint_file

    identity: Dict[str, Any] = {"path": str(path.resolve()), "sha256": sha256_file(str(path))}
    filename_match = re.search(r'_checkpoint_(\d+)\.(?:pt|pth)$', path.name)
    filename_step = int(filename_match.group(1)) if filename_match else None
    try:
        integrity_digest = validate_checkpoint_file(
            str(path), require_checksum=require_integrity, require_metadata=require_integrity,
            expected_step=filename_step)
        identity["integrity"] = "verified"
        identity["integrity_sha256"] = integrity_digest
    except (OSError, ValueError) as exc:
        if require_integrity:
            raise
        identity["integrity"] = "unverified"
        identity["integrity_error"] = str(exc)
    # Keep the pre-normalized mapping so the legacy loader's default ``0``
    # cannot turn a missing runtime field into a falsely verified step.
    import torch
    try:
        raw_payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch 1.10 has no weights_only keyword; do not hide other load
        # failures, which must remain visible as checkpoint corruption.
        raw_payload = torch.load(str(path), map_location="cpu")
    payload = load_checkpoint_file(str(path), map_location="cpu")
    raw_step = raw_payload.get("global_step") if isinstance(raw_payload, Mapping) else None
    if raw_step is None:
        if require_integrity:
            raise ValueError("checkpoint global_step is missing; strict runtime-state verification is unavailable")
        identity["global_step"] = None
        identity["runtime_step_status"] = "unverified"
    else:
        try:
            identity["global_step"] = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint global_step is invalid: %r" % (raw_step,)) from exc
        if filename_step is not None and identity["global_step"] != filename_step:
            raise ValueError("checkpoint global_step conflicts with filename step: %s != %s" %
                             (identity["global_step"], filename_step))
        identity["runtime_step_status"] = "verified" if filename_step is not None else "unverified"
    if require_integrity and identity.get("runtime_step_status") != "verified":
        raise ValueError("checkpoint runtime step cannot be strictly verified from filename metadata")
    identity["payload_config_present"] = payload.get("config") is not None
    return {"payload": payload, "identity": identity}


def _config_value(config: Any, dotted: str) -> Any:
    current = config
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        else:
            current = getattr(current, part, None)
    return current


def _config_has(config: Any, dotted: str) -> bool:
    current = config
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return False
            current = current[part]
        else:
            if not hasattr(current, part):
                return False
            current = getattr(current, part)
    return True


def _checkpoint_config_compatibility(payload_config: Any, runtime_config: Any) -> Dict[str, Any]:
    """Check architecture/control identity before loading tensor state.

    Dataset paths and other host-specific fields are intentionally excluded;
    vocabulary and sequence dimensions are checked after adapting the runtime
    config from the validation dataset.
    """
    if payload_config is None:
        return {"status": "unverifiable", "reason": "checkpoint has no model_cfg/config payload"}
    def leaf_fields(value, prefix):
        if isinstance(value, Mapping):
            return [field for key, item in value.items() for field in leaf_fields(item, prefix + "." + str(key))]
        return [prefix]
    fields = set(leaf_fields(_config_value(payload_config, "model"), "model")) | set(leaf_fields(_config_value(runtime_config, "model"), "model"))
    fields.update(("train.protocol_id", "data.use_semantic_maps", "data.semantic_diff_only", "data.semantic_diff_binary"))
    runtime_state_fields = {"train.global_step"}
    fields.update(field for field in
                  (set(leaf_fields(_config_value(payload_config, "train"), "train")) |
                   set(leaf_fields(_config_value(runtime_config, "train"), "train")))
                  if field not in runtime_state_fields)
    for config in (payload_config, runtime_config):
        for field in leaf_fields(_config_value(config, "data"), "data"):
            if "semantic" in field and not any(word in field for word in ("path", "root", "file", "dir")):
                fields.add(field)
    fields = sorted(fields)
    mismatches = []
    for field in fields:
        recorded = jsonable(_config_value(payload_config, field))
        runtime = jsonable(_config_value(runtime_config, field))
        if recorded != runtime:
            mismatches.append({"field": field, "checkpoint": recorded, "runtime": runtime})
    # ``None`` is a legitimate explicit value for optional controls such as
    # start_from.  Only an absent key is missing evidence.
    missing = [field for field in fields
               if not _config_has(payload_config, field) or not _config_has(runtime_config, field)]
    return {"status": "mismatch" if mismatches else ("unverifiable" if missing else "match"), "missing_fields": missing, "checked_fields": list(fields),
            "mismatches": mismatches, "vocabulary_mapping_identity": "unverified", "historical_executing_source_identity": "unverified"}


def _rng_fingerprint(state: Mapping[str, Any]) -> str:
    """Record RNG state without putting large binary state into the report."""
    def plain(value):
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
    return hashlib.sha256(json.dumps(plain(state), sort_keys=True).encode()).hexdigest()


def _unpack_change_detector_output(outputs: Any) -> Any:
    """Keep the diagnostic independent from the import-time test entrypoint."""
    if not isinstance(outputs, (tuple, list)) or len(outputs) not in (6, 7, 8):
        size = len(outputs) if hasattr(outputs, "__len__") else "unknown"
        raise ValueError("Unexpected CARD output size: %s" % size)
    return outputs[0], outputs[5]


def run_fixed_forward_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    from configs.config_transformer import cfg
    from utils.seed import seeded_initialization
    saved = copy.deepcopy(cfg)
    cwd = Path.cwd()
    try:
        with seeded_initialization(args.sample_seed):
            return _run_fixed_forward_diagnostic(args)
    finally:
        cfg.clear()
        cfg.update(saved)
        os.chdir(cwd)


def _run_fixed_forward_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the existing validation-only hook without importing a train script."""
    import torch

    from configs.config_transformer import cfg, merge_cfg_from_file
    from datasets.datasets import create_dataset
    from models.CARD import CARD
    from models.transformer_decoder import DynamicSpeaker
    from utils.checkpointing import split_model_states
    from utils.seed import capture_rng_state, restore_rng_state, seeded_initialization
    from utils.utils import decode_sequence_transformer

    if args.count < 1:
        raise ValueError("count must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("diagnostic output already exists; refusing to overwrite: %s" % output)
    checkpoint_path = Path(args.checkpoint).resolve()
    loaded = _load_checkpoint_for_diagnostic(checkpoint_path, require_integrity=args.require_checkpoint_integrity)
    os.chdir(PROJECT)
    merge_cfg_from_file(args.cfg)
    cfg.data.return_dict = True
    cfg.train.validation_greedy = True
    cfg.train.global_step = loaded["identity"]["global_step"]
    dataset, _ = create_dataset(cfg, "val")
    cfg.model.transformer_decoder.vocab_size = dataset.get_vocab_size()
    cfg.model.transformer_decoder.seq_length = dataset.get_max_seq_length()
    config_compatibility = _checkpoint_config_compatibility(loaded["payload"].get("config"), cfg)
    if config_compatibility.get("status") == "mismatch":
        raise ValueError("checkpoint model config is incompatible: %s" % config_compatibility["mismatches"])
    if args.require_checkpoint_integrity and config_compatibility.get("status") != "match":
        raise ValueError("checkpoint configuration is unverified; explicit --allow-unverified-checkpoint is required")
    before_rng = capture_rng_state()
    with seeded_initialization(args.sample_seed):
        model = CARD(cfg).to(args.device).eval()
        speaker = DynamicSpeaker(cfg).to(args.device).eval()
    restore_rng_state(before_rng)
    model_state, speaker_state = split_model_states(loaded["payload"])
    model.load_state_dict(model_state, strict=True)
    speaker_load_error = None
    try:
        speaker.load_state_dict(speaker_state, strict=True)
    except RuntimeError as exc:
        if args.require_checkpoint_integrity:
            raise
        # Keep the existing hook-only diagnostic useful for legacy fixtures,
        # while making a vocabulary/config mismatch explicit rather than
        # silently loading a partial decoder.
        speaker = None
        speaker_load_error = str(exc)
    if loaded["identity"].get("global_step") is not None and hasattr(model, "set_global_step"):
        model.set_global_step(loaded["identity"]["global_step"])
    records = []
    active: Dict[str, Any] = {}
    inference_rng = capture_rng_state()

    def observe(module: Any, inputs: Any, output: Any) -> None:
        query = inputs[0]
        gate = getattr(module, "last_reliability_gate", None)
        coverage = getattr(module, "last_change_coverage", None)
        delta = output - query
        query_norm = query.norm().clamp_min(1e-12)
        active.update({
            "gamma": float(module.gamma.detach()),
            "gate": gate.detach().cpu().tolist() if gate is not None else None,
            "coverage": coverage.detach().cpu().tolist() if coverage is not None else None,
            "residual_l2": float(delta.norm()),
            "query_l2": float(query.norm()),
            "relative_residual": float(delta.norm() / query_norm),
            "hook_status": "available",
        })

    handles = [module.register_forward_hook(observe) for module in model.modules()
               if module.__class__.__name__ == "SemanticCrossAttentionFusion"]
    indices = sorted(random.Random(args.sample_seed).sample(range(len(dataset)), min(args.count, len(dataset))))
    try:
        with torch.no_grad():
            for index in indices:
                active = {}
                sample = dataset[index]

                def tensor(key: str) -> Any:
                    value = sample.get(key)
                    return value.unsqueeze(0).to(args.device) if torch.is_tensor(value) else None

                outputs = model(tensor("feature_before"), tensor("feature_after"),
                                semantic_before=tensor("semantic_before"), semantic_after=tensor("semantic_after"),
                                semantic_diff=tensor("semantic_diff"), semantic_confidence=tensor("semantic_confidence"))
                encoder_output, _ = _unpack_change_detector_output(outputs)
                record = {"sample_id": sample["image_id"], "dataset_index": index}
                if speaker is not None:
                    speaker_output, _ = speaker.sample(encoder_output, sample_max=1)
                    prediction = decode_sequence_transformer(dataset.get_idx_to_word(), speaker_output[:, 1:])[0]
                    record["prediction"] = prediction
                if active:
                    record.update(active)
                else:
                    record.update({"hook_status": "not_applicable", "reason": "no SemanticCrossAttentionFusion module enabled"})
                records.append(record)
    finally:
        for handle in handles:
            handle.remove()
        restore_rng_state(inference_rng)
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "kind": "checkpoint_forward_diagnostic",
        "split": "val",
        "decode": "greedy",
        "sample_seed": args.sample_seed,
        "count": len(records),
        "checkpoint": loaded["identity"],
        "config_compatibility": config_compatibility,
        "config": {"path": str(Path(args.cfg).resolve()), "sha256": hashlib.sha256(Path(args.cfg).read_bytes()).hexdigest()},
        "runtime": capture_runtime_identity(),
        "rng": {
            "scope": "validation inference only; state is restored after the run",
            "before_sha256": _rng_fingerprint(inference_rng),
            "after_sha256": _rng_fingerprint(capture_rng_state()),
        },
        "records": records,
        "predictions": [{"image_id": row["sample_id"], "prediction": row["prediction"]}
                        for row in records if "prediction" in row],
        "prediction_status": "available" if speaker is not None else "unavailable",
        "prediction_error": speaker_load_error,
        "interpretation": "Validation-only eval/greedy inference; predictions and fusion behavior are diagnostic evidence, not a retrained ablation or full validation score.",
    }


def _run_report(args: argparse.Namespace, output_dir: Optional[Path]) -> Dict[str, Any]:
    inventory = _run_inventory(args)
    curves = curve_report(inventory, require_checkpoints=not args.no_checkpoint_validation)
    report: Dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "kind": "semantic_control_report",
        "experiment_root": str(Path(args.experiment_root).resolve()),
        "diagnostic_source": source_identity(args.project),
        "historical_claim_boundary": "unknown; consult original audit, training logs and frozen admission artifacts",
        "inventory": inventory,
        "curves": curves,
        "scoring": [],
        "inference": {"status": "not_run", "command": "infer"},
        "content": None,
        "evidence_assessment": {
            "model_change_ready": False,
            "reason": "This command only audits existing artifacts; no new model structure or training result is enabled.",
            "required_next_checks": ["same-checkpoint validation inference repetition", "original scorer environment and ID audit", "failure-case content/fusion review"],
        },
    }
    if args.prediction and args.reference:
        for index, value in enumerate(args.prediction):
            prediction_path = value.split("=", 1)[1] if "=" in value else value
            score = reproduce_scores(
                prediction_path, args.reference, scorer_command=args.scorer_command,
                original_score_path=args.original_score, output_dir=output_dir / ("score_%03d" % index) if output_dir else None,
                tolerance=args.tolerance, spice=args.spice, dry_run=output_dir is None)
            report["scoring"].append({"arm": value.split("=", 1)[0] if "=" in value else "unknown", **score})
        report["content"] = analyze_content(_parse_key_value_paths(args.prediction), sample_seed=args.sample_seed,
                                             sample_count=args.sample_count, failure_ids=args.failure_ids)
    if args.fusion:
        report["fusion"] = analyze_fusion(_parse_key_value_paths(args.fusion))
    return report


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=1111)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-checkpoint-integrity", action="store_true")
    return parser


def _add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument("--steps", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="list expected and observed artifacts")
    _add_common_filters(inventory)
    curves = sub.add_parser("curves", help="recompute validation curves and selection")
    _add_common_filters(curves)
    curves.add_argument("--no-checkpoint-validation", action="store_true")
    score = sub.add_parser("score", help="re-score existing predictions")
    score.add_argument("--experiment-root")
    score.add_argument("--prediction", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--expected-ids-file")
    score.add_argument("--original-score")
    # REMAINDER allows commands such as ``python -c ...``; keep this option
    # last on the command line because all following tokens belong to scorer.
    score.add_argument("--scorer-command", nargs=argparse.REMAINDER)
    score.add_argument("--spice", action="store_true")
    score.add_argument("--tolerance", type=float, default=1e-12)
    score.add_argument("--dry-run", action="store_true")
    content = sub.add_parser("content", help="compare descriptions and fusion hook records")
    content.add_argument("--experiment-root")
    content.add_argument("--prediction", nargs="+", required=True)
    content.add_argument("--fusion", nargs="+")
    content.add_argument("--sample-seed", type=int, default=1111)
    content.add_argument("--sample-count", type=int, default=16)
    content.add_argument("--failure-ids", nargs="*", default=None)
    content.add_argument("--output-dir", required=True)
    content.add_argument("--dry-run", action="store_true")
    infer = sub.add_parser("infer", help="repeat validation-only checkpoint inference")
    infer.add_argument("--experiment-root")
    infer.add_argument("--cfg", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--count", type=int, default=16)
    infer.add_argument("--sample-seed", type=int, default=1111)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--repeat", type=int, default=1)
    infer.add_argument("--require-checkpoint-integrity", dest="require_checkpoint_integrity", action="store_true")
    infer.add_argument("--allow-unverified-checkpoint", dest="require_checkpoint_integrity", action="store_false")
    infer.set_defaults(require_checkpoint_integrity=True)
    infer.add_argument("--dry-run", action="store_true")
    report = sub.add_parser("report", help="write a combined Chinese Markdown and JSON report")
    _add_common_filters(report)
    report.add_argument("--no-checkpoint-validation", action="store_true")
    report.add_argument("--prediction", nargs="+")
    report.add_argument("--reference")
    report.add_argument("--original-score")
    report.add_argument("--scorer-command", nargs=argparse.REMAINDER)
    report.add_argument("--fusion", nargs="+")
    report.add_argument("--spice", action="store_true")
    report.add_argument("--tolerance", type=float, default=1e-12)
    report.add_argument("--sample-seed", type=int, default=1111)
    report.add_argument("--sample-count", type=int, default=16)
    report.add_argument("--failure-ids", nargs="*", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--"):
        args = _legacy_parser().parse_args(argv)
        _guard_output(args)
        payload = run_fixed_forward_diagnostic(args)
        _write_json(Path(args.output), payload)
        return 0
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _guard_output(args)
        if args.command == "infer" and args.repeat < 1:
            raise ValueError("repeat must be positive")
        if args.command == "inventory":
            payload = _run_inventory(args)
        elif args.command == "curves":
            payload = _run_curves(args)
        elif args.command == "score":
            payload = _run_score(args, None if args.dry_run else Path(args.output_dir).resolve())
        elif args.command == "content":
            payload = _run_content(args)
        elif args.command == "infer":
            if args.dry_run:
                payload = {
                    "schema": DIAGNOSTIC_SCHEMA,
                    "kind": "checkpoint_forward_diagnostic_plan",
                    "status": "dry_run",
                    "split": "val",
                    "decode": "greedy",
                    "cfg": str(Path(args.cfg).resolve()),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "output": str(Path(args.output).resolve()),
                    "repeat": max(1, args.repeat),
                    "note": "No checkpoint was loaded and no output was written.",
                }
                print(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True))
                return 0
            repeats = []
            for _ in range(max(1, args.repeat)):
                repeats.append(run_fixed_forward_diagnostic(args))
            payload = repeats[0]
            if len(repeats) > 1:
                first_records = repeats[0].get("records", [])
                payload["repeat"] = {
                    "requested": len(repeats),
                    "records_identical": all(item.get("records", []) == first_records for item in repeats[1:]),
                    "predictions_identical": bool(repeats[0].get("predictions")) and all(item.get("predictions", []) == repeats[0].get("predictions", []) for item in repeats[1:]),
                    "comparison": "exact JSON equality of fixed validation records and greedy predictions",
                }
            payload["repeat_evidence"] = repeats[1:]
            _write_json(Path(args.output).resolve(), payload)
            print(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True))
            return 0
        elif args.command == "report":
            payload = _run_report(args, None if args.dry_run else Path(args.output_dir).resolve())
        else:
            parser.error("unknown command")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True))
    if not getattr(args, "dry_run", False):
        output_dir = Path(args.output_dir).resolve()
        markdown = render_markdown(payload if args.command == "report" else {"inventory": payload.get("inventory", payload), "curves": payload.get("curves", {})})
        _write_outputs(output_dir, payload, markdown=markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
