"""Pure, side-effect-free helpers for semantic-control experiment audits.

The functions in this module intentionally do not construct a dataset, import a
training entry point, or mutate an experiment directory.  They operate on
already published artifacts and return JSON-serialisable dictionaries.  The
CLI in :mod:`scripts.diagnose_semantic_controls` is responsible for writing
reports to a separate output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.checkpoint_integrity import sha256_file, validate_checkpoint_file
from utils.experiment_tracking import stable_hash
from utils.semantic_controls import ARMS, DATASETS, METRICS, SEEDS, PROTOCOL


EXPECTED_STEPS = tuple(range(1000, 10001, 1000))
COMPARISON_PAIRS = (
    ("D-B", "plain_fusion", "card"),
    ("C0-B", "rsaca", "card"),
    ("C0-D", "rsaca", "plain_fusion"),
    ("C0-G", "rsaca", "fixed_gate"),
)
DIAGNOSTIC_SCHEMA = "semantic_control_diagnostics.v1"


def read_json(path: os.PathLike[str] | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonable(value: Any) -> Any:
    """Convert common scientific values without hiding unsupported objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def file_identity(path: os.PathLike[str] | str, *, hash_file: bool = True) -> Dict[str, Any]:
    path = Path(path)
    result: Dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        result["status"] = "missing"
        return result
    result["size_bytes"] = path.stat().st_size
    if hash_file:
        try:
            result["sha256"] = sha256_file(str(path))
            result["status"] = "verified"
        except OSError as exc:
            result["status"] = "unverifiable"
            result["error"] = str(exc)
    else:
        result["status"] = "present"
    return result


def _run_git(project: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(project), *args], text=True).strip()


def source_identity(project: os.PathLike[str] | str) -> Dict[str, Any]:
    """Record the diagnostic source separately from each historical run source."""
    project = Path(project).resolve()
    try:
        commit = _run_git(project, "rev-parse", "HEAD")
        tracked = _run_git(project, "ls-files").splitlines()
        relevant = [
            name for name in tracked
            if name.endswith((".py", ".yaml", ".yml", ".sh"))
            and (name.split("/", 1)[0] in {"configs", "models", "datasets", "utils", "scripts", "tools"}
                 or "/" not in name)
        ]
        files = {name: sha256_file(project / name) for name in relevant
                 if (project / name).is_file()}
        dirty = _run_git(project, "status", "--porcelain", "--", *relevant)
        return {
            "project": str(project),
            "commit": commit,
            "execution_files": files,
            "execution_sha256": stable_hash(files),
            "dirty_execution": bool(dirty),
            "status": "verified",
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"project": str(project), "status": "unverifiable", "error": str(exc)}


def _selected_values(values: Optional[Iterable[Any]], allowed: Sequence[Any]) -> List[Any]:
    if values is None:
        return list(allowed)
    selected = list(values)
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError("unknown filter value(s): %s" % unknown)
    return selected


def run_matrix(
    experiment_root: os.PathLike[str] | str,
    *,
    arms: Optional[Iterable[str]] = None,
    datasets: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
) -> Iterable[Tuple[str, int, str, Path]]:
    root = Path(experiment_root).resolve()
    for dataset in _selected_values(datasets, DATASETS):
        for seed in _selected_values(seeds, SEEDS):
            for arm in _selected_values(arms, ARMS):
                yield dataset, int(seed), arm, root / arm / ("%s_%s_seed%d" % (arm, dataset, seed))


def _normalise_path(path: Any, base: Path) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _registered_hashes(root: Path) -> Dict[str, str]:
    """Read hash registrations without assuming one historical audit schema."""
    result: Dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        if "audit" not in path.name.lower() and path.name not in {"protocol.json", "frozen.json"}:
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    key_text = str(key).lower()
                    if key_text in {"sha256", "hash", "digest"} and isinstance(item, str) and len(item) == 64:
                        result[str(path.resolve())] = item.lower()
                    if isinstance(item, Mapping):
                        candidate = item.get("path") or item.get("file")
                        digest = item.get("sha256") or item.get("hash") or item.get("digest")
                        if candidate and isinstance(digest, str) and len(digest) == 64:
                            resolved = _normalise_path(candidate, path.parent)
                            if resolved:
                                result[str(resolved)] = digest.lower()
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)
    return result


def _hash_registration(path: Path, registrations: Mapping[str, str]) -> Dict[str, Any]:
    digest = registrations.get(str(path.resolve()))
    if digest is None:
        return {"status": "unregistered"}
    if not path.is_file():
        return {"status": "registered_missing", "registered_sha256": digest}
    actual = sha256_file(str(path))
    return {
        "status": "match" if actual == digest else "mismatch",
        "registered_sha256": digest,
        "actual_sha256": actual,
    }


def _checkpoint_identity(path: Path, registrations: Mapping[str, str], step: Optional[int] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "registration": _hash_registration(path, registrations)}
    if not path.is_file():
        result["status"] = "missing"
        return result
    try:
        digest = validate_checkpoint_file(
            str(path), require_checksum=True, require_metadata=True, expected_step=step)
    except (OSError, ValueError) as exc:
        result["status"] = "damaged_or_unverifiable"
        result["error"] = str(exc)
        result["sha256"] = sha256_file(str(path))
        return result
    result.update({"status": "verified", "sha256": digest, "size_bytes": path.stat().st_size})
    if result["registration"].get("status") == "mismatch":
        result["status"] = "identity_mismatch"
    return result


def _status_values(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        if "status" in value:
            yield str(value["status"])
        for item in value.values():
            yield from _status_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _status_values(item)


def _find_checkpoint(run: Path, step: int) -> Optional[Path]:
    names = ("*_checkpoint_%d.pt" % step, "*_checkpoint_%d.pth" % step)
    candidates = []
    for name in names:
        candidates.extend(run.glob("snapshots/" + name))
        candidates.extend(run.glob(name))
    return sorted({p.resolve() for p in candidates})[0] if candidates else None


def inventory_artifacts(
    experiment_root: os.PathLike[str] | str,
    *,
    arms: Optional[Iterable[str]] = None,
    datasets: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    steps: Optional[Iterable[int]] = None,
    project: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Inventory expected runs and distinguish missing from damaged artifacts."""
    root = Path(experiment_root).resolve()
    selected_steps = [int(s) for s in (steps if steps is not None else EXPECTED_STEPS)]
    registrations = _registered_hashes(root)
    runs: List[Dict[str, Any]] = []
    expected_files = ("val_metrics.csv", "best_snapshot_p1.json")
    for dataset, seed, arm, run in run_matrix(root, arms=arms, datasets=datasets, seeds=seeds):
        artifacts: Dict[str, Any] = {}
        for name in expected_files:
            path = run / name
            artifacts[name] = {**file_identity(path), "registration": _hash_registration(path, registrations)}
        for name in ("audit.json", "config_resolved.json", "resolved_config.json", "request_config.json",
                     "actual_model_config.json", "command.txt", "environment.txt", "git_info.txt",
                     "config_hash.txt"):
            path = run / name
            if path.exists() or name in ("audit.json", "config_resolved.json", "resolved_config.json"):
                artifacts[name] = {**file_identity(path), "registration": _hash_registration(path, registrations)}
        snapshots = []
        for step in selected_steps:
            path = _find_checkpoint(run, step)
            snapshots.append({"step": step, **_checkpoint_identity(path, registrations, step)
                              } if path else {"step": step, "status": "missing", "path": None})
        artifacts["checkpoints"] = snapshots
        statuses = list(_status_values(artifacts))
        if not run.is_dir():
            status = "missing_run"
        elif any(item in {"damaged_or_unverifiable", "identity_mismatch", "mismatch", "unverifiable"} for item in statuses):
            status = "damaged_or_identity_mismatch"
        elif any(item in {"missing", "registered_missing"} for item in statuses):
            status = "incomplete"
        else:
            status = "complete_or_partially_unregistered"
        runs.append({
            "dataset": dataset,
            "dataset_display": "SECOND-CC-AUG" if dataset == "second_cc" else dataset.upper().replace("_", "-"),
            "seed": seed,
            "arm": arm,
            "run": str(run),
            "historical_source": _historical_source_identity(run),
            "artifacts": artifacts,
            "status": status,
        })
    counts: Dict[str, int] = {}
    for item in runs:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "kind": "artifact_inventory",
        "experiment_root": str(root),
        "protocol_id": PROTOCOL,
        "filters": {"arms": list(arms) if arms is not None else list(ARMS),
                     "datasets": list(datasets) if datasets is not None else list(DATASETS),
                     "seeds": list(seeds) if seeds is not None else list(SEEDS),
                     "steps": selected_steps},
        "audit_registrations": len(registrations),
        "diagnostic_source": source_identity(project or Path(__file__).resolve().parents[1]),
        "runs": runs,
        "summary": counts,
    }


def _historical_source_identity(run: Path) -> Dict[str, Any]:
    """Read recorded historical identity; never replace it with current git."""
    result: Dict[str, Any] = {}
    for name in ("git_info.txt", "environment.txt", "command.txt", "config_hash.txt"):
        path = run / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                result[name] = {"path": str(path), "sha256": sha256_file(str(path)), "preview": text[:2000]}
            except OSError as exc:
                result[name] = {"path": str(path), "status": "unverifiable", "error": str(exc)}
        else:
            result[name] = {"path": str(path), "status": "missing"}
    for name in ("config_resolved.json", "resolved_config.json", "request_config.json", "actual_model_config.json"):
        path = run / name
        if path.is_file():
            result[name] = file_identity(path)
    return result


def finite_metric(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("missing/non-numeric %s in validation row" % name) from exc
    if not math.isfinite(converted):
        raise ValueError("non-finite %s in validation row" % name)
    if converted < 0:
        raise ValueError("negative %s in validation row" % name)
    return converted


def parse_validation_curve(
    csv_path: os.PathLike[str] | str,
    *,
    expected_steps: Sequence[int] = EXPECTED_STEPS,
) -> Dict[str, Any]:
    """Parse every non-empty row and report all structural failures explicitly."""
    path = Path(csv_path)
    result: Dict[str, Any] = {"path": str(path.resolve()), "status": "missing", "rows": [],
                              "ignored_empty_rows": [], "errors": [], "warnings": []}
    if not path.is_file():
        result["errors"].append("file missing")
        return result
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        result["status"] = "unverifiable"
        result["errors"].append(str(exc))
        return result
    with handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing_fields = [name for name in ("iter", "snapshot_path", *METRICS) if name not in fields]
        if missing_fields:
            result["errors"].append("missing columns: " + ", ".join(missing_fields))
        for row_number, row in enumerate(reader, start=2):
            def has_text(value: Any) -> bool:
                if isinstance(value, (list, tuple)):
                    return any(has_text(item) for item in value)
                return bool(str(value or "").strip())

            if not any(has_text(value) for value in row.values()):
                result["ignored_empty_rows"].append(row_number)
                continue
            try:
                iteration = int(row.get("iter", ""))
            except (TypeError, ValueError):
                result["errors"].append("row %d has invalid iter" % row_number)
                continue
            parsed: Dict[str, Any] = {"row": row_number, "iter": iteration,
                                      "snapshot_path": str(row.get("snapshot_path", ""))}
            try:
                parsed["metrics"] = {metric: finite_metric(row.get(metric, ""), metric) for metric in METRICS}
            except ValueError as exc:
                result["errors"].append("row %d: %s" % (row_number, exc))
                continue
            result["rows"].append(parsed)
    steps = [int(row["iter"]) for row in result["rows"]]
    duplicates = sorted({step for step in steps if steps.count(step) > 1})
    missing = sorted(set(int(s) for s in expected_steps) - set(steps))
    extras = sorted(set(steps) - set(int(s) for s in expected_steps))
    result.update({"steps": steps, "duplicates": duplicates, "missing_steps": missing, "extra_steps": extras})
    if duplicates:
        result["errors"].append("duplicate validation steps: %s" % duplicates)
    if missing:
        result["errors"].append("missing validation steps: %s" % missing)
    if extras:
        result["errors"].append("unexpected validation steps: %s" % extras)
    result["status"] = "ok" if not result["errors"] and sorted(steps) == sorted(int(s) for s in expected_steps) else "invalid"
    return result


def _resolve_snapshot(run: Path, snapshot_path: str, expected_iter: int) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    snapshot_dir = (run / "snapshots").resolve()
    candidates = [Path(snapshot_path), run / snapshot_path, snapshot_dir / Path(snapshot_path).name]
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = run / candidate
        if not candidate.is_file() or candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        try:
            if os.path.commonpath((str(snapshot_dir), str(resolved))) != str(snapshot_dir):
                continue
        except ValueError:
            continue
        match = re.search(r"_checkpoint_(\d+)\.(?:pt|pth)$", resolved.name)
        if not match or int(match.group(1)) != expected_iter:
            continue
        try:
            digest = validate_checkpoint_file(str(resolved), require_checksum=True,
                                              require_metadata=True, expected_step=expected_iter)
        except (OSError, ValueError):
            continue
        return resolved, digest, str(resolved.stat().st_size)
    return None, None, None


def recompute_selection(
    curve: Mapping[str, Any],
    run: os.PathLike[str] | str,
    *,
    reference: Optional[Mapping[str, float]] = None,
    require_checkpoints: bool = True,
) -> Dict[str, Any]:
    """Reproduce the frozen selector's exact score and early-step tie-break."""
    reference = dict(reference or {"Bleu_4": 0.4375, "METEOR": 0.3377,
                                   "ROUGE_L": 0.6942, "CIDEr": 1.2299, "SPICE": 0.2607})
    result: Dict[str, Any] = {"rule": "five_metric_equal_weight_log", "reference": reference,
                              "tie_break": "earlier_step", "status": "invalid", "candidates": [], "errors": []}
    if curve.get("errors"):
        result["errors"].extend(curve["errors"])
        return result
    run_path = Path(run).resolve()
    for row in curve.get("rows", []):
        candidate = dict(row)
        candidate["snapshot_path_original"] = row.get("snapshot_path", "")
        path, digest, size = _resolve_snapshot(run_path, str(row.get("snapshot_path", "")), int(row["iter"]))
        if require_checkpoints and path is None:
            result["errors"].append("row %s has no valid checksummed snapshot for iter %s" % (row.get("row"), row.get("iter")))
            continue
        if path is not None:
            candidate.update({"snapshot_path": str(path), "snapshot_sha256": digest,
                              "snapshot_size_bytes": int(size) if size is not None else None})
        values = row["metrics"]
        candidate["score"] = sum(math.log(max(values[name], 1e-12) / reference[name]) for name in METRICS) / len(METRICS)
        result["candidates"].append(candidate)
    if result["errors"]:
        return result
    if not result["candidates"]:
        result["errors"].append("no validation rows")
        return result
    best = sorted(result["candidates"], key=lambda item: (-item["score"], item["iter"]))[0]
    result["status"] = "ok"
    result["best"] = best
    result["selected_step"] = best["iter"]
    return result


def compare_selection(recomputed: Mapping[str, Any], selection_path: os.PathLike[str] | str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(Path(selection_path).resolve()), "status": "missing", "mismatches": []}
    path = Path(selection_path)
    if not path.is_file():
        result["mismatches"].append("selection JSON missing")
        return result
    try:
        recorded = read_json(path)
    except (OSError, ValueError) as exc:
        result["status"] = "unverifiable"
        result["mismatches"].append(str(exc))
        return result
    result["recorded_sha256"] = sha256_file(str(path))
    if recomputed.get("status") != "ok":
        result["status"] = "not_comparable"
        result["mismatches"].append("recomputed selection is not valid")
        return result
    best = recomputed["best"]
    if recorded.get("protocol_id") != PROTOCOL and recorded.get("protocol_id") != "p1_rsaca_20260913":
        result["mismatches"].append("protocol_id mismatch")
    recorded_best = recorded.get("best", {})
    for key in ("iter", "score"):
        left = recorded_best.get(key)
        right = best.get(key)
        if key == "score" and left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
            continue
        if left != right:
            result["mismatches"].append("best.%s differs: recorded=%r recomputed=%r" % (key, left, right))
    recorded_path = _normalise_path(recorded.get("best_snapshot"), path.parent)
    if recorded_path and Path(best.get("snapshot_path", "")).resolve() != recorded_path:
        result["mismatches"].append("best_snapshot path identity differs")
    if recorded.get("best_snapshot_sha256") and best.get("snapshot_sha256"):
        if recorded["best_snapshot_sha256"] != best["snapshot_sha256"]:
            result["mismatches"].append("best_snapshot_sha256 differs")
    result["status"] = "match" if not result["mismatches"] else "mismatch"
    return result


def _metrics_from_selection(selection: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    try:
        values = selection["best"]["metrics"]
        return {metric: finite_metric(values[metric], metric) for metric in METRICS}
    except (KeyError, TypeError, ValueError):
        return None


def curve_report(inventory: Mapping[str, Any], *, require_checkpoints: bool = True) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    run_reports = []
    for item in inventory.get("runs", []):
        run = Path(item["run"])
        curve = parse_validation_curve(run / "val_metrics.csv")
        selection = recompute_selection(curve, run, require_checkpoints=require_checkpoints)
        selection_check = compare_selection(selection, run / "best_snapshot_p1.json")
        if selection.get("status") == "ok":
            metrics = _metrics_from_selection(selection)
            if metrics is not None:
                rows.append({"dataset": item["dataset"], "seed": item["seed"], "arm": item["arm"], "metrics": metrics,
                             "source": str(run / "val_metrics.csv"), "step": selection["selected_step"]})
        run_reports.append({"dataset": item["dataset"], "seed": item["seed"], "arm": item["arm"],
                            "run": str(run), "curve": curve, "selection": selection,
                            "selection_check": selection_check})
    # These comparisons use the same validation step across paired seeds and
    # are explicitly separate from the frozen selected-checkpoint result.
    curve_values: Dict[Tuple[str, int, str, int], Dict[str, float]] = {}
    for report in run_reports:
        for row in report["curve"].get("rows", []):
            if not report["curve"].get("errors"):
                curve_values[(report["dataset"], int(row["iter"]), report["arm"], int(report["seed"]))] = row["metrics"]
    same_step: Dict[str, Any] = {}
    for label, candidate, baseline in COMPARISON_PAIRS:
        dataset_payload: Dict[str, Any] = {}
        for dataset in DATASETS:
            step_payload: Dict[str, Any] = {}
            for step in EXPECTED_STEPS:
                seeds = sorted(set(seed for (d, s, arm, seed) in curve_values
                                    if d == dataset and s == step and arm == candidate) &
                               set(seed for (d, s, arm, seed) in curve_values
                                   if d == dataset and s == step and arm == baseline))
                if not seeds:
                    continue
                differences = {seed: {metric: curve_values[dataset, step, candidate, seed][metric] -
                                      curve_values[dataset, step, baseline, seed][metric]
                                      for metric in METRICS} for seed in seeds}
                step_payload[str(step)] = {
                    "seeds": seeds,
                    "per_seed_difference": differences,
                    "mean": {metric: statistics.mean(differences[seed][metric] for seed in seeds) for metric in METRICS},
                    "sample_std": {metric: statistics.stdev([differences[seed][metric] for seed in seeds])
                                   if len(seeds) > 1 else None for metric in METRICS},
                    "positive_seed_count": {metric: sum(differences[seed][metric] > 0 for seed in seeds)
                                             for metric in METRICS},
                }
            if step_payload:
                dataset_payload[dataset] = step_payload
        same_step[label] = {"direction": "%s minus %s" % (candidate, baseline),
                            "unit": "paired seed at identical validation step", "datasets": dataset_payload}
    trajectory: Dict[str, Any] = {}
    for dataset in DATASETS:
        for arm in ARMS:
            step_payload = {}
            for step in EXPECTED_STEPS:
                seeds = sorted(seed for (d, s, a, seed) in curve_values if d == dataset and s == step and a == arm)
                if not seeds:
                    continue
                step_payload[str(step)] = {
                    "seeds": seeds,
                    "mean": {metric: statistics.mean(curve_values[dataset, step, arm, seed][metric] for seed in seeds)
                              for metric in METRICS},
                    "sample_std": {metric: statistics.stdev([curve_values[dataset, step, arm, seed][metric] for seed in seeds])
                                   if len(seeds) > 1 else None for metric in METRICS},
                }
            if step_payload:
                trajectory.setdefault(dataset, {})[arm] = step_payload
    try:
        from utils.semantic_control_audit import statistics_report
        statistics_payload = statistics_report(rows)
    except (ImportError, ValueError) as exc:
        statistics_payload = {"status": "unavailable", "error": str(exc)}
    comparisons: Dict[str, Any] = {}
    if isinstance(statistics_payload, Mapping):
        for label, candidate, baseline in COMPARISON_PAIRS:
            per_dataset = {}
            for dataset in DATASETS:
                paired = statistics_payload.get("datasets", {}).get(dataset, {}).get("paired", {}).get(candidate + "-" + baseline)
                if paired:
                    per_dataset[dataset] = paired
            comparisons[label] = {"direction": "%s minus %s" % (candidate, baseline), "datasets": per_dataset}
    return {"schema": DIAGNOSTIC_SCHEMA, "kind": "validation_curves", "runs": run_reports,
            "selected_rows": rows, "statistics": statistics_payload, "comparisons": comparisons,
            "same_step_comparisons": same_step,
            "late_trajectory": {"status": "exploratory", "unit": "seed; validation steps are not independent repetitions",
                                "datasets": trajectory},
            "selection_rule": "five_metric_equal_weight_log", "legacy_balanced_score_used": False,
            "historical_acceptance_fields_used_for_selection": False,
            "split": "validation", "seed_count": len(SEEDS)}


def _prediction_rows(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("predictions", "results", "annotations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("prediction file must be a list or contain predictions/results")


def _reference_ids(payload: Any) -> List[str]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("annotations"), list):
        raise ValueError("reference file must contain annotations")
    return [str(row["image_id"]) for row in payload["annotations"] if isinstance(row, Mapping) and "image_id" in row]


def validate_prediction_ids(
    prediction_path: os.PathLike[str] | str,
    reference_path: os.PathLike[str] | str,
    expected_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    prediction = _prediction_rows(read_json(prediction_path))
    reference = _reference_ids(read_json(reference_path))
    predicted_ids = [str(row.get("image_id")) for row in prediction]
    prediction_missing_id = [index for index, row in enumerate(prediction) if not row.get("image_id")]
    reference_payload = read_json(reference_path)
    reference_rows = reference_payload.get("annotations", []) if isinstance(reference_payload, Mapping) else []
    reference_missing_id = [index for index, row in enumerate(reference_rows)
                            if not isinstance(row, Mapping) or not row.get("image_id")]
    duplicates = sorted({item for item in predicted_ids if predicted_ids.count(item) > 1})
    reference_duplicates = sorted({item for item in reference if reference.count(item) > 1})
    expected = [str(item) for item in (expected_ids if expected_ids is not None else reference)]
    expected_duplicates = sorted({item for item in expected if expected.count(item) > 1})
    missing = sorted(set(expected) - set(predicted_ids))
    extra = sorted(set(predicted_ids) - set(expected))
    reference_missing = sorted(set(predicted_ids) - set(reference))
    status = "ok" if not (prediction_missing_id or reference_missing_id or duplicates or reference_duplicates
                           or expected_duplicates or missing or extra or reference_missing) else "invalid"
    return {"status": status, "prediction_count": len(predicted_ids), "reference_count": len(reference),
            "expected_count": len(expected), "duplicates": duplicates, "missing_ids": missing,
            "extra_ids": extra, "prediction_ids_not_in_reference": reference_missing,
            "prediction_missing_id_rows": prediction_missing_id,
            "reference_missing_id_rows": reference_missing_id,
            "reference_duplicates": reference_duplicates,
            "expected_duplicates": expected_duplicates,
            "prediction_sha256": sha256_file(str(prediction_path)), "reference_sha256": sha256_file(str(reference_path)),
            "sample_ids_sha256": stable_hash(sorted(predicted_ids))}


def _render_command(command: Sequence[str], prediction: Path, reference: Path, output: Path) -> List[str]:
    # Replace only the documented placeholders; scorer arguments may contain
    # ordinary JSON braces which must not be interpreted as format fields.
    replacements = {
        "{prediction}": str(prediction),
        "{reference}": str(reference),
        "{output}": str(output),
    }
    rendered = []
    for argument in command:
        value = str(argument)
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        rendered.append(value)
    return rendered


def _parse_score_output(text: str) -> Dict[str, float]:
    candidates = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
        candidates.extend(line.strip() for line in stripped.splitlines()[::-1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(payload, Mapping):
            source = payload.get("metrics", payload)
            if all(metric in source for metric in METRICS):
                return {metric: finite_metric(source[metric], metric) for metric in METRICS}
    raise ValueError("scorer output did not contain all five metrics as JSON")


def _java_identity() -> Dict[str, Any]:
    executable = shutil.which("java")
    if not executable:
        return {"available": False, "status": "missing"}
    try:
        process = subprocess.run([executable, "-version"], capture_output=True, text=True, check=False)
        text = (process.stdout or "") + (process.stderr or "")
        return {"available": True, "executable": executable, "returncode": process.returncode,
                "version": text.splitlines()[:3], "status": "ok" if process.returncode == 0 else "error"}
    except OSError as exc:
        return {"available": True, "executable": executable, "status": "error", "error": str(exc)}


def reproduce_scores(
    prediction_path: os.PathLike[str] | str,
    reference_path: os.PathLike[str] | str,
    *,
    expected_ids: Optional[Sequence[str]] = None,
    scorer_command: Optional[Sequence[str]] = None,
    original_score_path: Optional[os.PathLike[str] | str] = None,
    output_dir: Optional[os.PathLike[str] | str] = None,
    tolerance: float = 1e-12,
    spice: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    prediction = Path(prediction_path).resolve()
    reference = Path(reference_path).resolve()
    identity = validate_prediction_ids(prediction, reference, expected_ids)
    result: Dict[str, Any] = {"kind": "score_reproduction", "status": "blocked" if identity["status"] != "ok" else "not_run",
                              "prediction": str(prediction), "reference": str(reference), "identity": identity,
                              "scorer": {"version": "unavailable", "command": list(scorer_command or []),
                                         "spice": bool(spice), "java": _java_identity() if spice else {"status": "not_requested"}},
                              "tolerance": tolerance, "issues": []}
    if identity["status"] != "ok":
        result["issues"].append("prediction/reference IDs are not an exact match; no intersection fallback was used")
        return result
    if original_score_path:
        original = Path(original_score_path)
        if original.is_file():
            try:
                payload = read_json(original)
                values = payload.get("metrics", payload)
                result["original"] = {"path": str(original), "sha256": sha256_file(str(original)),
                                       "metrics": {metric: finite_metric(values[metric], metric) for metric in METRICS}}
            except (OSError, KeyError, TypeError, ValueError) as exc:
                result["original"] = {"path": str(original), "status": "unverifiable", "error": str(exc)}
        else:
            result["original"] = {"path": str(original), "status": "missing"}
    if not scorer_command:
        result["issues"].append("no scorer command supplied; historical scores were not replaced")
        return result
    if dry_run:
        planned_output = Path(output_dir or "<diagnostic-output>").resolve() / "scorer.stdout.txt"
        result["scorer"]["command"] = _render_command(scorer_command, prediction, reference, planned_output)
        result["status"] = "dry_run"
        result["issues"].append("scorer command was planned but not executed")
        return result
    if output_dir is None:
        raise ValueError("output_dir is required when running a scorer")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scorer_output = output / "scorer.stdout.txt"
    command = _render_command(scorer_command, prediction, reference, scorer_output)
    started = time.time()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    scorer_output.write_text(process.stdout, encoding="utf-8")
    result["scorer"].update({"command": command, "returncode": process.returncode,
                              "elapsed_seconds": time.time() - started,
                              "stdout_sha256": sha256_file(str(scorer_output))})
    if process.returncode != 0:
        result["status"] = "scorer_error"
        result["issues"].append("scorer returned non-zero status")
        result["scorer"]["stderr"] = process.stderr[-4000:]
        return result
    try:
        metrics = _parse_score_output(process.stdout)
    except ValueError as exc:
        result["status"] = "parse_error"
        result["issues"].append(str(exc))
        return result
    result["recomputed"] = {"metrics": metrics, "stdout_path": str(scorer_output)}
    if "original" in result and "metrics" in result["original"]:
        result["absolute_difference"] = {metric: abs(metrics[metric] - result["original"]["metrics"][metric]) for metric in METRICS}
        result["status"] = "match" if all(delta <= tolerance for delta in result["absolute_difference"].values()) else "different"
    else:
        result["status"] = "new_environment_recomputed"
        result["issues"].append("original scoring environment/result is unavailable; this is a new-environment recomputation")
    return result


def capture_runtime_identity() -> Dict[str, Any]:
    result = {"python": sys.version, "platform": sys.platform}
    try:
        import numpy as np
        result["numpy"] = np.__version__
    except Exception:
        result["numpy"] = "unavailable"
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        result["torch"] = "unavailable"
        result["cuda_available"] = False
    return result


def compare_prediction_runs(first_path: os.PathLike[str] | str, second_path: os.PathLike[str] | str,
                            *, metrics_first: Optional[Mapping[str, float]] = None,
                            metrics_second: Optional[Mapping[str, float]] = None,
                            tolerance: float = 1e-12) -> Dict[str, Any]:
    first = _prediction_rows(read_json(first_path))
    second = _prediction_rows(read_json(second_path))
    def text(row: Mapping[str, Any]) -> Any:
        return row.get("caption", row.get("text", row.get("prediction", "")))

    first_map = {str(row.get("image_id")): text(row) for row in first}
    second_map = {str(row.get("image_id")): text(row) for row in second}
    ids_equal = list(first_map) == list(second_map)
    text_equal = ids_equal and all(first_map[key] == second_map[key] for key in first_map)
    result = {"ids_equal": ids_equal, "text_equal": text_equal,
              "missing_in_second": sorted(set(first_map) - set(second_map)),
              "extra_in_second": sorted(set(second_map) - set(first_map)),
              "metric_tolerance": tolerance}
    if metrics_first is not None and metrics_second is not None:
        result["metric_absolute_difference"] = {metric: abs(float(metrics_first[metric]) - float(metrics_second[metric]))
                                                 for metric in METRICS}
        result["metrics_equal_within_tolerance"] = all(value <= tolerance for value in result["metric_absolute_difference"].values())
    return result


def _caption_text(row: Mapping[str, Any]) -> str:
    caption = row.get("caption", row.get("text", row.get("prediction", "")))
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    return str(caption)


def _text_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    texts = [_caption_text(row) for row in rows]
    lengths = [len(text.split()) for text in texts]
    unigrams = [word.lower() for text in texts for word in re.findall(r"[A-Za-z0-9_]+", text)]
    repeated = sum(1 for text in texts if len(text.split()) > 1 and len(set(text.lower().split())) < len(text.split()))
    return {"count": len(texts), "mean_length": statistics.mean(lengths) if lengths else None,
            "sample_std_length": statistics.stdev(lengths) if len(lengths) > 1 else None,
            "min_length": min(lengths) if lengths else None, "max_length": max(lengths) if lengths else None,
            "repeated_expression_count": repeated,
            "empty_caption_count": sum(not text.strip() for text in texts),
            "token_count": len(unigrams)}


def select_fixed_samples(sample_ids: Sequence[str], *, seed: int, count: int,
                         failure_ids: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Select failure-focused and overview samples deterministically."""
    if count < 1:
        raise ValueError("count must be positive")
    all_ids = sorted({str(item) for item in sample_ids})
    failures = [item for item in sorted({str(x) for x in (failure_ids or [])}) if item in all_ids]
    failure_selection = failures[:count]
    remaining = [item for item in all_ids if item not in set(failure_selection)]
    randomiser = random.Random(int(seed))
    randomiser.shuffle(remaining)
    return {"failure_cases": failure_selection, "overview_sample": sorted(remaining[:count])}


def analyze_content(prediction_paths: Mapping[str, os.PathLike[str] | str], *, sample_seed: int = 1111,
                    sample_count: int = 16, failure_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    loaded: Dict[str, List[Mapping[str, Any]]] = {}
    issues = []
    for arm, path in prediction_paths.items():
        try:
            loaded[arm] = _prediction_rows(read_json(path))
        except (OSError, ValueError) as exc:
            issues.append({"arm": arm, "status": "unverifiable", "error": str(exc)})
    id_sets = {arm: [str(row.get("image_id")) for row in rows] for arm, rows in loaded.items()}
    missing_ids = {arm: [index for index, row in enumerate(rows) if not row.get("image_id")]
                   for arm, rows in loaded.items()}
    duplicate_ids = {arm: sorted({item for item in ids if ids.count(item) > 1}) for arm, ids in id_sets.items()}
    distinct_sets = {arm: set(ids) for arm, ids in id_sets.items()}
    same_ids = bool(distinct_sets) and len({frozenset(ids) for ids in distinct_sets.values()}) == 1
    if any(duplicate_ids.values()):
        issues.append({"status": "invalid", "reason": "duplicate sample IDs", "duplicates": duplicate_ids})
    if any(missing_ids.values()):
        issues.append({"status": "invalid", "reason": "missing sample IDs", "missing": missing_ids})
    if not same_ids:
        issues.append({"status": "invalid", "reason": "prediction sample ID sets differ"})
    common = sorted(set.intersection(*(set(ids) for ids in id_sets.values()))) if id_sets else []
    selections = select_fixed_samples(common, seed=sample_seed, count=sample_count, failure_ids=failure_ids) if common else {"failure_cases": [], "overview_sample": []}
    return {"kind": "content_analysis", "status": "ok" if loaded and not issues and len(common) else "incomplete",
            "arms": {arm: {"path": str(Path(path).resolve()), "sha256": sha256_file(str(path)),
                           "text": _text_stats(loaded[arm])}
                     for arm, path in prediction_paths.items() if arm in loaded},
            "id_counts": id_sets, "missing_id_rows": missing_ids, "duplicate_ids": duplicate_ids, "same_id_set": same_ids,
            "common_ids": common, "samples": selections, "issues": issues,
            "interpretation": "Text statistics are descriptive; they do not establish causality or model superiority."}


def analyze_fusion(records_by_arm: Mapping[str, os.PathLike[str] | str]) -> Dict[str, Any]:
    result = {"kind": "fusion_analysis", "arms": {}, "issues": []}
    for arm, path in records_by_arm.items():
        try:
            payload = read_json(path)
        except (OSError, ValueError) as exc:
            result["issues"].append({"arm": arm, "status": "unverifiable", "error": str(exc)})
            continue
        records = payload.get("records", []) if isinstance(payload, Mapping) else []
        if not records:
            result["arms"][arm] = {"status": "not_applicable", "reason": "no fusion hook records"}
            continue
        if all(row.get("hook_status") == "not_applicable" for row in records):
            result["arms"][arm] = {
                "status": "not_applicable",
                "record_count": len(records),
                "reason": "no SemanticCrossAttentionFusion module enabled",
            }
            continue
        keys = ("gamma", "gate", "coverage", "relative_residual", "residual_l2", "query_l2")
        present = {key: [row[key] for row in records if key in row] for key in keys}
        stats = {}
        for key, values in present.items():
            flattened = []
            for value in values:
                if isinstance(value, list):
                    flattened.extend(float(x) for x in value if isinstance(x, (int, float)))
                elif isinstance(value, (int, float)):
                    flattened.append(float(value))
            if flattened:
                stats[key] = {"count": len(flattened), "mean": statistics.mean(flattened),
                              "sample_std": statistics.stdev(flattened) if len(flattened) > 1 else None}
        result["arms"][arm] = {"status": "ok", "record_count": len(records), "available": sorted(stats), "stats": stats,
                                "relative_residual_definition": "||fusion_after-fusion_before||_2 / max(||fusion_before||_2, 1e-12)"}
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    inventory = report.get("inventory", {})
    curves = report.get("curves", {})
    lines = ["# 语义控制诊断报告", "",
             "> 本报告只复算已有工件并记录可验证缺口，不启动训练、不修改历史实验目录，也不把行为诊断当作重新训练消融。", "",
             "## 输入与身份", "",
             "- 实验根目录：`%s`" % inventory.get("experiment_root", "未提供"),
             "- 协议：`%s`" % inventory.get("protocol_id", PROTOCOL),
             "- 诊断工具身份：`%s`" % json.dumps(inventory.get("diagnostic_source", {}), ensure_ascii=False, sort_keys=True),
             "- 历史运行身份：逐运行保存在机器可读报告的 `historical_source`，不使用当前 commit 冒充。", "",
             "## 工件完整性", ""]
    for status, count in sorted(inventory.get("summary", {}).items()):
        lines.append("- `%s`：%s 个运行" % (status, count))
    lines.extend(["", "缺失 checkpoint 不会阻止 CSV、预测和曲线的独立分析；损坏或身份不匹配不会被记为通过。", "",
                  "## 原协议选点与统计", "",
                  "- 验证划分：`validation`；网格：`1000..10000`，步长 `1000`；选择：五指标等权对数，平分选较早步。",
                  "- 比较方向：`D-B`、`C0-B`、`C0-D`、`C0-G`，均为前者减后者。",
                  "- 统计使用 seed 作为重复单位，均值和样本标准差使用 `ddof=1`；不同训练步数不作为独立 seed。",
                  "- 选点不使用旧 `balanced_score` 或历史验收字段；机器报告中以布尔字段显式记录。",
                  "- `same_step_comparisons` 是相同步数的逐 seed 配对对照；`late_trajectory` 仅为训练后期探索性轨迹，不能改写原协议选点。", ""])
    statistics_payload = curves.get("statistics", {})
    if statistics_payload.get("mean_goal") is not None:
        lines.append("- 当前可复算的 C0-B 均值 15/15 门槛：`%s`。" % ("通过" if statistics_payload["mean_goal"] else "未通过"))
    else:
        lines.append("- 当前无法从筛选工件完整复算 36 运行的 15/15 均值门槛。")
    lines.extend(["", "## 评分复现", "", "评分复现结果位于机器可读报告的 `scoring`；ID 缺失、重复或多余会明确阻止评分，不取交集默认为通过。", "",
                  "## 推理复现", "", "未显式执行 `infer` 时状态为未运行；真实 checkpoint 推理只允许验证集、eval、greedy，并记录随机状态和固定容差。", "",
                  "## 描述与融合行为", "", "文本长度/重复表达和融合 hook 只作描述性证据；未启用模块标为不适用，不用 0 冒充。", "",
                  "## 事实、解释与待验证假设", "",
                  "- 已确认事实：报告中 `status=verified/match/ok` 的文件、曲线和身份检查。",
                  "- 间接解释：同一步配对曲线和文本/融合统计，仅能支持相关性描述。",
                  "- 待验证假设：必须补充同 checkpoint 重复推理、评分环境核对，或新的验证集对照；本报告不会自动选择 `context_pre_norm` 或 `changed_mean`。", "",
                  "## 证据边界", "",
                  "当前流程是否具备提出模型改动的证据由机器可读字段 `evidence_assessment` 给出；本轮不把训练完成、验证准入失败的历史记录改写成正式测试通过。", ""])
    return "\n".join(lines)
