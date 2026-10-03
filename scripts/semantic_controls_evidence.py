#!/usr/bin/env python3
"""Read-only inventory, lightweight export and offline validation evidence.

This tool never edits the historical experiment directory.  Validation uses
the reference stored in each historical selection record; it never silently
uses a current default when that evidence is absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.semantic_controls import ARMS, DATASETS, METRICS, SEEDS, PROTOCOL
from utils.semantic_diagnostics import EXPECTED_STEPS, parse_validation_curve, recompute_selection

PAIRS = (("D-B", "plain_fusion", "card"), ("C0-B", "rsaca", "card"),
         ("C0-D", "rsaca", "plain_fusion"), ("C0-G", "rsaca", "fixed_gate"))
REQUIRED_RUN_FILES = (
    "run_summary.json", "val_metrics.csv", "best_snapshot_p1.json",
    "request_config.json", "resolved_config.json", "actual_model_config.json",
    "initial_parameter_summary.json", "controls_provenance.json",
    "validation_prediction_identity.json", "command.txt", "environment.txt",
    "git_info.txt", "execution_ledger.json")
ROOT_FILES = ("protocol.json", "audit.json", "inputs_levir_cc.json",
              "inputs_levir_mci.json", "inputs_second_cc.json")


def digest(path: Path):
    if not path.is_file():
        return {"status": "missing", "source": str(path)}
    h = hashlib.sha256(); size = 0
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                size += len(block); h.update(block)
    except OSError as exc:
        return {"status": "unverifiable", "source": str(path), "error": str(exc)}
    return {"status": "present", "source": str(path), "size_bytes": size, "sha256": h.hexdigest()}


def run_path(root, dataset, seed, arm):
    return root / arm / (f"{arm}_{dataset}_seed{seed}")


def inventory(root: Path):
    root = root.resolve()
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                run = run_path(root, dataset, seed, arm)
                files = {name: digest(run / name) for name in REQUIRED_RUN_FILES}
                try:
                    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
                    status = summary.get("status", "unknown")
                except (OSError, ValueError) as exc:
                    status, files["run_summary.json"]["error"] = "unverifiable", str(exc)
                records.append({"dataset": dataset, "arm": arm, "seed": seed,
                                "run": str(run), "status": status if run.is_dir() else "missing_run",
                                "formal": True, "files": files})
    extras = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if path.is_dir() and path.name not in ARMS:
            extras.append({"path": str(path), "kind": "archive_or_failure", "files": digest_tree(path)})
    return {"schema": "semantic_controls_evidence.inventory.v1", "protocol_id": PROTOCOL,
            "experiment_root": str(root), "expected_formal_runs": 36, "formal_runs": records,
            "nonformal_directories": extras,
            "root_files": {name: digest(root / name) for name in ROOT_FILES},
            "summary": {"formal_present": sum(r["status"] != "missing_run" for r in records),
                        "formal_completed": sum(r["status"] == "completed" for r in records),
                        "nonformal_directories": len(extras)}}


def digest_tree(path: Path):
    # Archives can contain hundreds of megabytes of checkpoint payloads.  The
    # inventory classifies these separately and records file names/sizes only.
    return [{"relative_path": str(p.relative_to(path)), "size_bytes": p.stat().st_size,
             "checkpoint_payload_not_hashed": p.suffix in (".pt", ".pth")}
            for p in sorted(path.rglob("*")) if p.is_file()]


def guard_output(root: Path, output: Path):
    if output.is_symlink():
        raise ValueError("output path may not be a symlink")
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside experiment root")
    if output.exists() and (output.is_symlink() or output.is_file() or any(output.iterdir())):
        raise ValueError("output must be a new empty directory and may not be a symlink")


def export(root: Path, output: Path, *, dry_run=False, include_predictions=False):
    guard_output(root, output)
    inv = inventory(root)
    selected = []
    for name in ROOT_FILES:
        selected.append(root / name)
    selected.extend(p for p in root.iterdir() if p.is_file() and
                    (p.suffix.lower() == ".log" or p.name.endswith("_log.txt")))
    for rec in inv["formal_runs"]:
        run = Path(rec["run"])
        for name in REQUIRED_RUN_FILES:
            selected.append(run / name)
        for name in ("best_snapshot_p1.json",):
            selected.append(run / name)
        selected.extend(run.glob("snapshots/*.sha256"))
        selected.extend(run.glob("snapshots/*.metadata.json"))
        if include_predictions:
            selected.extend(p for pattern in ("eval_sents", "eval_gen_samples")
                            for p in run.glob(pattern + "/**/*") if p.is_file())
    # Keep original bytes and paths; no JSON path rewriting is performed.
    manifest = {"schema": "semantic_controls_evidence.export.v1", "tool": "semantic_controls_evidence.py",
                "source_root": str(root.resolve()), "output_root": str(output.resolve()),
                "dry_run": dry_run, "include_predictions": include_predictions, "files": []}
    for source in selected:
        item = digest(source)
        try: rel = source.resolve().relative_to(root.resolve())
        except ValueError: continue
        item.update({"relative_path": str(rel), "copied": False})
        if source.is_file() and item["status"] == "present" and not dry_run:
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            item["copied"] = True
            if digest(target).get("sha256") != item.get("sha256"):
                raise IOError("copied file hash mismatch: %s" % rel)
        manifest["files"].append(item)
    manifest["omitted"] = {"checkpoints": "hash and metadata only; checkpoint bytes are not copied",
                           "datasets_features": "not selected", "large_predictions": not include_predictions}
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "inventory.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def validation(root: Path):
    rows, run_results, issues = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                run = run_path(root, dataset, seed, arm)
                selection_path = run / "best_snapshot_p1.json"
                status = "unverifiable"
                result = {"dataset": dataset, "arm": arm, "seed": seed, "run": str(run), "status": status}
                try:
                    recorded = json.loads(selection_path.read_text(encoding="utf-8"))
                    reference = recorded.get("selection", {}).get("reference")
                    if not isinstance(reference, dict) or any(m not in reference for m in METRICS):
                        raise ValueError("selection reference missing; protocol cannot be independently verified")
                    curve = parse_validation_curve(run / "val_metrics.csv")
                    selected = recompute_selection(curve, run, reference=reference, require_checkpoints=False)
                    if selected["status"] != "ok":
                        raise ValueError("validation grid invalid: " + "; ".join(selected["errors"]))
                    best = selected["best"]
                    recorded_best = recorded.get("best", {})
                    recorded_checkpoint = str(recorded_best.get("snapshot_path", recorded.get("best_snapshot", "")))
                    checkpoint_step_match = re.search(r"_checkpoint_(\d+)\.(?:pt|pth)$", Path(recorded_checkpoint).name)
                    checkpoint_path_step_ok = bool(checkpoint_step_match and int(checkpoint_step_match.group(1)) == int(best["iter"]))
                    ref_hash = hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                    match = (recorded.get("protocol_id") == PROTOCOL and
                             recorded.get("selection_metric_split") == "validation" and
                             recorded.get("selection", {}).get("split") == "validation" and
                             recorded.get("selection", {}).get("rule") == "five_metric_equal_weight_log" and
                             recorded.get("selection", {}).get("tie_break") == "earlier_step" and
                             recorded.get("selection", {}).get("reference_sha256") == ref_hash and
                             recorded_best.get("iter") == best["iter"] and
                             checkpoint_path_step_ok and
                             recorded_best.get("snapshot_path") == recorded.get("best_snapshot") and
                             recorded_best.get("snapshot_sha256") == recorded.get("best_snapshot_sha256") and
                             recorded_best.get("metrics") == best["metrics"] and
                             math.isclose(float(recorded_best.get("score")), float(best["score"]), abs_tol=1e-12, rel_tol=0) and
                             len(recorded.get("candidates", [])) == len(selected["candidates"]))
                    if match:
                        for old, new in zip(recorded["candidates"], selected["candidates"]):
                            if (old.get("iter") != new.get("iter") or old.get("metrics") != new.get("metrics") or
                                    not math.isclose(float(old.get("score")), float(new.get("score")), abs_tol=1e-12, rel_tol=0)):
                                match = False; break
                    status = "match_csv_only" if match else "selection_mismatch"
                    result.update({"status": status, "selected_step": best["iter"], "metrics": best["metrics"],
                                   "score": best["score"], "selection_match": match,
                                   "checkpoint_identity": "not_checked", "checkpoint_path_step_match": checkpoint_path_step_ok,
                                   "checkpoint_content_hash_checked": False})
                    if match:
                        rows.append({"dataset": dataset, "arm": arm, "seed": seed, "step": best["iter"], "metrics": best["metrics"]})
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    result["error"] = str(exc); issues.append(result)
                run_results.append(result)
    paired = {}
    for label, candidate, baseline in PAIRS:
        by_key = {(r["dataset"], r["seed"], r["arm"]): r for r in rows}
        payload = {}
        for dataset in DATASETS:
            diffs = []
            for seed in SEEDS:
                c, b = by_key.get((dataset, seed, candidate)), by_key.get((dataset, seed, baseline))
                if c and b:
                    diffs.append({"seed": seed, **{m: c["metrics"][m] - b["metrics"][m] for m in METRICS}})
            if diffs:
                payload[dataset] = {"seeds": [d["seed"] for d in diffs], "per_seed": diffs,
                                    "mean": {m: statistics.mean(d[m] for d in diffs) for m in METRICS},
                                    "sample_std": {m: statistics.stdev([d[m] for d in diffs]) if len(diffs) > 1 else None for m in METRICS},
                                    "positive_zero_negative": {m: [sum(d[m] > 0 for d in diffs), sum(d[m] == 0 for d in diffs), sum(d[m] < 0 for d in diffs)] for m in METRICS}}
        paired[label] = {"direction": f"{candidate} minus {baseline}", "datasets": payload}
    audit_check = {"status": "missing"}
    audit_path = root / "audit.json"
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_check = {"status": "recorded_report_only", "protocol_audit_passed": audit.get("protocol_audit_passed"),
                           "statistics_match": None, "issues": []}
            historical = audit.get("validation", {}).get("datasets", {})
            if len(rows) == 36:
                for label, candidate, baseline in PAIRS:
                    for dataset in DATASETS:
                        try:
                            old = historical[dataset]["paired"][candidate + "-" + baseline]
                            now = paired[label]["datasets"][dataset]
                            for metric in METRICS:
                                if not math.isclose(float(old["mean"][metric]), float(now["mean"][metric]), abs_tol=1e-10, rel_tol=0):
                                    audit_check["issues"].append("%s/%s/%s mean differs" % (label, dataset, metric))
                        except (KeyError, TypeError):
                            audit_check["issues"].append("%s/%s missing recorded statistic" % (label, dataset))
                audit_check["statistics_match"] = not audit_check["issues"]
        except (OSError, ValueError) as exc:
            audit_check = {"status": "unverifiable", "error": str(exc)}
    return {"schema": "semantic_controls_evidence.validation.v1", "protocol_id": PROTOCOL,
            "experiment_root": str(root.resolve()), "selection_rule": "five_metric_equal_weight_log",
            "evidence_level": "CSV plus historical selection; checkpoint identity not independently checked",
            "runs": run_results, "selected_rows": rows, "paired": paired, "issues": issues,
            "historical_audit_declaration": audit_check,
            "complete_selected_matrix": len(rows) == 36}


def write_validation(payload, output: Path):
    guard_output(Path(payload["experiment_root"]), output); output.mkdir(parents=True)
    (output / "validation_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "validation_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "arm", "seed", "step", *METRICS]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in payload["selected_rows"]: w.writerow({**{k: r[k] for k in ("dataset", "arm", "seed", "step")}, **r["metrics"]})
    with (output / "validation_paired_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["comparison", "dataset", "row_type", "seed", *METRICS,
                  *["mean_" + m for m in METRICS], *["sample_std_" + m for m in METRICS],
                  *[name + "_positive_zero_negative" for name in METRICS]]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for label, item in payload["paired"].items():
            for dataset, d in item["datasets"].items():
                for row in d["per_seed"]: w.writerow({"comparison": label, "dataset": dataset, "row_type": "per_seed", **row})
                summary = {"comparison": label, "dataset": dataset, "row_type": "summary", "seed": "",
                           **{"mean_" + m: d["mean"][m] for m in METRICS},
                           **{"sample_std_" + m: d["sample_std"][m] for m in METRICS},
                           **{m + "_positive_zero_negative": "/".join(map(str, d["positive_zero_negative"][m])) for m in METRICS}}
                w.writerow(summary)
    lines = ["# Validation 独立复算报告", "", "- 证据级别：%s" % payload["evidence_level"],
             "- 完整36运行选点：%s" % ("是" if payload["complete_selected_matrix"] else "否"),
             "- 缺失/冲突运行不会填0，也不会生成冻结凭据。", "", "## 配对统计", "",
             "| 比较 | 数据集 | seeds | 均值(Bleu_4, METEOR, ROUGE_L, CIDEr, SPICE) |", "|---|---|---|---|"]
    for label, item in payload["paired"].items():
        for dataset, d in item["datasets"].items():
            lines.append("| %s | %s | %s | %s |" % (label, dataset, ",".join(map(str, d["seeds"])), ", ".join("%.6f" % d["mean"][m] for m in METRICS)))
    if payload["issues"]: lines += ["", "## 缺口", ""] + ["- %s/%s/seed%s：%s" % (x["dataset"], x["arm"], x["seed"], x.get("error", "")) for x in payload["issues"]]
    (output / "validation_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cc_diagnostic(root: Path, output: Path, sample_count=16):
    """Write CC curves and a reproducible, explicitly unlabelled review sheet."""
    guard_output(root, output); output.mkdir(parents=True)
    runs, samples = [], []
    reference_payload = None
    for seed in SEEDS:
        for arm in ARMS:
            run = run_path(root, "levir_cc", seed, arm)
            curve = parse_validation_curve(run / "val_metrics.csv")
            selection = None
            try:
                recorded = json.loads((run / "best_snapshot_p1.json").read_text(encoding="utf-8"))
                reference = recorded.get("selection", {}).get("reference")
                if reference:
                    selection = recompute_selection(curve, run, reference=reference, require_checkpoints=False)
            except (OSError, ValueError, TypeError):
                pass
            spice = {str(row["iter"]): row["metrics"].get("SPICE") for row in curve.get("rows", [])}
            runs.append({"dataset": "levir_cc", "seed": seed, "arm": arm,
                         "run": str(run), "curve": curve.get("rows", []),
                         "selected_step": selection.get("selected_step") if selection else None,
                         "selected_score": selection.get("best", {}).get("score") if selection else None,
                         "selected_metrics": selection.get("best", {}).get("metrics") if selection else None,
                         "spice_by_step": spice,
                         "selection_status": selection.get("status") if selection else "unverifiable"})
            if not samples:
                try:
                    identity = json.loads((run / "validation_prediction_identity.json").read_text())
                    first = identity.get("1000", {}).get("prediction")
                    reference_path = identity.get("1000", {}).get("reference")
                    if reference_path and Path(reference_path).is_file():
                        reference_payload = json.loads(Path(reference_path).read_text())
                    if first and Path(first).is_file():
                        payload = json.loads(Path(first).read_text())
                        rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("results", []))
                        ids = sorted({str(row.get("image_id")) for row in rows if isinstance(row, dict) and row.get("image_id") is not None})
                        samples = ids[:sample_count]
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
    payload = {"schema": "semantic_controls_evidence.cc_diagnostic.v1", "dataset": "levir_cc",
               "runs": runs, "sample_ids": samples, "sample_selection": "first sorted IDs from seed1111 step1000 prediction; no semantic filtering",
               "spice_interpretation": "SPICE values are descriptive curves; no causal or human error label is inferred.",
               "review_status": "unlabelled", "stratification": {"change_status": "unavailable_without_validated_annotations", "change_area": "unavailable", "semantic_empty": "unavailable"}}
    ref_by_id = {}
    anonymous = {arm: "M%d" % (index + 1) for index, arm in enumerate(ARMS)}
    (output / "cc_anonymous_method_map.json").write_text(json.dumps(anonymous, indent=2) + "\n", encoding="utf-8")
    if isinstance(reference_payload, dict):
        for row in reference_payload.get("annotations", []):
            if isinstance(row, dict) and row.get("image_id") is not None:
                ref_by_id.setdefault(str(row["image_id"]), []).append(row.get("caption", row.get("description", "")))
    rsaca = {r["seed"]: r.get("selected_metrics", {}).get("SPICE") for r in runs if r["arm"] == "rsaca"}
    card = {r["seed"]: r.get("selected_metrics", {}).get("SPICE") for r in runs if r["arm"] == "card"}
    spice_deltas = {str(seed): (rsaca[seed] - card[seed]) for seed in SEEDS
                    if rsaca.get(seed) is not None and card.get(seed) is not None}
    payload["spice_selected_delta_rsaca_minus_card"] = spice_deltas
    payload["spice_pattern"] = ("mixed: negative for %d/%d seeds" %
                                 (sum(value < 0 for value in spice_deltas.values()), len(spice_deltas))
                                 if spice_deltas else "unavailable")
    (output / "cc_curves.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (output / "cc_manual_review.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "split", "seed", "checkpoint", "anonymous_method", "description", "reference_descriptions", "object_error", "action_error", "attribute_error", "relation_error", "annotator", "status"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for sample_id in samples:
            for seed in SEEDS:
                for arm in ARMS:
                    run = run_path(root, "levir_cc", seed, arm)
                    selected = next((r for r in runs if r["seed"] == seed and r["arm"] == arm), {})
                    identity = {}
                    try: identity = json.loads((run / "validation_prediction_identity.json").read_text())
                    except (OSError, ValueError): pass
                    prediction_path = identity.get("%s" % (selected.get("selected_step") or 1000), {}).get("prediction")
                    description = ""
                    try:
                        pred_payload = json.loads(Path(prediction_path).read_text())
                        pred_rows = pred_payload if isinstance(pred_payload, list) else pred_payload.get("predictions", pred_payload.get("results", []))
                        description = next((str(row.get("caption", row.get("prediction", ""))) for row in pred_rows
                                            if isinstance(row, dict) and str(row.get("image_id")) == sample_id), "")
                    except (OSError, ValueError, TypeError, AttributeError): pass
                    w.writerow({"sample_id": sample_id, "split": "val", "seed": seed, "checkpoint": selected.get("selected_step"),
                                "anonymous_method": anonymous[arm], "description": description,
                                "reference_descriptions": json.dumps(ref_by_id.get(sample_id, []), ensure_ascii=False), "object_error": "unlabelled",
                                "action_error": "unlabelled", "attribute_error": "unlabelled", "relation_error": "unlabelled", "annotator": "", "status": "unlabelled"})
    (output / "cc_report_zh.md").write_text("# LEVIR-CC SPICE 诊断\n\n" +
        "本报告展示四组、三个 seed 的十点 validation 曲线及历史选中 step。SPICE 下降是否集中于 seed 需查看 `cc_curves.json` 的 `spice_by_step`；不重新选择 checkpoint。\n\n" +
        "人工核查表 `cc_manual_review.csv` 使用同一固定样本 ID；对象、动作、属性、关系字段保持未标注，文本长度或重复率不能替代事实错误标签。变化分层没有经过有效标注时记为不可用。\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    for name in ("inventory", "export", "validation", "cc-diagnostic"):
        q = sub.add_parser(name); q.add_argument("--experiment-root", required=True); q.add_argument("--output-dir", required=True); q.add_argument("--dry-run", action="store_true")
        if name == "export": q.add_argument("--include-predictions", action="store_true")
        if name == "cc-diagnostic": q.add_argument("--sample-count", type=int, default=16)
    args = p.parse_args(); root = Path(args.experiment_root).resolve(); out = Path(args.output_dir).resolve()
    if args.command == "inventory":
        guard_output(root, out)
        if args.dry_run: print(json.dumps(inventory(root), indent=2, ensure_ascii=False)); return
        out.mkdir(parents=True); (out / "inventory.json").write_text(json.dumps(inventory(root), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif args.command == "export":
        print(json.dumps(export(root, out, dry_run=args.dry_run, include_predictions=args.include_predictions), indent=2, ensure_ascii=False))
    elif args.command == "cc-diagnostic":
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "dataset": "levir_cc", "sample_count": args.sample_count}, indent=2)); return
        cc_diagnostic(root, out, args.sample_count)
    else:
        payload = validation(root)
        if args.dry_run: print(json.dumps(payload, indent=2, ensure_ascii=False)); return
        write_validation(payload, out)


if __name__ == "__main__": main()
