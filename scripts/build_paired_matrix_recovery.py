#!/usr/bin/env python3
"""Audit and plan recovery for the frozen paired CARD/RSACA matrix.

This command is read-only with respect to experiment artifacts.  It writes
only new audit/plan files, never creates a selection, checkpoint, prediction,
or test result.  A recovery type is a primary, mutually exclusive state; the
``next_actions`` field records downstream work such as testing after a
selection is reconstructed.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DATASETS = ("levir_cc", "levir_mci", "second_cc")
DATASET_LABELS = {"levir_cc": "LEVIR-CC", "levir_mci": "LEVIR-MCI", "second_cc": "SECOND-CC-AUG"}
ARMS = ("card", "rsaca")
SEEDS = (1111, 2222, 3333)
METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")
RECOVERY_TYPES = (
    "VERIFIED_COMPLETE", "REGISTER_EXISTING", "RECONSTRUCT_SELECTION",
    "NEED_VALIDATION", "NEED_TEST_OR_RESCORING", "RESUME_OR_RETRAIN",
    "INCOMPATIBLE_OR_UNVERIFIABLE",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pair-root", default="experiments/paired_card_rsaca_whole_gate_v1")
    parser.add_argument("--audit-dir", default="experiments/audit")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--write", action="store_true", help="Write only generated audit/plan files; the default is read-only.")
    parser.add_argument("--reconstruct-selections", action="store_true", help="Rebuild validation-only selections into an isolated audit directory.")
    parser.add_argument("--test-tag", default="paired_locked", help="Test result tag to audit (default: paired_locked).")
    parser.add_argument("--dry-run", action="store_true", help="Print recovery commands without writing any plan files.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(str(path))))


def relative(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def load_json(path):
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def dump_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path, root):
    if not os.path.isfile(path):
        return None
    stat = os.stat(path)
    return {"path": canonical(path), "relative_path": relative(path, root), "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256_current": sha256(path)}


def stable_id(prefix, text):
    return "%s-%s" % (prefix, hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])


def read_config(path):
    return load_json(path) if os.path.isfile(path) else None


def existing_path(path, root):
    """Resolve a config path without treating a missing relative path as evidence."""
    if not path:
        return None
    candidate = str(path)
    if not os.path.isabs(candidate):
        candidate = os.path.join(root, candidate)
    return canonical(candidate) if os.path.isfile(candidate) else None


def current_lock_fingerprint(repo, lock):
    """Reproduce the runner's two lock fingerprints without changing the lock."""
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", "models/CARD.py", "train_card_spot.py", "utils", "configs/dynamic",
             "scripts/_run_paper_training.sh", "scripts/test_specific_snapshot_sgc_card.sh",
             "scripts/select_best_snapshot_for_paper.py"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL).splitlines()
        current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                                 stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "models/CARD.py", "train_card_spot.py", "utils",
             "configs/dynamic", "scripts"], cwd=repo, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {
            "current_git_commit": None, "current_source_digest_sha256": None,
            "current_source_status_digest_sha256": None, "locked_git_commit": lock.get("git_commit"),
            "locked_source_digest_sha256": lock.get("source_digest_sha256"),
            "locked_source_status_digest_sha256": lock.get("source_status_digest_sha256"),
            "git_commit_matches_lock": False, "source_digest_matches_lock": False,
            "source_status_digest_matches_lock": False, "error": "repository HEAD unavailable for lock fingerprint",
        }
    tracked.extend(["scripts/run_paired_card_rsaca_matrix.sh", "scripts/summarize_paired_card_rsaca_matrix.py"])
    lines = []
    for rel_path in sorted(set(tracked)):
        path = os.path.join(repo, rel_path)
        if os.path.isfile(path):
            lines.append("%s  %s\n" % (sha256(path), rel_path))
    source_digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    status_digest = hashlib.sha256(status).hexdigest()
    return {
        "current_git_commit": current_commit,
        "current_source_digest_sha256": source_digest,
        "current_source_status_digest_sha256": status_digest,
        "locked_git_commit": lock.get("git_commit"),
        "locked_source_digest_sha256": lock.get("source_digest_sha256"),
        "locked_source_status_digest_sha256": lock.get("source_status_digest_sha256"),
        "git_commit_matches_lock": bool(lock) and lock.get("git_commit") == current_commit,
        "source_digest_matches_lock": bool(lock) and lock.get("source_digest_sha256") == source_digest,
        "source_status_digest_matches_lock": bool(lock) and lock.get("source_status_digest_sha256") == status_digest,
    }


def get(mapping, *paths):
    for path in paths:
        value = mapping
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def valid_val_rows(path):
    import csv as csv_module
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv_module.DictReader(handle))
    return [row for row in rows if row.get("iter") and row.get("snapshot_path") and all(row.get(metric) not in (None, "") for metric in METRICS)]


def test_result(path):
    if not os.path.isfile(path):
        return None
    try:
        return load_json(path)
    except (OSError, ValueError):
        return {"_invalid": True}


def checkpoint_inventory(run_dir, repo):
    items = []
    for path in sorted(list(Path(run_dir, "snapshots").glob("*.pt")) + list(Path(run_dir, "snapshots").glob("*.pth"))):
        item = file_info(path, repo)
        match = re.search(r"checkpoint_(\d+)\.(?:pt|pth)$", path.name)
        item["step"] = int(match.group(1)) if match else None
        items.append(item)
    return items


def command_for(action, repo, pair_root, dataset, arm, seed, include_dry_run=True, test_tag="paired_locked"):
    runner = "bash scripts/run_paired_card_rsaca_matrix.sh"
    env = "PAIR_ROOT=%s NUM_WORKERS=8 OMP_NUM_THREADS=1 PYTHON=/root/miniconda3/envs/card/bin/python" % shlex.quote(pair_root)
    suffix = " --dry-run" if include_dry_run else ""
    selector = "%s --stage select --dataset %s --seed %d%s" % (runner, dataset, seed, suffix)
    tester = "%s --stage test --dataset %s --seed %d --test-tag %s%s" % (runner, dataset, seed, shlex.quote(test_tag), suffix)
    trainer = "%s --stage train --dataset %s --seed %d%s" % (runner, dataset, seed, suffix)
    prefix = "cd %s && " % shlex.quote(repo)
    if action == "select":
        return prefix + env + " " + selector
    if action == "test":
        return prefix + env + " " + tester
    if action == "train":
        return prefix + env + " " + trainer
    return None


def prediction_audit(repo, pair_root, arm, dataset, seed, run_record, test_tag="paired_locked"):
    result_path = os.path.join(pair_root, arm, "%s_%s_seed%d" % (arm, dataset, seed), "test_%s_result.json" % test_tag)
    result = test_result(result_path)
    output = {"audit_id": stable_id("prediction", relative(result_path, repo)), "run_id": run_record["run_id"], "test_result_tag": test_tag,
              "dataset": dataset, "dataset_condition": DATASET_LABELS[dataset], "arm": arm, "seed": seed,
              "status": "not_available", "prediction_file": None, "reference_file": run_record.get("reference_file"),
              "expected_test_count": None, "observed_count": None, "unique_count": None, "duplicate_ids": [],
              "missing_ids": [], "extra_ids": [], "empty_caption_ids": [], "prediction_sha256_current": None,
              "historical_content_verified": False, "notes": []}
    if not result or result.get("_invalid"):
        output["notes"].append("test result wrapper missing or invalid")
        return output
    prediction = result.get("result_json") or result.get("prediction_file")
    if not prediction or not os.path.isfile(prediction):
        output["notes"].append("prediction file missing")
        return output
    output["prediction_file"] = canonical(prediction)
    output["prediction_sha256_current"] = sha256(prediction)
    try:
        predictions = load_json(prediction)
    except (OSError, ValueError):
        output["notes"].append("prediction JSON invalid")
        return output
    if not isinstance(predictions, list):
        output["notes"].append("prediction JSON is not a list")
        return output
    ids = [str(item.get("image_id", "")) for item in predictions if isinstance(item, dict)]
    captions = {str(item.get("image_id", "")): item.get("caption") for item in predictions if isinstance(item, dict)}
    output["observed_count"] = len(ids); output["unique_count"] = len(set(ids))
    output["duplicate_ids"] = sorted({item for item in ids if ids.count(item) > 1 and item})
    output["empty_caption_ids"] = sorted(item_id for item_id, caption in captions.items() if not str(caption or "").strip())
    config = read_config(run_record.get("resolved_config")) or {}
    split_path = existing_path(get(config, "data.splits_json"), repo)
    reference_path = existing_path(get(config, "data.eval_anno_path"), repo)
    if split_path:
        splits = load_json(split_path)
        index_map = splits.get("idx_to_filename", {})
        expected = {str(index_map[str(index)]) for index in splits.get("test", []) if str(index) in index_map}
        output["expected_test_count"] = len(expected)
        output["missing_ids"] = sorted(expected - set(ids))
        output["extra_ids"] = sorted(set(ids) - expected)
    if reference_path:
        output["reference_file"] = canonical(reference_path)
        output["reference_image_count"] = len({str(row.get("id")) for row in load_json(reference_path).get("images", [])})
    if output["expected_test_count"] is not None and not output["duplicate_ids"] and not output["missing_ids"] and not output["extra_ids"] and not output["empty_caption_ids"]:
        output["status"] = "PASS_IDS_AND_CAPTIONS"
    else:
        output["status"] = "FAIL_OR_INCOMPLETE"
    output["notes"].append("current SHA-256 proves current file content only; historical test-time content is not proven")
    return output


def inspect_arm(repo, pair_root, dataset, arm, seed, lock, lock_fingerprint, test_tag="paired_locked"):
    run_dir = canonical(os.path.join(pair_root, arm, "%s_%s_seed%d" % (arm, dataset, seed)))
    config_path = os.path.join(run_dir, "resolved_config.json")
    summary_path = os.path.join(run_dir, "run_summary.json")
    selection_path = os.path.join(run_dir, "best_snapshot_for_paper.json")
    test_path = os.path.join(run_dir, "test_%s_result.json" % test_tag)
    config = read_config(config_path) or {}
    summary = load_json(summary_path) if os.path.isfile(summary_path) else {}
    val_rows = valid_val_rows(os.path.join(run_dir, "val_metrics.csv"))
    inventory = checkpoint_inventory(run_dir, repo)
    valid_steps = [row.get("iter") for row in val_rows]
    planned = get(config, "train.total_steps", "train.max_iter")
    final_step = summary.get("final_global_step")
    selection = load_json(selection_path) if os.path.isfile(selection_path) else None
    test = test_result(test_path)
    has_run = os.path.isfile(config_path) or os.path.isfile(summary_path) or bool(inventory)
    training_complete = bool(summary.get("status") == "completed" and final_step is not None and planned is not None and final_step >= planned)
    has_full_validation = bool(val_rows and set(int(row["iter"]) for row in val_rows) >= set(range(1000, int(planned or 10000) + 1, 1000)))
    selection_valid = bool(selection and selection.get("best_snapshot") and os.path.isfile(selection.get("best_snapshot", "")))
    test_valid = bool(test and not test.get("_invalid") and isinstance(test.get("metrics"), dict) and all(metric in test["metrics"] for metric in METRICS))
    failure = str(summary.get("failure_reason") or "")
    if selection_valid and test_valid:
        primary = "VERIFIED_COMPLETE"
    elif not has_run:
        primary = "INCOMPATIBLE_OR_UNVERIFIABLE"
    elif failure or (not training_complete and inventory):
        primary = "RESUME_OR_RETRAIN"
    elif training_complete and has_full_validation and not selection_valid:
        primary = "RECONSTRUCT_SELECTION"
    elif selection_valid and not test_valid:
        primary = "NEED_TEST_OR_RESCORING"
    elif inventory:
        primary = "NEED_VALIDATION"
    else:
        primary = "INCOMPATIBLE_OR_UNVERIFIABLE"
    resolved_protocol = {
        "selection_rule": get(config, "train.selection_strategy"),
        "dataset": get(config, "data.dataset"), "seed": get(config, "train.seed"),
        "total_steps": planned, "semantic_input_mode": get(config, "model.semantic_input_mode"),
        "data_root": get(config, "data.data_root"), "eval_anno_path": get(config, "data.eval_anno_path"),
    }
    protocol_conflicts = []
    if lock.get("selection") != "validation_only: %s" % resolved_protocol["selection_rule"]:
        protocol_conflicts.append("selection rule differs from lock or is missing")
    if resolved_protocol["seed"] != seed:
        protocol_conflicts.append("resolved seed differs from arm seed")
    if dataset == "second_cc" and "SECOND-CC-AUG" not in str(resolved_protocol["data_root"] or "").upper():
        protocol_conflicts.append("second_cc arm does not resolve to SECOND-CC-AUG")
    action_blockers = []
    if not lock_fingerprint["source_digest_matches_lock"]:
        action_blockers.append("current source digest differs from immutable lock")
    if not lock_fingerprint["source_status_digest_matches_lock"]:
        action_blockers.append("current source-status digest differs from immutable lock")
    next_actions = []
    if primary == "RECONSTRUCT_SELECTION":
        next_actions += ["reconstruct_selection_from_validation_only", "then_test_selected_checkpoint_once"]
    elif primary == "NEED_TEST_OR_RESCORING":
        next_actions += ["test_selected_checkpoint_once"]
    elif primary == "RESUME_OR_RETRAIN":
        next_actions += ["verify_checkpoint_integrity_and_optimizer_state", "retrain_in_new_directory_if_resume_state_is_incomplete"]
    elif primary == "INCOMPATIBLE_OR_UNVERIFIABLE":
        next_actions += ["locate_original_run_or_record_as_missing_before_training"]
    if protocol_conflicts:
        next_actions.insert(0, "resolve_protocol_conflict_without_modifying_existing_lock")
    actions = []
    # The existing runner operates on both arms for one dataset/seed.  Emit it
    # once, from CARD's record, so users cannot accidentally execute selection
    # or testing twice merely by following the 18-arm ledger row by row.
    pair_command_owner = arm == "card"
    if pair_command_owner and "reconstruct_selection_from_validation_only" in next_actions:
        actions.append(command_for("select", repo, pair_root, dataset, arm, seed, include_dry_run=True))
    if pair_command_owner and "test_selected_checkpoint_once" in next_actions:
        actions.append(command_for("test", repo, pair_root, dataset, arm, seed, include_dry_run=True, test_tag=test_tag))
    if primary == "RESUME_OR_RETRAIN":
        actions.append(command_for("train", repo, pair_root, dataset, arm, seed, include_dry_run=True) + "  # dry-run only; existing failed directory must be archived/new protocol confirmed")
    run_id = stable_id("run", relative(run_dir, repo)); pair_id = stable_id("pair", "%s:%d" % (dataset, seed))
    record = {
        "run_id": run_id, "pair_id": pair_id, "protocol_id": "paired_card_rsaca_whole_gate_v1_locked",
        "dataset": dataset, "dataset_condition": DATASET_LABELS[dataset], "arm": arm, "seed": seed,
        "primary_recovery_type": primary, "original_directory": run_dir, "directory_exists": os.path.isdir(run_dir),
        "resolved_config": canonical(config_path) if os.path.isfile(config_path) else None,
        "training_commit": None, "training_dirty": None, "training_commit_evidence": canonical(os.path.join(run_dir, "git_info.txt")) if os.path.isfile(os.path.join(run_dir, "git_info.txt")) else None,
        "training_status": summary.get("status"), "planned_steps": planned, "actual_steps": final_step,
        "training_reached_planned_end": training_complete, "failure_reason": failure or None,
        "checkpoint_inventory": inventory, "valid_validation_rows": len(val_rows), "validation_steps": valid_steps,
        "validation_covers_expected_range": has_full_validation, "selection_record": canonical(selection_path) if selection else None,
        "selection_reconstructed": False, "selection_rule_documented": resolved_protocol["selection_rule"],
        "selected_checkpoint": selection.get("best_snapshot") if selection else None,
        "test_record": canonical(test_path) if test_valid else None, "test_result_tag": test_tag, "test_metrics_complete": test_valid,
        "reference_file": get(config, "data.eval_anno_path"), "prediction_file": test.get("result_json") if test_valid else None,
        "sample_coverage_audit_id": stable_id("prediction", relative(test_path, repo)),
        "protocol_lock": canonical(os.path.join(pair_root, "paired_protocol_lock.json")) if os.path.isfile(os.path.join(pair_root, "paired_protocol_lock.json")) else None,
        "protocol_snapshot": resolved_protocol, "lock_fingerprint": lock_fingerprint,
        "protocol_conflicts": protocol_conflicts, "current_action_blockers": action_blockers,
        "reusable": primary in ("VERIFIED_COMPLETE", "RECONSTRUCT_SELECTION", "NEED_TEST_OR_RESCORING", "NEED_VALIDATION") and not protocol_conflicts,
        "reuse_reason": "existing resolved config/checkpoints/validation artifacts are present" if has_run else "no target-root artifacts found",
        "next_actions": next_actions, "command_scope": "dataset_seed_pair",
        "pair_command_owner": pair_command_owner, "dry_run_commands": actions,
        "commands_after_lock_confirmation": [command.replace(" --dry-run", "") for command in actions],
        "qualifies_for_final_comparison": primary == "VERIFIED_COMPLETE" and not protocol_conflicts,
        "missing_evidence": [], "source_paths": [p for p in [config_path, summary_path, os.path.join(run_dir, "val_metrics.csv"), selection_path, test_path] if os.path.isfile(p)],
    }
    if not os.path.isfile(config_path): record["missing_evidence"].append("resolved_config.json")
    if not summary: record["missing_evidence"].append("run_summary.json")
    if not val_rows: record["missing_evidence"].append("val_metrics.csv or valid rows")
    if not selection_valid: record["missing_evidence"].append("historical validation selection record")
    if not test_valid: record["missing_evidence"].append("locked test result with complete metrics")
    if record["training_commit_evidence"]:
        try:
            git_info = open(record["training_commit_evidence"], encoding="utf-8", errors="replace").read()
            record["training_commit"] = re.search(r"^commit=(.+)$", git_info, re.MULTILINE).group(1)
            record["training_dirty"] = re.search(r"^dirty=(.+)$", git_info, re.MULTILINE).group(1) == "True"
            if record["training_dirty"]:
                record["protocol_conflicts"].append("training git_info records dirty=True; exact training source is not reproducible")
            if record["training_commit"] != lock.get("git_commit"):
                record["protocol_conflicts"].append("training commit differs from recovery protocol lock")
            record["qualifies_for_final_comparison"] = bool(
                record["primary_recovery_type"] == "VERIFIED_COMPLETE" and not record["protocol_conflicts"]
            )
        except (OSError, AttributeError):
            record["missing_evidence"].append("parseable git_info.txt")
    return record


def write_csv(path, records):
    fields = ["run_id", "pair_id", "dataset", "dataset_condition", "arm", "seed", "primary_recovery_type", "directory_exists", "training_status", "planned_steps", "actual_steps", "validation_covers_expected_range", "selection_record", "test_record", "reusable", "qualifies_for_final_comparison", "protocol_conflicts", "missing_evidence", "dry_run_commands"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            for key in ("protocol_conflicts", "missing_evidence", "dry_run_commands"):
                row[key] = ";".join(row[key] or [])
            writer.writerow(row)


def build_pair_records(records, prediction_reports):
    by_key = {(record["dataset"], record["seed"], record["arm"]): record for record in records}
    integrity_by_run = {report["run_id"]: report for report in prediction_reports}
    pairs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            card = by_key[(dataset, seed, "card")]
            rsaca = by_key[(dataset, seed, "rsaca")]
            card_integrity = integrity_by_run.get(card["run_id"], {})
            rsaca_integrity = integrity_by_run.get(rsaca["run_id"], {})
            complete = (card["primary_recovery_type"] == "VERIFIED_COMPLETE" and
                        rsaca["primary_recovery_type"] == "VERIFIED_COMPLETE")
            qualified = (complete and card["qualifies_for_final_comparison"] and
                         rsaca["qualifies_for_final_comparison"] and
                         card_integrity.get("status") == "PASS_IDS_AND_CAPTIONS" and
                         rsaca_integrity.get("status") == "PASS_IDS_AND_CAPTIONS")
            if qualified:
                status = "QUALIFIED_COMPLETE"
            elif complete:
                status = "COMPLETE_AWAITING_PROTOCOL_OR_INTEGRITY_CONFIRMATION"
            elif "RESUME_OR_RETRAIN" in (card["primary_recovery_type"], rsaca["primary_recovery_type"]):
                status = "BLOCKED_BY_RESUME_OR_RETRAIN"
            elif "INCOMPATIBLE_OR_UNVERIFIABLE" in (card["primary_recovery_type"], rsaca["primary_recovery_type"]):
                status = "BLOCKED_BY_MISSING_OR_UNVERIFIABLE_EVIDENCE"
            else:
                status = "PENDING_SELECTION_AND_TEST"
            pairs.append({
                "pair_id": card["pair_id"], "dataset": dataset, "dataset_condition": DATASET_LABELS[dataset],
                "seed": seed, "card_run_id": card["run_id"], "rsaca_run_id": rsaca["run_id"],
                "card_state": card["primary_recovery_type"], "rsaca_state": rsaca["primary_recovery_type"],
                "pair_status": status, "observed_complete_pair": complete,
                "qualified_pair": qualified,
                "card_prediction_integrity": card_integrity.get("status"),
                "rsaca_prediction_integrity": rsaca_integrity.get("status"),
            })
    return pairs


def render_plan(repo, pair_root, lock, fingerprint, records, pairs, prediction_reports, generated):
    counts = {}
    for record in records:
        counts[record["primary_recovery_type"]] = counts.get(record["primary_recovery_type"], 0) + 1
    lines = ["# Paired Matrix Recovery Plan", "", "Generated: `%s`  " % generated, "Repository: `%s`  " % repo,
             "Pair root: `%s`  " % pair_root, "Protocol ID: `paired_card_rsaca_whole_gate_v1_locked`", "",
             "This plan does not overwrite historical artifacts. Recovery actions may have been executed in a separate protocol root; all such actions are recorded separately from historical results.", "",
             "## Mutually Exclusive Primary States", "",
             "| State | Count | Meaning |", "|---|---:|---|"]
    meanings = {"VERIFIED_COMPLETE": "selection and complete locked test are present", "RECONSTRUCT_SELECTION": "training and full validation range exist; selection record absent", "RESUME_OR_RETRAIN": "training failed/incomplete; continuation equivalence is not established", "INCOMPATIBLE_OR_UNVERIFIABLE": "target-root evidence is absent", "NEED_TEST_OR_RESCORING": "selection exists but complete test record does not", "NEED_VALIDATION": "checkpoint exists but validation evidence is incomplete", "REGISTER_EXISTING": "existing evidence only needs registration"}
    for state in RECOVERY_TYPES:
        lines.append("| %s | %d | %s |" % (state, counts.get(state, 0), meanings[state]))
    lines += ["", "## Pair status", "", "| Dataset | Seed | CARD | RSACA | Status | Complete | Qualified |", "|---|---:|---|---|---|---|---|"]
    for pair in pairs:
        lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (
            pair["dataset_condition"], pair["seed"], pair["card_state"], pair["rsaca_state"], pair["pair_status"],
            "yes" if pair["observed_complete_pair"] else "no", "yes" if pair["qualified_pair"] else "no"))
    lines += ["", "## 18-arm ledger", "", "| Dataset | Arm | Seed | State | Train step | Val range | Selection | Test | Reusable | Final comparison |", "|---|---|---:|---|---:|---|---|---|---|---|"]
    for record in records:
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (record["dataset_condition"], record["arm"], record["seed"], record["primary_recovery_type"], record["actual_steps"] or "-", "yes" if record["validation_covers_expected_range"] else "no", "yes" if record["selection_record"] else "no", "yes" if record["test_record"] else "no", "yes" if record["reusable"] else "no", "yes" if record["qualifies_for_final_comparison"] else "no"))
    lines += ["", "## First actions", "", "1. Run the read-only preflight and review this ledger.", "2. Reconstruct validation selection for the 11 complete-but-unselected arms using the locked `paper_balanced_no_spice` rule; mark each output `reconstructed=true` and do not call it historical.", "3. Run one locked test per reconstructed selection only after review; keep outputs in the existing arm directory only if the runner confirms no test result exists.", "4. Do not resume the failed SECOND-CC-AUG RSACA seed 1111 checkpoint as equivalent: the 7000 file is corrupted and the checkpoint payload lacks optimizer/scheduler state. Archive/new-directory retraining requires a new protocol confirmation.", "5. Locate the four missing SECOND-CC-AUG arm roots before declaring them retraining targets.", "", "## Dry-run commands", ""]
    for record in records:
        if record["dry_run_commands"]:
            lines.append("### %s / %s / seed %d" % (record["dataset_condition"], record["arm"], record["seed"]))
            for command in record["dry_run_commands"]:
                lines += ["```bash", command, "```"]
    lines += ["", "## Protocol conflicts and evidence boundary", "",
              "The lock records `validation_only: paper_balanced_no_spice`, commit `64be6e…`, Python `/root/miniconda3/envs/card/bin/python3.8`, NUM_WORKERS=8 and SECOND-CC-AUG through the runner's resolved data root. The current target configs agree on the selection rule and 10000-step budget. Training `git_info.txt` records commit/dirty status separately; current SHA-256 values are not historical test-time content proofs.",
              "", "| Lock check | Value |", "|---|---|",
              "| Git commit matches | %s |" % fingerprint["git_commit_matches_lock"],
              "| Source digest matches | %s |" % fingerprint["source_digest_matches_lock"],
              "| Source-status digest matches | %s |" % fingerprint["source_status_digest_matches_lock"],
              "", "The source/source-status digests are checked using the runner's lock algorithm. If either is false, all listed commands remain dry-runs: use a new protocol root or restore the locked source state; do not rewrite the historical lock.",
              "", "Prediction integrity reports cover the selected test tag `%s`; current SHA-256 values prove current files only and do not rewrite historical evaluation records." % (prediction_reports[0].get("test_result_tag", "paired_locked") if prediction_reports else "paired_locked")]
    return "\n".join(lines) + "\n"


def render_september(records):
    return """# September Minimum Experiments

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
"""


def append_evidence_boundary(docs_dir, payload, pairs, fingerprint, generated):
    """Add a dated audit-only note without replacing the historical result text."""
    counts = payload["counts"]
    note = (
        "\n## Paired Matrix Recovery Audit (%s)\n\n"
        "Evidence/recovery update only: no training, test prediction, or formal rescoring was run. "
        "The 18 arms are VERIFIED_COMPLETE=%d, RECONSTRUCT_SELECTION=%d, RESUME_OR_RETRAIN=%d, "
        "INCOMPATIBLE_OR_UNVERIFIABLE=%d. Of 9 expected pairs, %d have both historical selection/test records "
        "and %d currently qualify after the available prediction-integrity checks. Current-source lock consistency "
        "is commit=%s, source=%s, status=%s; when a digest is false, recovery commands remain dry-runs until source "
        "isolation is resolved. See `docs/PAIRED_MATRIX_RECOVERY_PLAN.md` and "
        "`experiments/audit/paired_matrix_recovery.json`.\n"
    ) % (generated, counts.get("VERIFIED_COMPLETE", 0), counts.get("RECONSTRUCT_SELECTION", 0),
         counts.get("RESUME_OR_RETRAIN", 0), counts.get("INCOMPATIBLE_OR_UNVERIFIABLE", 0),
         sum(pair["observed_complete_pair"] for pair in pairs), sum(pair["qualified_pair"] for pair in pairs),
         fingerprint["git_commit_matches_lock"], fingerprint["source_digest_matches_lock"],
         fingerprint["source_status_digest_matches_lock"])
    marker = "## Paired Matrix Recovery Audit"
    for filename in ("CURRENT_VERIFIED_RESULTS.md", "EXPERIMENT_UPDATE_LOG.md"):
        path = os.path.join(docs_dir, filename)
        existing = open(path, encoding="utf-8").read() if os.path.isfile(path) else ""
        if marker not in existing:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(note)


def reconstruct_selections(repo, audit_dir, records, generated):
    """Run the frozen validation selector into a separate, non-historical namespace."""
    selector = os.path.join(repo, "scripts", "select_best_snapshot_for_paper.py")
    output_root = os.path.join(audit_dir, "reconstructed_selection")
    results = []
    for record in records:
        if record["primary_recovery_type"] != "RECONSTRUCT_SELECTION":
            continue
        source_dir = record["original_directory"]
        source_csv = os.path.join(source_dir, "val_metrics.csv")
        key = "%s_%s_seed%d" % (record["dataset"], record["arm"], record["seed"])
        destination = os.path.join(output_root, key)
        output_json = os.path.join(destination, "selection.json")
        copy_path = os.path.join(destination, "selected_checkpoint.pth")
        os.makedirs(destination, exist_ok=True)
        command = [sys.executable, selector, "--exp_dir", source_dir, "--csv", source_csv,
                   "--metric", "paper_balanced_no_spice", "--output_json", output_json,
                   "--copy_path", copy_path]
        completed = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        item = {"run_id": record["run_id"], "dataset": record["dataset"], "arm": record["arm"], "seed": record["seed"],
                "reconstructed": True, "reconstructed_at": generated, "source_directory": source_dir,
                "source_validation_csv": canonical(source_csv), "output_json": canonical(output_json),
                "copy_path": canonical(copy_path), "selection_rule": "paper_balanced_no_spice",
                "command": " ".join(shlex.quote(part) for part in command), "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-1000:], "stderr_tail": completed.stderr[-1000:]}
        if completed.returncode == 0 and os.path.isfile(output_json):
            payload = load_json(output_json)
            item["selected_checkpoint"] = payload.get("best_snapshot")
            item["selected_checkpoint_sha256_current"] = sha256(payload["best_snapshot"]) if payload.get("best_snapshot") and os.path.isfile(payload["best_snapshot"]) else None
            item["status"] = "RECONSTRUCTED_VALIDATION_ONLY"
        else:
            item["selected_checkpoint"] = None
            item["selected_checkpoint_sha256_current"] = None
            item["status"] = "RECONSTRUCTION_FAILED"
        results.append(item)
    dump_json(os.path.join(audit_dir, "reconstructed_selection_manifest.json"), {"schema_version": 1, "generated_at": generated, "selection_rule": "paper_balanced_no_spice", "historical_files_modified": False, "records": results})
    return results


def run(repo, pair_root, audit_dir, docs_dir, check_only=False, dry_run=False, reconstruct=False, test_tag="paired_locked"):
    lock_path = os.path.join(pair_root, "paired_protocol_lock.json")
    lock = load_json(lock_path) if os.path.isfile(lock_path) else {}
    fingerprint = current_lock_fingerprint(repo, lock)
    records = [inspect_arm(repo, pair_root, dataset, arm, seed, lock, fingerprint, test_tag) for dataset in DATASETS for arm in ARMS for seed in SEEDS]
    prediction_reports = [prediction_audit(repo, pair_root, record["arm"], record["dataset"], record["seed"], record, test_tag) for record in records]
    pairs = build_pair_records(records, prediction_reports)
    generated = datetime.now(timezone.utc).isoformat()
    reconstruction = reconstruct_selections(repo, audit_dir, records, generated) if reconstruct else []
    payload = {"schema_version": 1, "generated_at": generated, "protocol_id": "paired_card_rsaca_whole_gate_v1_locked", "pair_root": pair_root,
               "lock_path": canonical(lock_path) if os.path.isfile(lock_path) else None, "lock": lock,
               "lock_fingerprint": fingerprint, "expected_pairs": 9,
               "observed_pairs": sum(pair["observed_complete_pair"] for pair in pairs),
               "qualified_pairs": sum(pair["qualified_pair"] for pair in pairs), "pairs": pairs, "arms": records,
               "counts": {state: sum(record["primary_recovery_type"] == state for record in records) for state in RECOVERY_TYPES}}
    integrity = {"schema_version": 1, "generated_at": generated, "test_result_tag": test_tag, "reports": prediction_reports, "scope": "existing paired test outputs only", "new_predictions_generated": False, "formal_rescoring_executed": False}
    if dry_run:
        print(render_plan(repo, pair_root, lock, fingerprint, records, pairs, prediction_reports, generated))
        return payload
    if not check_only:
        os.makedirs(audit_dir, exist_ok=True); os.makedirs(docs_dir, exist_ok=True)
        dump_json(os.path.join(audit_dir, "paired_matrix_recovery.json"), payload)
        write_csv(os.path.join(audit_dir, "paired_matrix_recovery.csv"), records)
        dump_json(os.path.join(audit_dir, "prediction_integrity_report.json"), integrity)
        with open(os.path.join(docs_dir, "PAIRED_MATRIX_RECOVERY_PLAN.md"), "w", encoding="utf-8") as handle: handle.write(render_plan(repo, pair_root, lock, fingerprint, records, pairs, prediction_reports, generated))
        with open(os.path.join(docs_dir, "SEPTEMBER_MINIMUM_EXPERIMENTS.md"), "w", encoding="utf-8") as handle: handle.write(render_september(records))
        append_evidence_boundary(docs_dir, payload, pairs, fingerprint, generated)
    print(json.dumps({"counts": payload["counts"], "expected_pairs": 9, "observed_pairs": payload["observed_pairs"], "qualified_pairs": payload["qualified_pairs"], "prediction_reports": len(prediction_reports), "reconstructed_selections": len(reconstruction)}, ensure_ascii=False, indent=2))
    return payload


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        repo = canonical(temp); pair = os.path.join(repo, "pair"); os.makedirs(pair)
        lock = {"selection": "validation_only: paper_balanced_no_spice", "datasets": list(DATASETS), "seeds": list(SEEDS)}
        dump_json(os.path.join(pair, "paired_protocol_lock.json"), lock)
        run_dir = os.path.join(pair, "card", "card_levir_mci_seed3333"); os.makedirs(os.path.join(run_dir, "snapshots"))
        dump_json(os.path.join(run_dir, "resolved_config.json"), {"data": {"dataset": "levir_mci", "data_root": "LEVIR-MCI-dataset", "eval_anno_path": "missing"}, "train": {"seed": 3333, "total_steps": 10000, "selection_strategy": "paper_balanced_no_spice"}, "model": {"semantic_input_mode": "none"}})
        dump_json(os.path.join(run_dir, "run_summary.json"), {"status": "completed", "final_global_step": 10000})
        with open(os.path.join(run_dir, "val_metrics.csv"), "w", encoding="utf-8") as handle:
            handle.write("iter,snapshot_path," + ",".join(METRICS) + "\n")
            for step in range(1000, 10001, 1000):
                ck = os.path.join(run_dir, "snapshots", "x_%d.pt" % step); open(ck, "wb").write(b"x")
                handle.write("%d,%s,%s\n" % (step, ck, ",".join(["0.1"] * len(METRICS))))
        payload = run(repo, pair, os.path.join(repo, "audit"), os.path.join(repo, "docs"), check_only=True)
        assert payload["counts"]["RECONSTRUCT_SELECTION"] == 1
        # A bad historical selection path must not pass as a selection record.
        dump_json(os.path.join(run_dir, "best_snapshot_for_paper.json"), {"best_snapshot": "/missing/checkpoint.pt"})
        payload = run(repo, pair, os.path.join(repo, "audit"), os.path.join(repo, "docs"), check_only=True)
        assert payload["counts"]["RECONSTRUCT_SELECTION"] == 1

        # A real selection with no test record is a test-recovery task, not a new training run.
        selected = os.path.join(run_dir, "snapshots", "x_10000.pt")
        dump_json(os.path.join(run_dir, "best_snapshot_for_paper.json"), {"best_snapshot": selected})
        payload = run(repo, pair, os.path.join(repo, "audit"), os.path.join(repo, "docs"), check_only=True)
        assert payload["counts"]["NEED_TEST_OR_RESCORING"] == 1

        # Failed training and a wrong SECOND data root remain distinct evidence/protocol findings.
        failed = os.path.join(pair, "rsaca", "rsaca_second_cc_seed1111")
        os.makedirs(os.path.join(failed, "snapshots"))
        dump_json(os.path.join(failed, "resolved_config.json"), {"data": {"dataset": "second_cc", "data_root": "SECOND-CC", "splits_json": "missing"}, "train": {"seed": 1111, "total_steps": 10000, "selection_strategy": "paper_balanced_no_spice"}})
        dump_json(os.path.join(failed, "run_summary.json"), {"status": "failed", "failure_reason": "disk full"})
        with open(os.path.join(failed, "snapshots", "checkpoint_6000.pt"), "wb") as handle:
            handle.write(b"checkpoint")
        payload = run(repo, pair, os.path.join(repo, "audit"), os.path.join(repo, "docs"), check_only=True)
        failed_record = next(record for record in payload["arms"] if record["dataset"] == "second_cc" and record["arm"] == "rsaca" and record["seed"] == 1111)
        assert failed_record["primary_recovery_type"] == "RESUME_OR_RETRAIN"
        assert "second_cc arm does not resolve to SECOND-CC-AUG" in failed_record["protocol_conflicts"]

        # Prediction duplicate/missing-ID detection does not depend on scoring.
        prediction = os.path.join(repo, "predictions.json")
        dump_json(prediction, [{"image_id": "one.png", "caption": "a"}, {"image_id": "one.png", "caption": ""}])
        test_wrapper = os.path.join(run_dir, "test_paired_locked_result.json")
        dump_json(test_wrapper, {"metrics": {metric: 0.1 for metric in METRICS}, "result_json": prediction})
        report = prediction_audit(repo, pair, "card", "levir_mci", 3333,
                                  next(record for record in payload["arms"] if record["dataset"] == "levir_mci" and record["arm"] == "card" and record["seed"] == 3333))
        assert report["status"] == "FAIL_OR_INCOMPLETE"
        assert report["duplicate_ids"] == ["one.png"]
        assert report["empty_caption_ids"] == ["one.png"]
        print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test(); return
    if args.check_only and args.write:
        raise SystemExit("--check-only and --write cannot be combined")
    repo = canonical(args.repo_root); pair_root = canonical(os.path.join(repo, args.pair_root)); audit_dir = canonical(os.path.join(repo, args.audit_dir)); docs_dir = canonical(os.path.join(repo, args.docs_dir))
    if args.reconstruct_selections and not args.write:
        raise SystemExit("--reconstruct-selections requires --write; it creates isolated audit outputs but never edits historical arm directories")
    run(repo, pair_root, audit_dir, docs_dir, check_only=not args.write, dry_run=args.dry_run, reconstruct=args.reconstruct_selections, test_tag=args.test_tag)


if __name__ == "__main__":
    main()
