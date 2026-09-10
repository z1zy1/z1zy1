#!/usr/bin/env python3
"""Build evidence-first CARD/RSACA experiment registries and audit reports.

The tool reads experiment artifacts without modifying them.  It deliberately
does not infer a successful test from a directory name or from a checkpoint.
Use --check-only in automation to validate evidence without writing outputs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")
METRIC_ALIASES = {
    "bleu_1": "Bleu_1", "bleu-1": "Bleu_1", "b1": "Bleu_1",
    "bleu_2": "Bleu_2", "bleu-2": "Bleu_2", "b2": "Bleu_2",
    "bleu_3": "Bleu_3", "bleu-3": "Bleu_3", "b3": "Bleu_3",
    "bleu_4": "Bleu_4", "bleu-4": "Bleu_4", "b4": "Bleu_4",
    "meteor": "METEOR", "rouge_l": "ROUGE_L", "rouge-l": "ROUGE_L",
    "rougel": "ROUGE_L", "cider": "CIDEr", "spice": "SPICE",
}
RESULT_NAMES = ("test_paired_locked_result.json", "test_unified_locked_result.json",
                "test_card_baseline_locked_result.json", "test_7_6_locked_result.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--output-dir", default="experiments/audit")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def canonical(path):
    return os.path.realpath(os.path.abspath(os.path.normpath(str(path))))


def rel(path, root):
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def load_json(path):
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_key_value_text(path):
    """Read the simple key=value provenance files emitted by the runners."""
    values = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key, separator, value = line.rstrip("\n").partition("=")
            if separator:
                values[key.strip()] = value.strip()
    return values


def dump_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def stable_id(prefix, relative_path):
    digest = hashlib.sha256(relative_path.replace(os.sep, "/").encode("utf-8")).hexdigest()[:16]
    return "%s-%s" % (prefix, digest)


def git_value(repo, args):
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_value(mapping, *paths):
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


def existing(path, root):
    if not path:
        return None
    candidate = path if os.path.isabs(str(path)) else os.path.join(root, str(path))
    return canonical(candidate) if os.path.isfile(candidate) else None


def classify_dataset(config, run_path):
    dataset = get_value(config, "data.dataset", "dataset")
    if dataset is None:
        text = run_path.lower()
        dataset = "second_cc" if "second" in text else "levir_mci" if "mci" in text else "levir_cc" if "levir" in text else None
    root = str(get_value(config, "data.data_root") or "")
    augmentation = "SECOND-CC-AUG" in root.upper() or "SECOND-CC-AUG" in run_path.upper()
    label = "SECOND-CC-AUG" if dataset == "second_cc" and augmentation else dataset
    return dataset, label, augmentation


def model_arm(config, run_path):
    mode = get_value(config, "model.semantic_input_mode")
    if mode == "none":
        return "CARD"
    if mode == "cross_attention" or get_value(config, "train.use_semantic_cross_attention"):
        return "RSACA" if "rsaca" in run_path.lower() or "reliability" in run_path.lower() else "semantic_cross_attention"
    return "other" if mode else "unknown"


def status_for(run_dir, config, summary):
    if summary:
        status = str(summary.get("status", "")).lower()
        if status in ("completed", "complete", "done", "success"):
            return "completed"
        if status in ("failed", "failure"):
            return "failed"
        if status in ("interrupted", "cancelled"):
            return "interrupted"
    if list(Path(run_dir).glob("snapshots/*")) or list(Path(run_dir).glob("checkpoints/*")):
        return "partial"
    return "unknown"


def normalize_metrics(raw):
    normalized, unknown = {}, {}
    if not isinstance(raw, dict):
        return normalized, unknown
    for key, value in raw.items():
        canonical_name = METRIC_ALIASES.get(str(key).strip().lower())
        if canonical_name is None:
            unknown[key] = value
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            unknown[key] = value
            continue
        if not math.isfinite(value):
            unknown[key] = value
            continue
        normalized[canonical_name] = value
    return normalized, unknown


def metric_scale(metrics):
    # No automatic conversion: values above 10 are recorded as a presentation
    # scale warning so 0.59 and 59 are never silently averaged together.
    return "raw_fraction_or_cider" if all(abs(value) <= 10 for value in metrics.values()) else "raw_scale_unverified"


def resolve_artifact_path(value, run_dir, repo):
    if not value:
        return None
    candidates = [str(value), os.path.join(run_dir, str(value)), os.path.join(repo, str(value))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return canonical(candidate)
    return canonical(candidates[0])


def find_results(run_dir):
    paths = []
    for name in RESULT_NAMES:
        paths.extend(Path(run_dir).glob(name))
    paths.extend(Path(run_dir).glob("test_*_result.json"))
    return sorted({canonical(path) for path in paths})


def config_summary(config):
    return {
        "semantic_input_mode": get_value(config, "model.semantic_input_mode"),
        "semantic_map_root": get_value(config, "data.semantic_map_root"),
        "semantic_diff_root": get_value(config, "data.semantic_diff_root"),
        "semantic_source": "unknown",
        "semantic_pair": bool(get_value(config, "data.use_semantic_maps")),
        "mask_type": get_value(config, "data.mask_type"),
        "seed": get_value(config, "train.seed"),
        "num_workers": get_value(config, "data.num_workers"),
        "batch_size": get_value(config, "data.train.batch_size"),
        "total_steps": get_value(config, "train.total_steps", "train.max_iter"),
        "optimizer": get_value(config, "train.optim.type"),
        "learning_rate": get_value(config, "train.optim.lr"),
        "scheduler": get_value(config, "train.optim.scheduler"),
        "warmup_steps": get_value(config, "train.optim.warmup_steps"),
        "init_checkpoint": get_value(config, "train.init_checkpoint", "train.start_from"),
        "selection_strategy": get_value(config, "train.selection_strategy"),
        "semantic_fusion_gamma_init": get_value(config, "model.semantic_fusion_gamma_init"),
        "semantic_fusion_gate_whole_adapter": get_value(config, "model.semantic_fusion_gate_whole_adapter"),
        "semantic_fusion_reliability_gate": get_value(config, "model.semantic_fusion_reliability_gate"),
        "semantic_fusion_global_token": get_value(config, "model.semantic_fusion_global_token"),
        "semantic_fusion_norm_mode": get_value(config, "model.semantic_fusion_norm_mode"),
        "caption_json": get_value(config, "data.caption_json"),
        "eval_anno_path": get_value(config, "data.eval_anno_path"),
        "vocab_json": get_value(config, "data.vocab_json"),
        "splits_json": get_value(config, "data.splits_json"),
        "feature_source": get_value(config, "data.default_feature_dir"),
        "deterministic": get_value(config, "train.deterministic"),
        "cudnn_benchmark": get_value(config, "train.benchmark"),
        "losses": {"lambda_mask": get_value(config, "train.lambda_mask"), "lambda_semantic": get_value(config, "train.lambda_semantic")},
    }


def build_registry(repo, experiments_root):
    configs = sorted(path for path in Path(experiments_root).rglob("resolved_config.json") if "audit" not in path.parts)
    runs, evaluations, missing = [], [], []
    eval_by_run = defaultdict(list)
    for config_path in configs:
        run_dir = canonical(config_path.parent)
        relative_run = rel(run_dir, repo)
        try:
            config = load_json(config_path)
        except (OSError, ValueError) as exc:
            missing.append({"kind": "invalid_resolved_config", "path": relative_run, "detail": str(exc), "impact": "high"})
            continue
        summary_path = os.path.join(run_dir, "run_summary.json")
        try:
            summary = load_json(summary_path) if os.path.isfile(summary_path) else {}
        except ValueError:
            summary = {}
            missing.append({"kind": "invalid_run_summary", "path": rel(summary_path, repo), "detail": "JSON parse failure", "impact": "medium"})
        dataset, dataset_label, augmentation = classify_dataset(config, run_dir)
        run_id = stable_id("run", relative_run)
        git_info = os.path.join(run_dir, "git_info.txt")
        environment = os.path.join(run_dir, "environment.txt")
        git_values = load_key_value_text(git_info)
        environment_values = load_key_value_text(environment)
        record = {
            "run_id": run_id, "original_experiment_name": config.get("exp_name") or os.path.basename(run_dir),
            "absolute_path": run_dir, "relative_path": relative_run, "dataset": dataset,
            "dataset_condition": dataset_label, "is_augmented": augmentation, "model_arm": model_arm(config, run_dir),
            "run_status": status_for(run_dir, config, summary), "training_commit": git_values.get("commit"),
            "training_commit_evidence": rel(git_info, repo) if os.path.isfile(git_info) else None,
            "uncommitted_changes_evidence": git_values.get("dirty"), "evaluation_commit": git_values.get("commit"),
            "resolved_config_path": canonical(config_path), "resolved_config_summary": config_summary(config),
            "environment_path": rel(environment, repo) if os.path.isfile(environment) else None,
            "environment": {"python": environment_values.get("python_executable") or environment_values.get("python_version"),
                            "pytorch": environment_values.get("torch_version"), "cuda": environment_values.get("cuda_version"),
                            "cudnn": environment_values.get("cudnn_version"), "gpu": environment_values.get("gpus"),
                            "environment_evidence": rel(environment, repo) if os.path.isfile(environment) else None},
            "data_inputs": {"train_input": "unknown", "validation_input": "unknown", "test_input": "unknown",
                            "caption_annotation": get_value(config, "data.caption_json"), "reference_annotation": get_value(config, "data.eval_anno_path"),
                            "split_file": get_value(config, "data.splits_json"), "vocab": get_value(config, "data.vocab_json"),
                            "feature_source": get_value(config, "data.default_feature_dir"), "semantic_source": "unknown",
                            "missing_semantic_count": None, "missing_semantic_behavior": get_value(config, "data.allow_missing_pseudo_mask")},
            "training": {"actual_steps": summary.get("final_global_step"), "planned_steps": get_value(config, "train.total_steps", "train.max_iter"),
                         "reached_planned_end": None, "checkpoint_save_interval": get_value(config, "train.snapshot_interval", "train.save_interval"),
                         "validation_interval": get_value(config, "train.eval_interval")},
            "selection": {"selection_file": None, "selection_rule": get_value(config, "train.selection_strategy"),
                          "selected_checkpoint": None, "selected_checkpoint_path_match_only": None,
                          "selection_uses_test_metrics": "unknown"},
            "evaluation_ids": [], "evidence_level": "config_only", "audit_labels": [], "main_comparison_groups": [],
            "missing_artifacts": [], "source_paths": [canonical(config_path)],
        }
        if record["training"]["actual_steps"] is not None and record["training"]["planned_steps"] is not None:
            record["training"]["reached_planned_end"] = record["training"]["actual_steps"] >= record["training"]["planned_steps"]
        select_path = os.path.join(run_dir, "best_snapshot_for_paper.json")
        if os.path.isfile(select_path):
            try:
                selection = load_json(select_path)
                record["selection"].update({"selection_file": canonical(select_path), "selection_rule": selection.get("selection_metric") or record["selection"]["selection_rule"],
                                            "selected_checkpoint": resolve_artifact_path(selection.get("best_snapshot"), run_dir, repo),
                                            "selection_uses_test_metrics": False})
                record["source_paths"].append(canonical(select_path))
            except ValueError:
                record["missing_artifacts"].append("invalid_selection_json")
        else:
            record["missing_artifacts"].append("best_snapshot_for_paper.json")
            missing.append({"kind": "missing_selection_record", "path": relative_run, "detail": "best_snapshot_for_paper.json not found", "impact": "high"})
        if not os.path.isfile(git_info):
            record["missing_artifacts"].append("git_info.txt")
            missing.append({"kind": "missing_training_commit_evidence", "path": relative_run, "detail": "git_info.txt not found", "impact": "medium"})
        for result_path in find_results(run_dir):
            try:
                result = load_json(result_path)
            except ValueError:
                missing.append({"kind": "invalid_test_result", "path": rel(result_path, repo), "detail": "JSON parse failure", "impact": "high"})
                continue
            metrics, unknown_metrics = normalize_metrics(result.get("metrics", {}))
            evaluation_id = stable_id("eval", rel(result_path, repo))
            tested = resolve_artifact_path(result.get("snapshot_path"), run_dir, repo)
            prediction = resolve_artifact_path(result.get("result_json") or result.get("prediction_file"), run_dir, repo)
            selected = record["selection"]["selected_checkpoint"]
            path_match = canonical(selected) == canonical(tested) if selected and tested else None
            evaluation = {
                "evaluation_id": evaluation_id, "run_id": run_id, "absolute_path": result_path,
                "relative_path": rel(result_path, repo), "checkpoint_path": tested,
                "checkpoint_content_sha256": None, "checkpoint_identity_evidence": "path_match_only" if path_match else "unverified",
                "selection_checkpoint_path": selected, "selection_checkpoint_path_match": path_match,
                "prediction_or_scorer_result_path": prediction, "reference_file": None,
                "scorer_version": None, "evaluation_time": None, "num_images_reported": result.get("num_images"),
                "test_coverage_verified": None, "raw_metrics": result.get("metrics", {}), "normalized_metrics": metrics,
                "unknown_metric_fields": unknown_metrics, "metric_scale": metric_scale(metrics),
                "evaluation_status": "completed" if len(metrics) == len(METRICS) else "partial",
                "source_paths": [result_path] + ([prediction] if prediction and os.path.isfile(prediction) else []),
            }
            if path_match is False:
                missing.append({"kind": "checkpoint_mismatch", "path": rel(result_path, repo), "detail": "selection and tested checkpoint paths differ", "impact": "critical"})
                record["audit_labels"].append("implementation_or_execution_error")
            if len(metrics) != len(METRICS):
                missing.append({"kind": "incomplete_metrics", "path": rel(result_path, repo), "detail": "missing: " + ", ".join(sorted(set(METRICS) - set(metrics))), "impact": "high"})
            if not prediction or not os.path.isfile(prediction):
                missing.append({"kind": "missing_score_result", "path": rel(result_path, repo), "detail": "result_json/prediction artifact not accessible", "impact": "high"})
            evaluations.append(evaluation)
            eval_by_run[run_id].append(evaluation_id)
        record["evaluation_ids"] = eval_by_run[run_id]
        if record["evaluation_ids"]:
            record["evidence_level"] = "selection_and_test_record" if record["selection"]["selected_checkpoint"] else "test_record_without_selection"
        if record["run_status"] != "completed":
            record["audit_labels"].append("evidence_insufficient")
        runs.append(record)
    seen_evaluations = {}
    for evaluation in evaluations:
        evaluation["duplicate_of_evaluation_id"] = None
        key = (evaluation["checkpoint_path"], evaluation["prediction_or_scorer_result_path"])
        if not all(key):
            continue
        original = seen_evaluations.get(key)
        if original:
            evaluation["duplicate_of_evaluation_id"] = original
            missing.append({"kind": "duplicate_or_nonindependent_evaluation", "path": evaluation["relative_path"],
                            "detail": "same checkpoint and scorer-result path as %s" % original, "impact": "medium"})
        else:
            seen_evaluations[key] = evaluation["evaluation_id"]
    return runs, evaluations, missing


def metric_vector(evaluation):
    return tuple((metric, evaluation["normalized_metrics"].get(metric)) for metric in METRICS)


def comparison_groups(repo, runs, evaluations, missing):
    by_rel = {run["relative_path"]: run for run in runs}
    evals_by_run = defaultdict(list)
    for evaluation in evaluations:
        evals_by_run[evaluation["run_id"]].append(evaluation)
    groups = []
    def final_evaluations(root, filename):
        return [item for item in evaluations if item["relative_path"].startswith(root + os.sep) and os.path.basename(item["relative_path"]) == filename and len(item["normalized_metrics"]) == len(METRICS)]

    baseline = {}
    for item in final_evaluations("experiments/card_baseline_locked_tests", "test_card_baseline_locked_result.json"):
        run = next((row for row in runs if row["run_id"] == item["run_id"]), None)
        if run and run["dataset"]:
            baseline[run["dataset"]] = item["normalized_metrics"]

    def exploratory_statistics(root):
        by_dataset = defaultdict(list)
        for item in final_evaluations(root, "test_unified_locked_result.json"):
            run = next((row for row in runs if row["run_id"] == item["run_id"]), None)
            if run and run["dataset"]:
                by_dataset[run["dataset"]].append(item)
        payload = {}
        for dataset, items in sorted(by_dataset.items()):
            means = {metric: statistics.mean(item["normalized_metrics"][metric] for item in items) for metric in METRICS}
            payload[dataset] = {
                "expected_seed_n": 3, "observed_seed_n": len(items), "candidate_mean": means,
                "candidate_sample_std": {metric: statistics.stdev(item["normalized_metrics"][metric] for item in items) if len(items) > 1 else None for metric in METRICS},
                "single_audit_baseline": baseline.get(dataset),
                "delta_vs_single_audit_baseline": {metric: means[metric] - baseline[dataset][metric] for metric in METRICS} if dataset in baseline else None,
                "comparison_label": "exploratory comparison relative to a single audit baseline",
            }
        return payload
    pair_root = "experiments/paired_card_rsaca_whole_gate_v1"
    expected = []
    observed = []
    observed_arms = []
    for dataset in ("levir_cc", "levir_mci", "second_cc"):
        for seed in (1111, 2222, 3333):
            pair_key = "%s seed %d" % (dataset, seed)
            expected.append(pair_key)
            pair_complete = True
            for arm in ("card", "rsaca"):
                relative = "%s/%s/%s_%s_seed%d" % (pair_root, arm, arm, dataset, seed)
                run = by_rel.get(relative)
                complete = run and run["selection"]["selected_checkpoint"] and evals_by_run[run["run_id"]]
                if complete:
                    observed_arms.append(relative)
                else:
                    pair_complete = False
            if pair_complete:
                observed.append(pair_key)
    strict_payload = {
        "group_id": "strict_paired_whole_gate_v1", "question": "D: same core settings, CARD versus whole-adapter RSACA",
        "protocol": "strict_paired", "expected_n": len(expected), "observed_n": len(observed), "expected_arm_n": 18, "observed_arm_n": len(observed_arms),
        "missing_runs": sorted(set(expected) - set(observed)),
        "eligible_for_verified_main_table": len(observed) == len(expected), "reason": "requires all 9 seed-pairs / 18 validation-selected locked tests and verified common conditions",
        "members": observed_arms,
    }
    paired_stats = {}
    for dataset in ("levir_cc", "levir_mci", "second_cc"):
        deltas = []
        for seed in (1111, 2222, 3333):
            card_path = "%s/card/card_%s_seed%d/test_paired_locked_result.json" % (pair_root, dataset, seed)
            rsaca_path = "%s/rsaca/rsaca_%s_seed%d/test_paired_locked_result.json" % (pair_root, dataset, seed)
            card_eval = next((item for item in evaluations if item["relative_path"] == card_path and len(item["normalized_metrics"]) == len(METRICS)), None)
            rsaca_eval = next((item for item in evaluations if item["relative_path"] == rsaca_path and len(item["normalized_metrics"]) == len(METRICS)), None)
            if card_eval and rsaca_eval:
                deltas.append({metric: rsaca_eval["normalized_metrics"][metric] - card_eval["normalized_metrics"][metric] for metric in METRICS})
        paired_stats[dataset] = {"expected_seed_n": 3, "observed_seed_n": len(deltas),
                                 "paired_delta_mean": {metric: statistics.mean(row[metric] for row in deltas) if deltas else None for metric in METRICS},
                                 "paired_delta_sample_std": {metric: statistics.stdev(row[metric] for row in deltas) if len(deltas) > 1 else None for metric in METRICS},
                                 "direction_consistency": {metric: sum(row[metric] > 0 for row in deltas) if deltas else None for metric in METRICS}}
    strict_payload["statistics"] = paired_stats
    groups.append(strict_payload)
    for root, group_id, question in (
        ("experiments/unified_rsaca", "unified_rsaca_vs_single_audit_card", "A: RGB CARD versus RSACA with added semantic input"),
        ("experiments/reliability_sparse_rsaca_v1_whole_gate_retry", "whole_gate_retry_vs_single_audit_card", "A: audit baseline versus whole-gate candidate"),
    ):
        members = [run for run in runs if run["relative_path"].startswith(root + os.sep)]
        selected_tested = [run for run in members if run["selection"]["selected_checkpoint"] and evals_by_run[run["run_id"]]]
        groups.append({"group_id": group_id, "question": question, "protocol": "exploratory_vs_single_audit_baseline",
                       "expected_n": 9, "observed_n": len(selected_tested), "missing_runs": [],
                       "eligible_for_verified_main_table": False,
                       "reason": "CARD is a single audit baseline, not a seed-matched control; do not report paired gain.",
                       "members": [run["relative_path"] for run in selected_tested], "statistics": exploratory_statistics(root)})
    transfer = [run for run in runs if ("_mci_init_seed" in run["relative_path"] or "second_cc_mci_" in run["relative_path"]) and "scratch" not in run["relative_path"]]
    groups.append({"group_id": "second_cc_mci_transfer", "question": "E: initialization/transfer effect", "protocol": "transfer_not_ablation",
                   "expected_n": None, "observed_n": len(transfer), "missing_runs": [], "eligible_for_verified_main_table": False,
                   "reason": "transfer initialization changes the training protocol and cannot establish scratch structural gain.",
                   "members": [run["relative_path"] for run in transfer]})
    scratch = [run for run in runs if "second_cc_scratch_" in run["relative_path"]]
    groups.append({"group_id": "second_cc_scratch_protocol_undetermined", "question": "F: scratch runs with no registered matched control", "protocol": "protocol_undetermined",
                   "expected_n": None, "observed_n": len(scratch), "missing_runs": [], "eligible_for_verified_main_table": False,
                   "reason": "scratch naming alone does not establish a matched baseline, fixed selection rule, or comparable scorer protocol.",
                   "members": [run["relative_path"] for run in scratch]})
    return groups


def csv_rows(runs):
    rows = []
    for run in runs:
        cfg = run["resolved_config_summary"]
        rows.append({"run_id": run["run_id"], "experiment": run["original_experiment_name"], "relative_path": run["relative_path"],
                     "dataset": run["dataset"], "dataset_condition": run["dataset_condition"], "is_augmented": run["is_augmented"],
                     "model_arm": run["model_arm"], "run_status": run["run_status"], "seed": cfg["seed"], "planned_steps": run["training"]["planned_steps"],
                     "actual_steps": run["training"]["actual_steps"], "selection_file": run["selection"]["selection_file"],
                     "selected_checkpoint": run["selection"]["selected_checkpoint"], "evaluation_count": len(run["evaluation_ids"]),
                     "evidence_level": run["evidence_level"], "missing_artifacts": ";".join(run["missing_artifacts"])})
    return rows


def render_docs(repo, runs, evaluations, groups, missing):
    strict = next(group for group in groups if group["group_id"] == "strict_paired_whole_gate_v1")
    verified = [item for item in evaluations if item["evaluation_status"] == "completed" and item["selection_checkpoint_path_match"] is True]
    group_lines = "\n".join("| %s | %s | %s/%s | %s |" % (item["group_id"], item["protocol"], item["observed_n"], item["expected_n"] if item["expected_n"] is not None else "N/A", "yes" if item["eligible_for_verified_main_table"] else "no") for item in groups)
    verified_lines = "\n".join("| %s | %s | %s | %d | %s |" % (item["run_id"], item["relative_path"], item["checkpoint_path"], len(item["normalized_metrics"]), item["metric_scale"]) for item in verified[:30]) or "| none | - | - | - | - |"
    exploratory_lines = []
    for group in groups:
        if group["protocol"] != "exploratory_vs_single_audit_baseline":
            continue
        for dataset, stats in group.get("statistics", {}).items():
            mean = stats["candidate_mean"]
            delta = stats["delta_vs_single_audit_baseline"]
            if delta:
                exploratory_lines.append("| %s | %s | %d/%d | %.4f +/- %.4f | %.4f | %.4f | %.4f |" % (group["group_id"], dataset, stats["observed_seed_n"], stats["expected_seed_n"], mean["Bleu_4"], stats["candidate_sample_std"]["Bleu_4"] if stats["candidate_sample_std"]["Bleu_4"] is not None else float("nan"), delta["Bleu_4"], delta["CIDEr"], delta["SPICE"]))
    exploratory_table = "\n".join(exploratory_lines) or "| none | - | - | - | - | - | - |"
    strict_mci = strict.get("statistics", {}).get("levir_mci", {})
    strict_delta = strict_mci.get("paired_delta_mean", {})
    strict_line = "B4=%+.6f, CIDEr=%+.6f, SPICE=%+.6f; paired sample standard deviations are N/A because observed_n=1." % (strict_delta.get("Bleu_4") or 0, strict_delta.get("CIDEr") or 0, strict_delta.get("SPICE") or 0) if strict_mci.get("observed_seed_n") else "No completed strict seed-pair."
    audit = """# Experiment Audit Report

Audit time: `%s`  
Code inspected: `%s`

This is a partial audit of locally accessible artifacts. It does not treat an experiment name, summary file, checkpoint, or test-output directory as proof that a complete comparable experiment exists.

## Evidence Coverage

- resolved training configurations: %d
- test-result records: %d
- selection-and-test records with matching checkpoint paths: %d
- missing/invalid evidence findings: %d

## Comparison Groups

| Group | Protocol | observed/expected | Verified main table |
|---|---|---:|---|
%s

The strict paired matrix is incomplete: only %d of 9 required seed-pairs (2/18 arms) currently have both validation selection and locked-test records. It must not be summarized as a completed `3 x 2 x 3` main experiment.

## Verified Evaluation Records

| Run | Evaluation | Checkpoint | Metric count | Scale |
|---|---|---|---:|---|
%s

## Dataset Conditions

All locally verifiable `dataset=second_cc` runs whose resolved config names a data root point to `SECOND-CC-AUG`. The relevant inputs are `second_cc_aug_captions_reformat.json`, `splits.json`, augmented vocabulary/H5 labels, and paired semantic maps. A separate non-AUG `SECOND-CC` root was not found; claims must therefore say `SECOND-CC-AUG` or record the condition as unknown.

## Audit Boundaries

Checkpoint equality is currently path-level evidence only: historical file content at test time cannot be reconstructed without a contemporaneous checksum. Prediction coverage, duplicate IDs, empty captions, scorer version, and reference-ID alignment remain unverified where the underlying prediction/reference files are unavailable or not linked from the result wrapper.
""" % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), git_value(repo, ["rev-parse", "HEAD"]) or "unknown", len(runs), len(evaluations), len(verified), len(missing), group_lines, strict["observed_n"], verified_lines)
    current = """# Current Verified Results

Audit time: `%s`  
Evidence status: **partial audit**. Values are not promoted to a paper claim unless their comparison group is marked verified.

## Main-Table Status

No group is currently eligible for a verified paper main table. The strict paired CARD/whole-gate RSACA matrix requires 9 seed-pairs / 18 selected-and-tested arms; local evidence covers %d/9 pairs (2/18 arms).

## What Can Be Written Now

- The project contains test records and validation-selection records for several CARD/RSACA runs.
- `SECOND-CC` local runs use the `SECOND-CC-AUG` data condition when a resolved configuration is available.
- Existing single-CARD versus three-seed RSACA summaries are exploratory comparisons against a single audit baseline, not three-seed paired gains.

## Audited Descriptive Summaries

These values are recomputed from the locally accessible selected-test result wrappers. They use the raw scorer scale and must not be treated as a paired main-table result.

| Group | Dataset | observed/expected seeds | B4 mean +/- sample sd | B4 delta | CIDEr delta | SPICE delta |
|---|---|---:|---:|---:|---:|---:|
%s

The only currently complete strict seed-pair is LEVIR-MCI seed 3333: %s

## What Cannot Be Written Now

- That CARD/RSACA completed a strict `3 x 2 x 3` paired matrix.
- A paired mean delta, confidence interval, significance claim, or zero standard deviation for any single-seed comparison.
- That a historical tested checkpoint has identical content to the validation-selected file based solely on matching paths.
- That SECOND-CC non-AUG conditions were evaluated.

## Minimal Next Experiments

1. Complete the remaining 16 validation-selected, locked-test records in the existing paired matrix without changing its lock.
2. Persist a SHA-256 checksum at selection and test time, plus scorer commit/version.
3. Retain/link prediction and reference IDs, then audit test coverage and duplicate/empty predictions.
4. Run a separately registered non-AUG SECOND-CC condition only if it is a paper target; never pool it with SECOND-CC-AUG.
""" % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), strict["observed_n"], exploratory_table, strict_line)
    update = """# Experiment Update Log

## %s audit correction

- Old behavior: `summarize_unified_rsaca.py` and `summarize_paired_card_rsaca_matrix.py` emitted `0.0` sample standard deviation when only one seed was present.
- New behavior: one-seed sample standard deviation is `null` / `N/A`; it is not numerical evidence of zero training variation.
- Old behavior: checkpoint agreement was inferred from path equality only and not surfaced as an evidence limitation.
- New behavior: the registry records `path_match_only` separately from a content checksum and reports missing historical checksum evidence.
- Result provenance: original result wrappers and raw scorer JSON are retained unchanged. This update is a registry/summary correction, not a re-score.
""" % datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return audit, current, update


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["run_id"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(repo, experiments_root, output_dir, docs_dir, check_only=False):
    runs, evaluations, missing = build_registry(repo, experiments_root)
    groups = comparison_groups(repo, runs, evaluations, missing)
    payload = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "repo_root": repo, "runs": runs}
    evaluation_payload = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "evaluations": evaluations}
    if not check_only:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(docs_dir, exist_ok=True)
        dump_json(os.path.join(output_dir, "experiment_registry.json"), payload)
        write_csv(os.path.join(output_dir, "experiment_registry.csv"), csv_rows(runs))
        dump_json(os.path.join(output_dir, "evaluation_registry.json"), evaluation_payload)
        dump_json(os.path.join(output_dir, "comparison_groups.json"), {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "groups": groups})
        write_csv(os.path.join(output_dir, "missing_evidence.csv"), missing or [{"kind": "none", "path": "", "detail": "", "impact": ""}])
        audit, current, update = render_docs(repo, runs, evaluations, groups, missing)
        Path(os.path.join(docs_dir, "EXPERIMENT_AUDIT_REPORT.md")).write_text(audit, encoding="utf-8")
        Path(os.path.join(docs_dir, "CURRENT_VERIFIED_RESULTS.md")).write_text(current, encoding="utf-8")
        Path(os.path.join(docs_dir, "EXPERIMENT_UPDATE_LOG.md")).write_text(update, encoding="utf-8")
    return {"runs": len(runs), "evaluations": len(evaluations), "missing": len(missing), "groups": groups}


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        run_dir = root / "experiments" / "paired_card_rsaca_whole_gate_v1" / "card" / "card_levir_cc_seed1111"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "snapshots" / "x.pt"
        checkpoint.parent.mkdir(); checkpoint.write_bytes(b"checkpoint")
        dump_json(run_dir / "resolved_config.json", {"exp_name": "card_levir_cc_seed1111", "model": {"semantic_input_mode": "none"}, "data": {"dataset": "levir_cc", "num_workers": 0, "train": {"batch_size": 1}}, "train": {"seed": 1111, "total_steps": 1}})
        dump_json(run_dir / "best_snapshot_for_paper.json", {"best_snapshot": str(checkpoint), "selection_metric": "validation_only"})
        dump_json(run_dir / "test_paired_locked_result.json", {"snapshot_path": str(checkpoint), "result_json": str(run_dir / "sc_results.json"), "metrics": {"B4": 0.5, "CIDEr": 1.0, "SPICE": 0.2}})
        dump_json(run_dir / "sc_results.json", {})
        dump_json(run_dir / "test_copy_result.json", {"snapshot_path": str(checkpoint), "result_json": str(run_dir / "sc_results.json"), "metrics": {"B4": 0.5, "CIDEr": 1.0, "SPICE": 0.2}})
        result = run(str(root), str(root / "experiments"), str(root / "audit"), str(root / "docs"))
        assert result["runs"] == 1 and result["evaluations"] == 2
        assert any(item["kind"] == "incomplete_metrics" for item in csv.DictReader(open(root / "audit" / "missing_evidence.csv", encoding="utf-8")))
        registry = load_json(root / "audit" / "evaluation_registry.json")
        assert registry["evaluations"][0]["metric_scale"] == "raw_fraction_or_cider"
        assert any(item["duplicate_of_evaluation_id"] for item in registry["evaluations"])

        def make_pair(pair_root, seed, arm, b4=0.5, checkpoint_mismatch=False, lr=0.1):
            run = pair_root / arm / ("%s_levir_cc_seed%d" % (arm, seed))
            (run / "snapshots").mkdir(parents=True, exist_ok=True)
            checkpoint = run / "snapshots" / ("%s.pt" % arm)
            checkpoint.write_bytes((arm + str(seed)).encode("ascii"))
            other = run / "snapshots" / "other.pt"
            other.write_bytes(b"other")
            config = {"exp_name": run.name, "model": {"semantic_input_mode": "none" if arm == "card" else "cross_attention"},
                      "data": {"dataset": "levir_cc", "num_workers": 0, "train": {"batch_size": 1}},
                      "train": {"seed": seed, "total_steps": 1, "optim": {"lr": lr}, "use_semantic_cross_attention": arm == "rsaca"}}
            dump_json(run / "resolved_config.json", config)
            dump_json(run / "best_snapshot_for_paper.json", {"best_snapshot": str(checkpoint), "selection_metric": "validation_only"})
            dump_json(run / "test_paired_locked_result.json", {"snapshot_path": str(other if checkpoint_mismatch else checkpoint), "metrics": {metric: b4 if metric == "Bleu_4" else 1.0 for metric in METRICS}})

        paired_script = Path(__file__).with_name("summarize_paired_card_rsaca_matrix.py")
        pair = root / "pair"
        pair.mkdir()
        dump_json(pair / "paired_protocol_lock.json", {"datasets": ["levir_cc"], "seeds": [1, 2]})
        make_pair(pair, 1, "card", 0.4); make_pair(pair, 1, "rsaca", 0.5)
        make_pair(pair, 2, "card", 0.4); make_pair(pair, 2, "rsaca", 0.6)
        normal = subprocess.run([sys.executable, str(paired_script), "--pair-root", str(pair), "--datasets", "levir_cc", "--seeds", "1,2", "--output", str(root / "normal.json")], text=True, capture_output=True)
        assert normal.returncode == 0, normal.stderr
        assert load_json(root / "normal.json")["datasets"]["levir_cc"]["paired_delta_sample_std"]["Bleu_4"] > 0

        one_pair = root / "one-pair"
        one_pair.mkdir()
        dump_json(one_pair / "paired_protocol_lock.json", {"datasets": ["levir_cc"], "seeds": [1]})
        make_pair(one_pair, 1, "card", 0.4); make_pair(one_pair, 1, "rsaca", 0.5)
        one_seed = subprocess.run([sys.executable, str(paired_script), "--pair-root", str(one_pair), "--datasets", "levir_cc", "--seeds", "1", "--output", str(root / "one.json")], text=True, capture_output=True)
        assert one_seed.returncode == 0, one_seed.stderr
        assert load_json(root / "one.json")["datasets"]["levir_cc"]["paired_delta_sample_std"]["Bleu_4"] is None

        missing_pair = root / "missing-pair"
        missing_pair.mkdir()
        dump_json(missing_pair / "paired_protocol_lock.json", {"datasets": ["levir_cc"], "seeds": [1, 2]})
        make_pair(missing_pair, 1, "card"); make_pair(missing_pair, 1, "rsaca")
        missing_seed = subprocess.run([sys.executable, str(paired_script), "--pair-root", str(missing_pair), "--datasets", "levir_cc", "--seeds", "1,2", "--output", str(root / "missing.json")], text=True, capture_output=True)
        assert missing_seed.returncode != 0 and "Missing paired artifact" in missing_seed.stderr

        mismatch_pair = root / "mismatch-pair"
        mismatch_pair.mkdir()
        dump_json(mismatch_pair / "paired_protocol_lock.json", {"datasets": ["levir_cc"], "seeds": [1]})
        make_pair(mismatch_pair, 1, "card", checkpoint_mismatch=True); make_pair(mismatch_pair, 1, "rsaca")
        mismatch = subprocess.run([sys.executable, str(paired_script), "--pair-root", str(mismatch_pair), "--datasets", "levir_cc", "--seeds", "1", "--output", str(root / "mismatch.json")], text=True, capture_output=True)
        assert mismatch.returncode != 0 and "checkpoint mismatch" in mismatch.stderr.lower()

        confounded_pair = root / "confounded-pair"
        confounded_pair.mkdir()
        dump_json(confounded_pair / "paired_protocol_lock.json", {"datasets": ["levir_cc"], "seeds": [1]})
        make_pair(confounded_pair, 1, "card", lr=0.1); make_pair(confounded_pair, 1, "rsaca", lr=0.2)
        confounded = subprocess.run([sys.executable, str(paired_script), "--pair-root", str(confounded_pair), "--datasets", "levir_cc", "--seeds", "1", "--output", str(root / "confounded.json")], text=True, capture_output=True)
        assert confounded.returncode != 0 and "allowlist" in confounded.stderr

        scale_run = root / "experiments" / "scale"
        scale_run.mkdir(); dump_json(scale_run / "resolved_config.json", {"exp_name": "scale", "data": {"dataset": "levir_cc"}, "model": {}, "train": {}})
        dump_json(scale_run / "test_unified_locked_result.json", {"metrics": {metric: 50.0 for metric in METRICS}})
        scale_result = run(str(root), str(root / "experiments"), str(root / "audit2"), str(root / "docs2"))
        assert scale_result["runs"] == 2
        assert any(item["metric_scale"] == "raw_scale_unverified" for item in load_json(root / "audit2" / "evaluation_registry.json")["evaluations"])
    print("self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    repo = canonical(args.repo_root)
    result = run(repo, canonical(os.path.join(repo, args.experiments_root)), canonical(os.path.join(repo, args.output_dir)), canonical(os.path.join(repo, args.docs_dir)), args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
