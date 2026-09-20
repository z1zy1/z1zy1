import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from utils.checkpoint_integrity import atomic_save_checkpoint
from utils.semantic_controls import METRICS
from utils.semantic_diagnostics import (
    analyze_content,
    analyze_fusion,
    compare_prediction_runs,
    compare_selection,
    curve_report,
    inventory_artifacts,
    parse_validation_curve,
    recompute_selection,
    select_fixed_samples,
    validate_prediction_ids,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_curve(run, values=None, duplicate=False, missing=None):
    run = Path(run)
    (run / "snapshots").mkdir(parents=True)
    values = values or {}
    steps = [step for step in range(1000, 10001, 1000) if step != missing]
    rows = []
    for step in steps:
        checkpoint = run / "snapshots" / ("run_checkpoint_%d.pt" % step)
        atomic_save_checkpoint({"global_step": step}, str(checkpoint), immutable=True,
                               write_checksum=True, expected_step=step)
        score = values.get(step, 1.0)
        rows.append({"iter": step, "snapshot_path": str(checkpoint), **{metric: score for metric in METRICS}})
    if duplicate and rows:
        rows.append(dict(rows[-1]))
    with (run / "val_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iter", "snapshot_path", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def test_curve_rejects_missing_duplicate_nan_and_accepts_empty_rows(tmp_path):
    _write_curve(tmp_path, duplicate=True, missing=3000)
    with (tmp_path / "val_metrics.csv").open("a") as handle:
        handle.write(",,,,,,,\n")
    parsed = parse_validation_curve(tmp_path / "val_metrics.csv")
    assert parsed["ignored_empty_rows"]
    assert 3000 in parsed["missing_steps"]
    assert parsed["duplicates"] == [10000]
    assert parsed["status"] == "invalid"
    path = tmp_path / "val_metrics.csv"
    text = path.read_text().replace(",1.0,1.0,1.0,1.0,1.0", ",nan,1.0,1.0,1.0,1.0", 1)
    path.write_text(text)
    assert any("non-finite" in error for error in parse_validation_curve(path)["errors"])


def test_selection_recomputes_reference_score_and_early_tie_break(tmp_path):
    _write_curve(tmp_path, values={1000: 1.0, 2000: 1.0})
    parsed = parse_validation_curve(tmp_path / "val_metrics.csv")
    selected = recompute_selection(parsed, tmp_path)
    assert selected["status"] == "ok"
    assert selected["selected_step"] == 1000
    selection = {
        "protocol_id": "p1_semantic_controls_20260915",
        "selection_metric_split": "validation",
        "selection": {"split": "validation", "rule": selected["rule"], "tie_break": "earlier_step", "reference": selected["reference"], "reference_sha256": __import__("utils.experiment_tracking", fromlist=["stable_hash"]).stable_hash(selected["reference"])},
        "candidates": selected["candidates"],
        "best": selected["best"],
        "best_snapshot": selected["best"]["snapshot_path"],
        "best_snapshot_sha256": selected["best"]["snapshot_sha256"],
    }
    path = tmp_path / "best_snapshot_p1.json"
    path.write_text(json.dumps(selection))
    assert compare_selection(selected, path)["status"] == "match"
    bad_selection = json.loads(json.dumps(selection))
    bad_selection["best"]["iter"] = 2000
    path.write_text(json.dumps(bad_selection))
    assert compare_selection(selected, path)["status"] == "mismatch"


def test_inventory_matrix_filters_and_does_not_write(tmp_path):
    report = inventory_artifacts(tmp_path, arms=["card", "rsaca"], datasets=["levir_cc"], seeds=[1111], steps=[1000], project=ROOT)
    assert len(report["runs"]) == 2
    assert all(item["status"] == "missing_run" for item in report["runs"])
    assert not (tmp_path / "card").exists()


def test_inventory_distinguishes_checkpoint_hash_mismatch(tmp_path):
    run = tmp_path / "card" / "card_levir_cc_seed1111"
    _write_curve(run)
    checkpoint = run / "snapshots" / "run_checkpoint_1000.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    report = inventory_artifacts(tmp_path, arms=["card"], datasets=["levir_cc"], seeds=[1111], steps=[1000], project=ROOT)
    item = report["runs"][0]
    assert item["artifacts"]["checkpoints"][0]["status"] == "damaged_or_unverifiable"
    assert item["status"] == "damaged_or_identity_mismatch"


def test_curve_analysis_can_continue_without_checkpoints(tmp_path):
    run = tmp_path / "card" / "card_levir_cc_seed1111"
    run.mkdir(parents=True)
    rows = [{"iter": step, "snapshot_path": "missing_checkpoint_%d.pt" % step,
             **{metric: 1.0 for metric in METRICS}} for step in range(1000, 10001, 1000)]
    with (run / "val_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iter", "snapshot_path", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    inventory = inventory_artifacts(tmp_path, arms=["card"], datasets=["levir_cc"], seeds=[1111], steps=[1000], project=ROOT)
    # The CSV and selector remain analyzable while checkpoint identity is explicitly absent.
    curves = curve_report(inventory, require_checkpoints=False)
    assert curves["runs"][0]["selection"]["status"] == "ok"
    assert curves["recomputed_rows"]
    assert not curves["selected_rows"]


def test_same_step_and_late_trajectory_are_separate_from_selection(tmp_path):
    for arm, offset in (("card", 0.0), ("rsaca", 0.1)):
        for seed in (1111, 2222):
            run = tmp_path / arm / ("%s_levir_cc_seed%d" % (arm, seed))
            _write_curve(run, values={step: 1.0 + offset for step in range(1000, 10001, 1000)})
    inventory = inventory_artifacts(tmp_path, arms=["card", "rsaca"], datasets=["levir_cc"],
                                    seeds=[1111, 2222], project=ROOT)
    curves = curve_report(inventory, require_checkpoints=False)
    step = curves["same_step_comparisons"]["C0-B"]
    assert step["unit"] == "paired seed at identical validation step"
    assert curves["late_trajectory"]["status"] == "exploratory"
    assert curves["late_trajectory"]["unit"].startswith("seed;")


def test_prediction_identity_is_exact_and_content_sampling_is_reproducible(tmp_path):
    reference = tmp_path / "reference.json"
    prediction = tmp_path / "prediction.json"
    reference.write_text(json.dumps({"annotations": [{"image_id": "a"}, {"image_id": "b"}]}))
    prediction.write_text(json.dumps([{"image_id": "a", "caption": "a building"}]))
    identity = validate_prediction_ids(prediction, reference)
    assert identity["status"] == "invalid"
    assert identity["missing_ids"] == ["b"]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    rows = [{"image_id": item, "caption": "a building changed"} for item in ("a", "b", "c")]
    first.write_text(json.dumps(rows))
    second.write_text(json.dumps(rows))
    analysis = analyze_content({"card": first, "rsaca": second}, sample_seed=1111, sample_count=2)
    assert analysis["status"] == "ok"
    assert analysis["samples"] == select_fixed_samples(["a", "b", "c"], seed=1111, count=2)


def test_prediction_duplicate_and_missing_ids_are_blocking(tmp_path):
    reference = tmp_path / "reference.json"
    prediction = tmp_path / "prediction.json"
    reference.write_text(json.dumps({"annotations": [{"image_id": "a"}, {"image_id": "b"}]}))
    prediction.write_text(json.dumps([{"image_id": "a", "caption": "one"}, {"image_id": "a", "caption": "two"}]))
    identity = validate_prediction_ids(prediction, reference)
    assert identity["status"] == "invalid"
    assert identity["duplicates"] == ["a"]
    assert identity["missing_ids"] == ["b"]


def test_prediction_repeat_comparison_reads_prediction_field(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"predictions": [{"image_id": "a", "prediction": "before"}]}))
    second.write_text(json.dumps({"predictions": [{"image_id": "a", "prediction": "after"}]}))
    comparison = compare_prediction_runs(first, second)
    assert comparison["ids_equal"] is True
    assert comparison["text_equal"] is False


def test_content_accepts_inference_prediction_field_and_marks_disabled_fusion(tmp_path):
    prediction = tmp_path / "prediction.json"
    prediction.write_text(json.dumps({"predictions": [
        {"image_id": "a", "prediction": "a building"},
        {"image_id": "b", "prediction": "a road"},
    ]}))
    analysis = analyze_content({"B": prediction, "C0": prediction}, sample_seed=1111, sample_count=1,
                               failure_ids=["b"])
    assert analysis["status"] == "ok"
    assert analysis["samples"]["failure_cases"] == ["b"]
    hook = tmp_path / "hook.json"
    hook.write_text(json.dumps({"records": [{"hook_status": "not_applicable"}]}))
    assert analyze_fusion({"B": hook})["arms"]["B"]["status"] == "not_applicable"


def test_cli_dry_run_does_not_create_output(tmp_path):
    experiment = tmp_path / "experiment"
    output = tmp_path / "diagnostic-output"
    command = [sys.executable, str(ROOT / "scripts/diagnose_semantic_controls.py"), "inventory",
               "--experiment-root", str(experiment), "--output-dir", str(output),
               "--arms", "card", "--datasets", "levir_cc", "--seeds", "1111", "--dry-run"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["summary"]["missing_run"] == 1
    assert not output.exists()


def test_infer_dry_run_does_not_load_or_write_checkpoint(tmp_path):
    output = tmp_path / "infer.json"
    command = [sys.executable, str(ROOT / "scripts/diagnose_semantic_controls.py"), "infer",
               "--cfg", str(tmp_path / "missing.yaml"), "--checkpoint", str(tmp_path / "missing.pt"),
               "--output", str(output), "--dry-run"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert not output.exists()

def test_multireference_zero_id_and_validation_subset(tmp_path):
    p, r = tmp_path/'p.json', tmp_path/'r.json'
    p.write_text(json.dumps([{'image_id': 0, 'caption': 'road'}]))
    refs = {'annotations': [{'image_id': 0, 'id': 1}, {'image_id': 0, 'id': 2}, {'image_id': 'test', 'id': 3}]}
    r.write_text(json.dumps(refs))
    assert validate_prediction_ids(p, r, ['0'])['status'] == 'ok'
    assert validate_prediction_ids(p, r)['status'] == 'invalid'
    refs['annotations'][1]['id'] = 1
    r.write_text(json.dumps(refs))
    assert validate_prediction_ids(p, r, ['0'])['status'] == 'invalid'


def test_hash_map_conflict_and_malformed_audit(tmp_path):
    from utils.semantic_diagnostics import _registered_hashes, _hash_registration
    from utils.checkpoint_integrity import sha256_file
    p = tmp_path/'data.csv'
    p.write_text('original')
    (tmp_path/'audit.json').write_text(json.dumps({'files': {str(p): sha256_file(p)}}))
    registered = _registered_hashes(tmp_path)
    assert _hash_registration(p, registered)['status'] == 'match'
    p.write_text('tampered')
    assert _hash_registration(p, registered)['status'] == 'mismatch'
    (tmp_path/'other_audit.json').write_text(json.dumps({'files': {str(p): '0'*64}}))
    (tmp_path/'broken_audit.json').write_text('{')
    registered = _registered_hashes(tmp_path)
    assert _hash_registration(p, registered)['status'] == 'conflict'
    assert len(registered.issues) == 2


def test_scorer_pretty_logs_and_structured_output(tmp_path):
    from utils.semantic_diagnostics import reproduce_scores, _parse_score_output
    scores = {metric: 1.0 for metric in METRICS}
    assert _parse_score_output('COCO log\n' + json.dumps({'metrics': scores}, indent=2)) == scores
    p, r = tmp_path/'p.json', tmp_path/'r.json'
    p.write_text(json.dumps([{'image_id': 'a'}]))
    r.write_text(json.dumps({'annotations': [{'image_id': 'a'}]}))
    script = tmp_path/'scorer.py'
    script.write_text('import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps('+repr({'metrics': scores})+'))\nprint("COCO logs only")\nprint("warning",file=sys.stderr)\n')
    output = tmp_path/'score'
    result = reproduce_scores(p, r, scorer_command=[sys.executable, str(script), '{output}'], output_dir=output)
    assert result['recomputed']['metrics'] == scores
    assert json.loads((output/'metrics.json').read_text())['metrics'] == scores
    assert 'warning' in (output/'stderr.log').read_text()
    with pytest.raises(ValueError, match='empty'):
        reproduce_scores(p, r, scorer_command=[sys.executable, str(script), '{output}'], output_dir=output)


def test_forward_config_changes_detected():
    import copy
    from scripts.diagnose_semantic_controls import _checkpoint_config_compatibility
    original = {'model': {'semantic_fusion_gamma_max': .5, 'semantic_fusion_global_token_mode': 'all_mean', 'semantic_fusion_warmup_steps': 0}}
    for key, value in [('semantic_fusion_gamma_max', .01), ('semantic_fusion_global_token_mode', 'changed_mean'), ('semantic_fusion_warmup_steps', 10000)]:
        changed = copy.deepcopy(original)
        changed['model'][key] = value
        assert _checkpoint_config_compatibility(original, changed)['status'] == 'mismatch'
    assert _checkpoint_config_compatibility({}, {})['status'] == 'unverifiable'


def test_nested_fusion_values_and_nonfinite_rejected(tmp_path):
    p = tmp_path/'hooks.json'
    record = {'gamma': .5, 'gate': [[.2]], 'coverage': [[.3]], 'relative_residual': .1}
    p.write_text(json.dumps({'records': [record]}))
    arm = analyze_fusion({'C0': p})['arms']['C0']
    assert arm['status'] == 'ok'
    assert arm['stats']['gate']['mean'] == .2
    record['gate'] = [[float('nan')]]
    p.write_text(json.dumps({'records': [record]}))
    with pytest.raises(ValueError, match='non-finite'):
        analyze_fusion({'C0': p})


def test_output_guard_protects_historical_and_existing(tmp_path):
    from argparse import Namespace
    from scripts.diagnose_semantic_controls import _guard_output, _write_json
    root = tmp_path/'historical'
    root.mkdir()
    with pytest.raises(ValueError, match='historical'):
        _guard_output(Namespace(output_dir=str(root/'new'), experiment_root=str(root)))
    output = tmp_path/'new.json'
    _write_json(output, {'first': True})
    with pytest.raises(FileExistsError):
        _write_json(output, {'second': True})
    assert json.loads(output.read_text()) == {'first': True}

@pytest.mark.parametrize('damage', ['metric', 'candidate', 'reference', 'reference_hash', 'tie'])
def test_selection_rejects_tampered_evidence(tmp_path, damage):
    from utils.experiment_tracking import stable_hash
    _write_curve(tmp_path)
    selected = recompute_selection(parse_validation_curve(tmp_path/'val_metrics.csv'), tmp_path)
    recorded = json.loads(json.dumps({'protocol_id': 'p1_semantic_controls_20260915',
        'selection_metric_split': 'validation',
        'selection': {'split': 'validation', 'rule': selected['rule'], 'tie_break': 'earlier_step',
                      'reference': selected['reference'], 'reference_sha256': stable_hash(selected['reference'])},
        'best': selected['best'], 'candidates': selected['candidates'],
        'best_snapshot': selected['best']['snapshot_path'], 'best_snapshot_sha256': selected['best']['snapshot_sha256']}))
    if damage == 'metric': recorded['best']['metrics']['SPICE'] = 999
    if damage == 'candidate': recorded['candidates'][2]['metrics']['SPICE'] = 999
    if damage == 'reference': recorded['selection']['reference']['SPICE'] = .1
    if damage == 'reference_hash': recorded['selection']['reference_sha256'] = '0'*64
    if damage == 'tie': recorded['selection']['tie_break'] = 'later_step'
    path = tmp_path/'best_snapshot_p1.json'
    path.write_text(json.dumps(recorded))
    assert compare_selection(selected, path)['status'] == 'mismatch'
