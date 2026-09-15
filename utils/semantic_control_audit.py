"""Evidence checks and paired statistics; no training or scoring side effects."""
import copy
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

from utils.semantic_controls import ARMS, DATASETS, SEEDS, METRICS, PROTOCOL
from utils.checkpoint_integrity import sha256_file, atomic_write_text
from utils.experiment_tracking import stable_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, payload, immutable=False):
    atomic_write_text(str(path), json.dumps(payload, indent=2, sort_keys=True,
                                          ensure_ascii=False) + '\n', immutable=immutable)


def matrix(root, arms=ARMS, datasets=DATASETS, seeds=SEEDS):
    for dataset in datasets:
        for seed in seeds:
            for arm in arms:
                yield dataset, seed, arm, Path(root)/arm/('%s_%s_seed%d' % (arm, dataset, seed))


def metrics(payload):
    result = {m: float(payload[m]) for m in METRICS}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('Metrics must be finite, nonnegative, and include all five names')
    return result


def statistics_report(rows):
    """Rows: dataset, seed, arm, metrics. Missing runs never become zeroes."""
    report = {'datasets': {}, 'mean_goal': None, 'positive_seed_metrics': 0,
              'paired_seed_metrics': 0, 'statistical_significance': 'not inferred from three seeds'}
    for dataset in DATASETS:
        values = {}
        for r in rows:
            if r['dataset'] == dataset:
                key = (r['arm'], int(r['seed']))
                if key in values:
                    raise ValueError('Duplicate run identity')
                values[key] = metrics(r['metrics'])
        item = {'arms': {}, 'paired': {}}
        for arm in ARMS:
            seeds = sorted(s for a, s in values if a == arm)
            if not seeds:
                continue
            item['arms'][arm] = {'seeds': seeds, 'per_seed': {s: values[arm, s] for s in seeds},
                'mean': {m: statistics.mean(values[arm, s][m] for s in seeds) for m in METRICS},
                'sample_std': {m: statistics.stdev(values[arm, s][m] for s in seeds)
                               if len(seeds) > 1 else None for m in METRICS}}
        for candidate, baseline in (('plain_fusion', 'card'), ('rsaca', 'card'),
                                    ('rsaca', 'plain_fusion'), ('rsaca', 'fixed_gate')):
            seeds = sorted(set(s for a, s in values if a == candidate) &
                           set(s for a, s in values if a == baseline))
            if not seeds:
                continue
            diffs = {s: {m: values[candidate, s][m]-values[baseline, s][m] for m in METRICS} for s in seeds}
            paired = {'seeds': seeds, 'per_seed_difference': diffs,
                      'mean': {m: statistics.mean(diffs[s][m] for s in seeds) for m in METRICS},
                      'sample_std': {m: statistics.stdev(diffs[s][m] for s in seeds)
                                     if len(seeds) > 1 else None for m in METRICS},
                      'positive_seed_metrics': sum(v > 0 for d in diffs.values() for v in d.values())}
            item['paired'][candidate+'-'+baseline] = paired
            if (candidate, baseline) == ('rsaca', 'card'):
                report['positive_seed_metrics'] += paired['positive_seed_metrics']
                report['paired_seed_metrics'] += len(seeds)*5
        report['datasets'][dataset] = item
    comparisons = [report['datasets'][d]['paired'].get('rsaca-card') for d in DATASETS]
    if all(p and p['seeds'] == list(SEEDS) for p in comparisons):
        report['positive_mean_metrics'] = sum(v > 0 for p in comparisons for v in p['mean'].values())
        report['mean_goal'] = report['positive_mean_metrics'] == 15
    return report


def historical_report(root):
    rows, runs = [], []
    for dataset, seed, arm, run in matrix(root, arms=('card', 'rsaca')):
        score = run/'test_paired_locked_result.json'
        missing = [f for f in ('resolved_config.json', 'initial_parameter_summary.json',
                   'command.txt', 'environment.txt') if not (run/f).exists()]
        completed = False
        if score.exists():
            try:
                scored = read(score)
                value = metrics(scored['metrics'])
                completed = True
                rows.append(dict(dataset=dataset, seed=seed, arm=arm, metrics=value, source=str(score)))
                for key in ('snapshot_path', 'result_json'):
                    target = scored.get(key)
                    if not target or not Path(target).is_file():
                        missing.append('%s: %s' % (key, target or 'not recorded'))
            except (KeyError, ValueError, TypeError):
                missing.append('valid five-metric scoring')
        runs.append({'run': str(run), 'scoring_completed_from_artifact': completed,
                     'statistics_recomputable_from_metrics': completed,
                     'old_test_completed': read(run/'run_summary.json').get('test_completed')
                     if (run/'run_summary.json').exists() else None,
                     'missing': missing, 'reuse_class': 'historical_evidence' if completed else 'incomplete',
                     'new_protocol_reuse_proven': False})
    return {'statistics': statistics_report(rows), 'runs': runs,
            'note': 'Recomputed metrics are historical evidence; source/config/init/checkpoint compatibility requires separate proof. Dirty alone is not a retraining verdict.'}


def source_identity(project):
    command = ['git', '-C', str(project)]
    sha = subprocess.check_output(command+['rev-parse', 'HEAD'], text=True).strip()
    names = subprocess.check_output(command+['ls-files'], text=True, encoding='utf-8').splitlines()
    relevant = [n for n in names if n.endswith(('.py', '.yaml', '.yml', '.sh')) and
                (n.split('/')[0] in ('configs', 'models', 'datasets', 'utils', 'scripts', 'tools') or '/' not in n)]
    files = {n: sha256_file(Path(project)/n) for n in relevant}
    dirty = subprocess.check_output(command+['status', '--porcelain', '--', *relevant], text=True).strip()
    return {'commit': sha, 'execution_files': files, 'execution_sha256': stable_hash(files), 'dirty_execution': bool(dirty)}


def input_manifest(cfg, source_kind='unknown'):
    """Resolve each split/sample using the dataset's actual resolution methods.

    No dataset constructor: avoids cache generation and caption/label leakage.
    Hash references but only check IDs; captions are never used for selection.
    """
    from datasets.rcc_dataset_transformer_levir import RCCDataset
    if source_kind not in ('annotation', 'prediction', 'unknown'):
        raise ValueError('Invalid semantic source kind')
    data = cfg.data
    splits = read(data.splits_json)
    ds = RCCDataset.__new__(RCCDataset)
    ds.semantic_map_root = data.semantic_map_root or data.semantic_img_dir
    ds.s_img_dir = data.semantic_img_dir
    ds.semantic_diff_root, ds.semantic_diff_phase = data.semantic_diff_root, data.semantic_diff_phase
    ds._missing_semantic_warnings = 0
    cache, rows, missing = {}, [], []

    def fingerprint(path):
        if not path or not Path(path).is_file():
            missing.append(str(path))
            return None
        key = str(Path(path))
        if key not in cache:
            cache[key] = sha256_file(path)
        return {'path': key, 'sha256': cache[key]}

    artifacts = {key: fingerprint(data[key]) for key in
                 ('splits_json', 'vocab_json', 'h5_label_file', 'eval_anno_path', 'caption_json')}
    all_ids, all_sample_ids = set(), set()
    for split in ('train', 'val', 'test'):
        seen = set()
        for index in splits[split]:
            index = str(index)
            filename = splits['idx_to_filename'][index]
            sample_id = os.path.basename(filename)
            if index in all_ids or sample_id in all_sample_ids or sample_id in seen or splits['idx_to_split'][index] != split:
                raise ValueError('Duplicate or mismatched split/sample_id: '+index)
            all_ids.add(index)
            all_sample_ids.add(sample_id)
            seen.add(sample_id)
            row = {'split': split, 'index': index, 'sample_id': sample_id, 'visual': [], 'semantic': []}
            for root, phase in ((data.default_feature_dir, data.default_phase),
                                (data.semantic_feature_dir, data.semantic_phase)):
                try:
                    path = ds._resolve_feature_path(root, split, phase, filename)
                except FileNotFoundError:
                    path = None
                row['visual'].append(fingerprint(path))
            if data.use_semantic_maps:
                if data.semantic_diff_only:
                    paths = [ds._resolve_semantic_diff_path(split, filename)]
                else:
                    paths = [ds._resolve_semantic_path(split, phase, filename) for phase in
                             (data.semantic_before_phase, data.semantic_after_phase)]
                row['semantic'] = [fingerprint(p) for p in paths]
            rows.append(row)
    ref = read(data.eval_anno_path)
    reference_ids = {str(a['image_id']) for a in ref['annotations']}
    expected_ids = {r['sample_id'] for r in rows}
    if not expected_ids <= reference_ids:
        raise ValueError('Reference IDs do not cover split IDs: '+str(sorted(expected_ids-reference_ids)[:10]))
    return {'source_kind': source_kind, 'source_evidence': 'user declaration; verify source-generation provenance',
            'aug_grouping': 'unverified: need original-image to augmentation groups' if data.dataset == 'second_cc' else 'not applicable',
            'artifacts': artifacts, 'samples': rows, 'missing': missing}


def verify_initialization(summaries):
    """Require complete public sets and common semantic sets, not one matching tensor."""
    if set(summaries) != set(ARMS):
        raise ValueError('Initialization requires B/D/C0/G')
    baseline = summaries['card']
    for arm, summary in summaries.items():
        if summary.get('protocol_id') != PROTOCOL or summary['seed'] != baseline['seed']:
            raise ValueError('Initial identity mismatch')
        if summary['speaker'] != baseline['speaker']:
            raise ValueError('Decoder initialization mismatch: '+arm)
        for name, value in baseline['change_detector'].items():
            if summary['change_detector'].get(name) != value:
                raise ValueError('Public initialization mismatch: '+arm+'/'+name)
    semantic = {n: v for n, v in summaries['plain_fusion']['change_detector'].items()
                if n not in baseline['change_detector']}
    if not semantic or not baseline['change_detector'] or not baseline['speaker']:
        raise ValueError('Missing parameter evidence')
    for arm in ('rsaca', 'fixed_gate'):
        if any(summaries[arm]['change_detector'].get(n) != v for n, v in semantic.items()):
            raise ValueError('Shared semantic initialization mismatch: '+arm)
    if summaries['rsaca']['change_detector'] != summaries['fixed_gate']['change_detector']:
        raise ValueError('C0/G initialization mismatch')


def prediction_identity(prediction, annotation, expected_ids):
    predicted = read(prediction)
    ids = [str(row['image_id']) for row in predicted]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Prediction IDs duplicate, missing or outside selected split')
    reference_ids = {str(row['image_id']) for row in read(annotation)['annotations']}
    if not set(ids) <= reference_ids:
        raise ValueError('Prediction/reference ID mismatch')
    return {'prediction_sha256': sha256_file(prediction), 'reference_sha256': sha256_file(annotation),
            'sample_ids_sha256': stable_hash(sorted(ids)), 'count': len(ids)}


def verify_frozen_files(frozen):
    for path, digest in frozen['files'].items():
        if not Path(path).is_file() or sha256_file(path) != digest:
            raise ValueError('Frozen evidence changed or missing: '+path)


def enforce_test_admission(project, cfg, checkpoint, result_path):
    """Enforce frozen controls admission before test-side effects.

    This is intentionally a pure filesystem/config check shared by the wrapper
    and the low-level test entry point.  Legacy/P1 runs and validation
    diagnostics are left untouched.
    """
    protocol_id = str(getattr(cfg.train, 'protocol_id', 'legacy'))
    if protocol_id != PROTOCOL:
        return 'not_controls_protocol'
    if str(getattr(cfg, 'exp_name', '')) == '' or str(getattr(cfg, 'exp_dir', '')) == '':
        raise ValueError('Controls test requires an experiment directory and name')
    run = (Path(str(cfg.exp_dir)).resolve() / str(cfg.exp_name)).resolve()
    protocol_root = Path(str(cfg.exp_dir)).resolve().parent
    if protocol_root.name != PROTOCOL:
        raise ValueError('Controls run is not under the frozen protocol root')
    frozen_path = protocol_root / 'frozen.json'
    frozen_identity_path = protocol_root / 'frozen_identity.json'
    lock_path = protocol_root / 'protocol.json'
    if not frozen_path.is_file() or not frozen_identity_path.is_file() or not lock_path.is_file():
        raise ValueError('Semantic controls test requires protocol.json, frozen.json and frozen_identity.json')
    if read(frozen_identity_path).get('sha256') != sha256_file(frozen_path):
        raise ValueError('Frozen manifest changed')
    frozen = read(frozen_path)
    lock = read(lock_path)
    if frozen.get('protocol_id') != PROTOCOL or lock.get('protocol_id') != PROTOCOL:
        raise ValueError('Frozen protocol identity mismatch')
    current_source = source_identity(project)
    if current_source.get('dirty_execution') or current_source != lock.get('source'):
        raise ValueError('Execution source identity differs from the frozen protocol lock')
    if frozen.get('source') != lock.get('source'):
        raise ValueError('Frozen source identity differs from the protocol lock')
    verify_frozen_files(frozen)

    from utils.experiment_tracking import cfg_to_plain
    rel_run = str(run.relative_to(protocol_root))
    configurations = lock.get('configurations', {})
    if rel_run not in configurations or cfg_to_plain(cfg) != configurations[rel_run]:
        raise ValueError('CLI/configuration does not match frozen run identity')
    selection_path = run / 'best_snapshot_p1.json'
    if not selection_path.is_file():
        raise ValueError('Frozen run selection is missing')
    selection = read(selection_path)
    if selection.get('protocol_id') != PROTOCOL or selection.get('selection_metric_split') != 'validation':
        raise ValueError('Frozen run selection protocol mismatch')
    selected = Path(str(selection.get('best_snapshot', ''))).resolve()
    requested = Path(str(checkpoint)).resolve()
    if requested != selected:
        raise ValueError('Requested checkpoint is not the frozen selected checkpoint')
    if str(selection_path.resolve()) not in frozen.get('files', {}):
        raise ValueError('Frozen manifest does not bind the run selection')
    manifest_path = protocol_root / ('inputs_' + str(cfg.data.dataset) + '.json')
    if not manifest_path.is_file():
        raise ValueError('Frozen input manifest is missing')
    manifest = read(manifest_path)
    actual_manifest = input_manifest(cfg, manifest.get('source_kind', 'unknown'))
    if actual_manifest != manifest or actual_manifest.get('missing'):
        raise ValueError('Frozen input manifest no longer matches dataset resolution')
    from utils.checkpoint_integrity import validate_checkpoint_file
    digest = validate_checkpoint_file(str(requested), require_checksum=True,
                                      require_metadata=True, expected_step=selection['best']['iter'])
    if digest != selection.get('best_snapshot_sha256'):
        raise ValueError('Selected checkpoint hash differs from frozen selection')

    expected_output = run / 'test_output' / 'captions' / 'controls_locked' / 'sc_results.json'
    if Path(str(result_path)).resolve() != expected_output.resolve():
        raise ValueError('Controls test output must remain in the frozen run directory')
    prediction = expected_output
    identity_path = run / 'test_prediction_identity.json'
    score = run / 'test_metrics.json'
    receipt = run / 'test_score_identity.json'
    if prediction.exists():
        if not identity_path.is_file():
            raise ValueError('Existing predictions lack frozen-run identity; refusing to overwrite')
        expected_ids = [row['sample_id'] for row in manifest['samples'] if row['split'] == 'test']
        observed = prediction_identity(prediction, cfg.data.eval_anno_path, expected_ids)
        recorded = read(identity_path)
        recorded_checkpoint = recorded.get('checkpoint_sha256')
        observed['checkpoint_sha256'] = selection['best_snapshot_sha256']
        if recorded != observed or recorded_checkpoint != digest:
            raise ValueError('Existing predictions/reference/checkpoint identity changed')
        if score.exists():
            if not receipt.is_file() or read(receipt) != {'score_sha256': sha256_file(score), 'identity': observed}:
                raise ValueError('Existing score lacks a matching frozen identity receipt')
            return 'complete'
        return 'prediction_only'
    if identity_path.exists() or score.exists() or receipt.exists():
        raise ValueError('Partial test artifacts without predictions; refusing regeneration')
    return 'new'
