#!/usr/bin/env python
"""Sequential AutoDL B/D/C0/G protocol. No implicit test or training stage."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if '--dry-run' in sys.argv:
    sys.dont_write_bytecode = True

PROJECT = Path(os.environ.get('PROJECT_DIR') or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(PROJECT))
from utils.semantic_controls import ARMS, DATASETS, SEEDS, PROTOCOL, configuration
from utils.semantic_control_audit import (read, write, matrix, metrics, statistics_report,
    historical_report, source_identity, input_manifest, verify_initialization,
    prediction_identity, verify_frozen_files, enforce_test_admission)
from utils.checkpoint_integrity import sha256_file, validate_checkpoint_file
from utils.experiment_tracking import cfg_to_plain, stable_hash


def execute(command, run):
    """Record actual command, environment and elapsed subprocess time, even on failure."""
    started = time.time()
    code = -1
    try:
        print('EXEC '+json.dumps(command), flush=True)
        code = subprocess.call(command, cwd=str(PROJECT))
        if code:
            raise RuntimeError('Command failed with status %d' % code)
    finally:
        log = run/'execution_ledger.json'
        entries = read(log) if log.exists() else []
        entries.append({'command': command, 'started_unix': started, 'elapsed_seconds': time.time()-started,
                        'exit_code': code, 'python': sys.executable, 'platform': platform.platform()})
        write(log, entries)


def assert_unfrozen(root):
    if (root/'frozen.json').exists():
        raise ValueError('Frozen protocol: training, selection and preflight are closed')
    if any(root.glob('*/*/test_metrics.json')) or any(root.glob('*/*/test_output/**/*.json')):
        raise ValueError('Test artifacts already exist; do not reselect or call this an unseen test')


def expected(root, dataset, seed, arm):
    return configuration(PROJECT, root, dataset, seed, arm, os.environ)


def ensure_lock(root):
    lock = read(root/'protocol.json')
    identity = source_identity(PROJECT)
    if lock['protocol_id'] != PROTOCOL or identity != lock['source'] or identity['dirty_execution']:
        raise ValueError('Protocol source identity changed: use a separate protocol directory and audit compatibility')
    if lock['runtime'] != runtime_identity():
        raise ValueError('Python/PyTorch/CUDA/NumPy runtime changed since preflight')
    # A changed path, worker count, architecture or seed cannot silently reuse a run.
    for dataset, seed, arm, run in matrix(root):
        key = str(run.relative_to(root))
        if cfg_to_plain(expected(root, dataset, seed, arm)) != lock['configurations'][key]:
            raise ValueError('Configuration/environment changed: '+key)
    return lock


def runtime_identity():
    import torch
    import numpy
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': numpy.__version__, 'cuda': torch.version.cuda,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')}


def verify_inputs(root, lock):
    for dataset in DATASETS:
        item = read(root/('inputs_'+dataset+'.json'))
        if stable_hash(item) != lock['input_hashes'][dataset]:
            raise ValueError('Input manifest changed: '+dataset)
        # Re-resolve to catch a newly added higher-priority filename as well as changes.
        actual = input_manifest(expected(root, dataset, 1111, 'rsaca'), item['source_kind'])
        if actual != item or actual['missing']:
            raise ValueError('Dataset files or resolution changed: '+dataset)


def preflight(root, sources):
    assert_unfrozen(root)
    if root.exists() and any(root.iterdir()) and not (root/'protocol.json').exists():
        raise ValueError('Nonempty directory without this protocol lock; preserve it and choose a new experiments subdirectory')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('AutoDL training preflight requires CUDA; use --dry-run for CPU configuration inspection')
    if not shutil.which('java'):
        raise RuntimeError('Java is required for METEOR/SPICE; verify the existing server environment')
    identity = source_identity(PROJECT)
    if identity['dirty_execution']:
        raise ValueError('Commit execution source before locking a new experiment')
    lock = {'protocol_id': PROTOCOL, 'source': identity, 'runtime': runtime_identity(), 'configurations': {}, 'input_hashes': {},
            'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')}}
    manifests = {}
    for dataset in DATASETS:
        cfg = expected(root, dataset, 1111, 'rsaca')
        manifest = input_manifest(cfg, sources.get(dataset, 'unknown'))
        if manifest['missing']:
            raise ValueError('Missing inputs for %s: %s' % (dataset, manifest['missing'][:20]))
        manifests[dataset] = manifest
        lock['input_hashes'][dataset] = stable_hash(manifest)
    for dataset, seed, arm, run in matrix(root):
        lock['configurations'][str(run.relative_to(root))] = cfg_to_plain(expected(root, dataset, seed, arm))
    if (root/'protocol.json').exists():
        old = read(root/'protocol.json')
        if old != lock:
            raise ValueError('Existing protocol lock differs; refusing overwrite')
        verify_inputs(root, old)
        return
    for dataset, manifest in manifests.items():
        write(root/('inputs_'+dataset+'.json'), manifest, immutable=True)
    write(root/'protocol.json', lock, immutable=True)


def actual_config_check(request, actual):
    """Only runtime vocabulary/length and semantic class adaptation are allowed."""
    import h5py
    expected_cfg = copy.deepcopy(request)
    expected_cfg['model']['transformer_decoder']['vocab_size'] = len(read(request['data']['vocab_json']))
    with h5py.File(request['data']['h5_label_file'], 'r') as handle:
        expected_cfg['model']['transformer_decoder']['seq_length'] = handle['labels'].shape[1]
    if request['data']['use_semantic_maps']:
        expected_cfg['model']['num_semantic_classes'] = request['data']['num_semantic_classes']+1
    expected_cfg['train']['content_word_token_ids'] = []
    for section in ('model', 'train', 'data'):
        for key, value in expected_cfg[section].items():
            if actual[section].get(key) != value:
                raise ValueError('Unexpected actual config difference: '+section+'.'+key)


def run_audit(root, lock, require_selection=True):
    errors, files, rows = [], {}, []
    initial = {}
    for dataset, seed, arm, run in matrix(root):
        key = str(run.relative_to(root))
        try:
            request = read(run/'request_config.json')
            if request != lock['configurations'][key]:
                raise ValueError('Request config mismatch')
            if read(run/'resolved_config.json') != request:
                raise ValueError('Resolved training config differs from requested protocol')
            actual_config_check(request, read(run/'actual_model_config.json'))
            provenance = read(run/'controls_provenance.json')
            if provenance['source'] != lock['source'] or provenance['input_hash'] != lock['input_hashes'][dataset]:
                raise ValueError('Run provenance mismatch')
            summary_init = read(run/'initial_parameter_summary.json')
            if summary_init.get('seed') != seed:
                raise ValueError('Initial seed does not match run identity')
            initial.setdefault((dataset, seed), {})[arm] = summary_init
            summary = read(run/'run_summary.json')
            if summary.get('status') != 'completed' or summary.get('final_global_step') != 10000:
                raise ValueError('Training incomplete')
            evidence = ['request_config.json', 'actual_model_config.json', 'controls_provenance.json',
                        'initial_parameter_summary.json', 'resolved_config.json', 'command.txt',
                        'environment.txt', 'val_metrics.csv', 'validation_prediction_identity.json']
            validation_ids = read(run/'validation_prediction_identity.json')
            if sorted(map(int, validation_ids)) != list(range(1000, 10001, 1000)):
                raise ValueError('Incomplete validation prediction identity grid')
            for item in validation_ids.values():
                manifest_ids = [r['sample_id'] for r in read(root/('inputs_'+dataset+'.json'))['samples'] if r['split'] == 'val']
                observed = prediction_identity(item['prediction'], item['reference'], manifest_ids)
                if any(item.get(k) != v for k, v in observed.items()):
                    raise ValueError('Validation sample identity differs from protocol')
                if sha256_file(item['prediction']) != item['prediction_sha256'] or sha256_file(item['reference']) != item['reference_sha256']:
                    raise ValueError('Validation prediction/reference changed')
                files[item['prediction']] = item['prediction_sha256']
                files[item['reference']] = item['reference_sha256']
            if require_selection:
                selected = read(run/'best_snapshot_p1.json')
                if selected['protocol_id'] != PROTOCOL or selected['selection_metric_split'] != 'validation':
                    raise ValueError('Selection protocol mismatch')
                # Recalculate CSV/grid/checksums independently and compare selected record.
                import tempfile
                with tempfile.TemporaryDirectory(dir=run) as scratch:
                    check = Path(scratch)/'selection.json'
                    subprocess.run([sys.executable, str(PROJECT/'scripts/select_best_snapshot_p1.py'),
                        '--exp-dir', str(run), '--protocol-id', PROTOCOL, '--output-json', str(check)],
                        check=True, stdout=subprocess.DEVNULL)
                    recalculated = read(check)
                if selected != recalculated:
                    raise ValueError('Selection differs from complete validation grid')
                checkpoint = selected['best_snapshot']
                digest = validate_checkpoint_file(checkpoint, require_checksum=True, require_metadata=True,
                                                   expected_step=selected['best']['iter'])
                if digest != selected['best_snapshot_sha256']:
                    raise ValueError('Selected checkpoint mismatch')
                from utils.checkpointing import load_checkpoint_file
                checkpoint_data = load_checkpoint_file(checkpoint)
                if checkpoint_data['global_step'] != selected['best']['iter']:
                    raise ValueError('Checkpoint payload step mismatch')
                actual_config_check(request, cfg_to_plain(checkpoint_data['config']))
                for model_key, init_key in (('model_state_dict', 'change_detector'), ('speaker_state_dict', 'speaker')):
                    state = checkpoint_data[model_key]
                    if any(name not in state or list(state[name].shape) != value['shape']
                           for name, value in summary_init[init_key].items()):
                        raise ValueError('Checkpoint parameter set/shape differs from initial model')
                del checkpoint_data
                files[checkpoint] = digest
                files[checkpoint+'.sha256'] = sha256_file(checkpoint+'.sha256')
                files[checkpoint+'.metadata.json'] = sha256_file(checkpoint+'.metadata.json')
                evidence.append('best_snapshot_p1.json')
                rows.append(dict(dataset=dataset, seed=seed, arm=arm, metrics=selected['best']['metrics']))
            for name in evidence:
                files[str(run/name)] = sha256_file(run/name)
        except (OSError, KeyError, ValueError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(key+': '+str(exc))
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                verify_initialization(initial.get((dataset, seed), {}))
            except ValueError as exc:
                errors.append('%s/%d: %s' % (dataset, seed, exc))
    return {'protocol_audit_passed': not errors, 'errors': errors, 'files': files,
            'new_protocol_compatible_runs': [str(run) for _, _, _, run in matrix(root)] if not errors else [],
            'validation': statistics_report(rows)}


def freeze(root, lock):
    assert_unfrozen(root)
    verify_inputs(root, lock)
    audit = run_audit(root, lock)
    write(root/'audit.json', audit)
    if not audit['protocol_audit_passed'] or audit['validation']['mean_goal'] is not True:
        raise ValueError('Test blocked: full audit and C0 validation 15/15 mean improvement are required')
    files = audit['files']
    for path in [root/'protocol.json'] + [root/('inputs_'+d+'.json') for d in DATASETS]:
        files[str(path)] = sha256_file(path)
    write(root/'frozen.json', {'protocol_id': PROTOCOL, 'files': files,
          'validation_admitted': True, 'source': lock['source'],
          'validation_statistics': audit['validation']}, immutable=True)
    write(root/'frozen_identity.json', {'sha256': sha256_file(root/'frozen.json')}, immutable=True)


def load_frozen(root, lock):
    if read(root/'frozen_identity.json')['sha256'] != sha256_file(root/'frozen.json'):
        raise ValueError('Frozen manifest changed')
    frozen = read(root/'frozen.json')
    if (frozen['protocol_id'] != PROTOCOL or frozen['validation_admitted'] is not True
            or frozen['source'] != lock['source']):
        raise ValueError('Test not admitted by this protocol')
    verify_frozen_files(frozen)
    return frozen


def train_one(root, lock, dataset, seed, arm, run):
    assert_unfrozen(root)
    if (run/'run_summary.json').exists() and read(run/'run_summary.json').get('status') == 'completed':
        if read(run/'run_summary.json').get('final_global_step') != 10000:
            raise ValueError('Completed flag with incomplete training steps')
        print('Existing training; audit before reuse: '+str(run))
        return
    if run.exists() and any(run.iterdir()):
        raise ValueError('Incomplete/nonempty run; preserve evidence and investigate: '+str(run))
    run.mkdir(parents=True, exist_ok=True)
    cfg = lock['configurations'][str(run.relative_to(root))]
    write(run/'request_config.json', cfg, immutable=True)
    write(run/'controls_provenance.json', {'source': lock['source'],
        'input_hash': lock['input_hashes'][dataset], 'environment': lock['environment']}, immutable=True)
    execute([sys.executable, 'train_card_spot.py', '--cfg', str(run/'request_config.json'),
             '--model', 'sgc_card', '--output_dir', str(run)], run)


def test_one(root, frozen, dataset, seed, arm, run):
    verify_frozen_files(frozen)
    cfg = read(run/'request_config.json')
    # Keep the wrapper and low-level test entry point on the same admission path.
    class _Node(dict):
        __getattr__ = dict.__getitem__
    def node(value):
        return _Node({k: node(v) if isinstance(v, dict) else v for k, v in value.items()}) if isinstance(value, dict) else value
    enforce_test_admission(PROJECT, node(cfg),
                           read(run/'best_snapshot_p1.json')['best_snapshot'],
                           run/'test_output/captions/controls_locked/sc_results.json')
    selected = read(run/'best_snapshot_p1.json')
    checkpoint = selected['best_snapshot']
    prediction = run/'test_output/captions/controls_locked/sc_results.json'
    annotation = cfg['data']['eval_anno_path']
    expected_ids = [r['sample_id'] for r in read(root/('inputs_'+dataset+'.json'))['samples'] if r['split'] == 'test']
    identity_path = run/'test_prediction_identity.json'
    if not prediction.exists():
        if identity_path.exists() or (run/'test_metrics.json').exists():
            raise ValueError('Completed test evidence missing; refusing regeneration')
        execute([sys.executable, 'test_card_spot.py', '--cfg', str(run/'request_config.json'),
            '--model', 'sgc_card', '--snapshot_path', checkpoint, '--split', 'test',
            '--result_json', str(prediction)], run)
        identity = prediction_identity(prediction, annotation, expected_ids)
        identity['checkpoint_sha256'] = selected['best_snapshot_sha256']
        write(identity_path, identity, immutable=True)
    if not identity_path.exists():
        raise ValueError('Predictions without frozen-run provenance; audit interrupted inference before reuse')
    identity = prediction_identity(prediction, annotation, expected_ids)
    identity['checkpoint_sha256'] = selected['best_snapshot_sha256']
    if identity != read(identity_path):
        raise ValueError('Test prediction/reference/checkpoint identity changed')
    score = run/'test_metrics.json'
    receipt = run/'test_score_identity.json'
    if score.exists():
        metrics(read(score)['metrics'])
        if not receipt.exists() or read(receipt) != {'score_sha256': sha256_file(score), 'identity': identity}:
            raise ValueError('Existing scoring lacks compatible receipt; do not silently regenerate')
        print('Already scored: '+str(run))
        return
    execute([sys.executable, 'tools/score_predictions.py', '--annotation', annotation,
             '--predictions', str(prediction), '--stage', 'test', '--run-dir', str(run),
             '--checkpoint', checkpoint, '--output-json', str(score)], run)
    metrics(read(score)['metrics'])
    write(receipt, {'score_sha256': sha256_file(score), 'identity': identity}, immutable=True)


def planned_command(stage, run):
    if stage == 'train':
        return [sys.executable, 'train_card_spot.py', '--cfg', str(run/'request_config.json'),
                '--model', 'sgc_card', '--output_dir', str(run)]
    if stage == 'select':
        return [sys.executable, 'scripts/select_best_snapshot_p1.py', '--exp-dir', str(run), '--protocol-id', PROTOCOL]
    if stage == 'test':
        return [sys.executable, 'test_card_spot.py', '--cfg', str(run/'request_config.json'),
                '--model', 'sgc_card', '--snapshot_path', '<resolved only from frozen selection>', '--split', 'test',
                '--result_json', str(run/'test_output/captions/controls_locked/sc_results.json')]
    return ['internal '+stage+' checks; no training subprocess']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('audit', 'preflight', 'train', 'select', 'freeze', 'test', 'summary'))
    parser.add_argument('--root', default='experiments/'+PROTOCOL)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--historical-root', help='Read-only audit of R2; never imported as new protocol runs')
    parser.add_argument('--semantic-source', action='append', default=[], metavar='DATASET=annotation|prediction|unknown')
    args = parser.parse_args()
    os.chdir(PROJECT)
    root = Path(args.root).resolve()
    runs = list(matrix(root, args.arms, args.datasets, args.seeds))
    if args.dry_run:
        # Before any output directory creation, lock inspection, data scan or subprocess.
        print(json.dumps({'stage': args.stage, 'protocol': PROTOCOL,
            'full_matrix_required': args.stage in ('preflight', 'freeze'),
            'runs': [{'dataset': d, 'seed': s, 'arm': a, 'directory': str(r),
                      'planned_command': planned_command(args.stage, r),
                      'configuration': cfg_to_plain(expected(root, d, s, a))} for d, s, a, r in runs]}, indent=2))
        return
    if args.historical_root:
        if args.stage != 'audit':
            raise ValueError('Historical root is audit only')
        print(json.dumps(historical_report(args.historical_root), indent=2))
        return
    experiment_parent = (PROJECT/'experiments').resolve()
    if root == experiment_parent or experiment_parent not in root.parents:
        raise ValueError('New protocol output must be an independent subdirectory of PROJECT_DIR/experiments')
    if args.stage == 'preflight':
        sources = dict(value.split('=', 1) for value in args.semantic_source)
        if set(sources)-set(DATASETS):
            raise ValueError('Unknown source dataset')
        preflight(root, sources)
        return
    lock = ensure_lock(root)
    if args.stage == 'audit':
        if (root/'frozen.json').exists():
            frozen = load_frozen(root, lock)
            verify_inputs(root, lock)
            report = {'protocol_audit_passed': True, 'errors': [], 'state': 'frozen evidence verified',
                      'validation': frozen['validation_statistics'],
                      'new_protocol_compatible_runs': [str(run) for _, _, _, run in matrix(root)]}
        else:
            report = run_audit(root, lock)
        write(root/'audit.json', report)
        print(json.dumps(report, indent=2))
    elif args.stage == 'freeze':
        freeze(root, lock)
    elif args.stage == 'train':
        verify_inputs(root, lock)
        for d, s, a, r in runs:
            ensure_lock(root)
            train_one(root, lock, d, s, a, r)
    elif args.stage == 'select':
        assert_unfrozen(root)
        for d, s, a, r in runs:
            execute([sys.executable, 'scripts/select_best_snapshot_p1.py', '--exp-dir', str(r),
                     '--protocol-id', PROTOCOL], r)
    elif args.stage == 'test':
        frozen = load_frozen(root, lock)
        verify_inputs(root, lock)
        for d, s, a, r in runs:
            ensure_lock(root)
            test_one(root, frozen, d, s, a, r)
    elif args.stage == 'summary':
        rows = []
        missing = []
        for d, s, a, r in runs:
            score = r/'test_metrics.json'
            if score.exists():
                receipt = read(r/'test_score_identity.json')
                if receipt['score_sha256'] != sha256_file(score):
                    raise ValueError('Score changed after completion')
                cfg = read(r/'request_config.json')
                expected_ids = [item['sample_id'] for item in read(root/('inputs_'+d+'.json'))['samples'] if item['split'] == 'test']
                observed = prediction_identity(r/'test_output/captions/controls_locked/sc_results.json',
                                               cfg['data']['eval_anno_path'], expected_ids)
                observed['checkpoint_sha256'] = read(r/'best_snapshot_p1.json')['best_snapshot_sha256']
                if observed != receipt['identity']:
                    raise ValueError('Scored prediction/reference identity changed')
                rows.append(dict(dataset=d, seed=s, arm=a, metrics=read(score)['metrics']))
            else:
                missing.append(str(r))
        result = statistics_report(rows)
        result['missing_runs'] = missing
        frozen_path = root/'frozen.json'
        result['validation_admitted'] = False
        result['protocol_audit_passed'] = False
        if frozen_path.exists():
            frozen = load_frozen(root, lock)
            result['validation_admitted'] = frozen['validation_admitted']
            result['protocol_audit_passed'] = True
        write(root/'summary.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
