#!/usr/bin/env python
"""Register, audit and lock E0/E1/E2; never start a stage implicitly."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROJECT = Path(os.environ.get('PROJECT_DIR') or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(PROJECT))
from scripts import run_semantic_controls as base
from utils import rsaca_position_protocol as protocol
from utils.semantic_controls import (POSITION_PROTOCOL as PROTOCOL, POSITION_ARMS as ARMS,
    POSITION_DATASETS as DATASETS, SEEDS, REFERENCE, position_configuration)
from utils.semantic_control_audit import (read, write, input_manifest, source_identity,
    prediction_identity, verify_frozen_files)
from utils.experiment_tracking import cfg_to_plain, stable_hash
from utils.checkpoint_integrity import sha256_file, validate_checkpoint_file


def expected(root, dataset, seed, arm):
    return position_configuration(PROJECT, root, dataset, seed, arm, os.environ)


def ensure_lock(root):
    lock = read(root/'protocol.json')
    protocol.verify_selection_registration(lock)
    source = source_identity(PROJECT)
    if lock['protocol_id'] != PROTOCOL or source != lock['source'] or source['dirty_execution']:
        raise ValueError('Position source changed since registration')
    if lock['runtime'] != protocol.runtime_identity():
        raise ValueError('Position runtime changed since registration')
    for d, s, a, run in protocol.matrix(root):
        if cfg_to_plain(expected(root, d, s, a)) != lock['configurations'][str(run.relative_to(root))]:
            raise ValueError('Position configuration/environment changed: '+str(run))
    return lock


def verify_inputs(root, lock):
    for d in DATASETS:
        manifest = read(root/('inputs_'+d+'.json'))
        if stable_hash(manifest) != lock['input_hashes'][d]:
            raise ValueError('Position input manifest changed: '+d)
        # E0 independently binds only public inputs; semantic arms bind both.
        for arm in ('E0', 'E1'):
            protocol.verify_manifest(expected(root, d, 1111, arm), manifest)


def preflight(root, sources):
    base.assert_unfrozen(root)
    if root.exists() and any(root.iterdir()) and not (root/'protocol.json').exists():
        raise ValueError('Nonempty position directory without a protocol lock')
    import torch
    if not torch.cuda.is_available() or not shutil.which('java'):
        raise RuntimeError('Training preflight requires existing CUDA and Java; use --dry-run for configuration inspection')
    source = source_identity(PROJECT)
    if source['dirty_execution']:
        raise ValueError('Commit execution source before registering a position experiment')
    lock = {'protocol_id': PROTOCOL, 'source': source, 'runtime': protocol.runtime_identity(),
            'reference': REFERENCE, 'selection_steps': list(range(1000, 10001, 1000)),
            'selection_tie_break': 'earlier_step', 'formal_test': 'all technically frozen runs',
            'environment': {'python': sys.version, 'executable': sys.executable},
            'configurations': {}, 'input_hashes': {}}
    manifests = {}
    for d in DATASETS:
        item = input_manifest(expected(root, d, 1111, 'E1'), sources.get(d, 'unknown'))
        if item['missing']:
            raise ValueError('Position missing inputs: '+str(item['missing'][:20]))
        manifests[d] = item
        lock['input_hashes'][d] = stable_hash(item)
    for d, s, a, run in protocol.matrix(root):
        lock['configurations'][str(run.relative_to(root))] = cfg_to_plain(expected(root, d, s, a))
    if (root/'protocol.json').exists():
        if read(root/'protocol.json') != lock:
            raise ValueError('Registered position lock differs; refusing overwrite')
        verify_inputs(root, lock)
        return
    for d, item in manifests.items():
        write(root/('inputs_'+d+'.json'), item, immutable=True)
    write(root/'protocol.json', lock, immutable=True)


def run_audit(root, lock):
    """Audit each full nine-run dataset without filtering by performance."""
    report = {'datasets': {}, 'files': {}, 'protocol_audit_passed': True, 'errors': []}
    for dataset in DATASETS:
        errors, files, rows, initial, configs = [], {}, [], {}, {}
        manifest = read(root/('inputs_'+dataset+'.json'))
        val_ids = [r['sample_id'] for r in manifest['samples'] if r['split'] == 'val']
        for d, seed, arm, run in protocol.matrix(root, datasets=(dataset,)):
            key = str(run.relative_to(root))
            try:
                request = read(run/'request_config.json')
                if request != lock['configurations'][key] or read(run/'resolved_config.json') != request:
                    raise ValueError('Position requested/resolved configuration mismatch')
                actual = read(run/'actual_model_config.json')
                base.actual_config_check(request, actual)
                configs[seed, arm] = actual
                provenance = read(run/'controls_provenance.json')
                if provenance['source'] != lock['source'] or provenance['input_hash'] != lock['input_hashes'][dataset]:
                    raise ValueError('Position source/input provenance mismatch')
                params = read(run/'initial_parameter_summary.json')
                if params['seed'] != seed:
                    raise ValueError('Position initial seed mismatch')
                initial.setdefault(seed, {})[arm] = params
                summary = read(run/'run_summary.json')
                if summary.get('status') != 'completed' or summary.get('final_global_step') != 10000:
                    raise ValueError('Position training incomplete')
                identities = read(run/'validation_prediction_identity.json')
                if sorted(map(int, identities)) != list(range(1000, 10001, 1000)):
                    raise ValueError('Position validation identity grid incomplete/duplicated')
                for identity in identities.values():
                    observed = prediction_identity(identity['prediction'], identity['reference'], val_ids)
                    if any(identity.get(k) != v for k, v in observed.items()):
                        raise ValueError('Position validation prediction/reference identity mismatch')
                    if Path(identity['reference']).resolve() != Path(request['data']['eval_anno_path']).resolve():
                        raise ValueError('Position validation uses another reference')
                    files[identity['prediction']] = observed['prediction_sha256']
                    files[identity['reference']] = observed['reference_sha256']
                selected = read(run/'best_snapshot_p1.json')
                with tempfile.TemporaryDirectory(dir=str(run)) as scratch:
                    output = Path(scratch)/'selection.json'
                    subprocess.run([sys.executable, str(PROJECT/'scripts/select_best_snapshot_p1.py'),
                        '--exp-dir', str(run), '--protocol-id', PROTOCOL, '--output-json', str(output)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if selected != read(output):
                        raise ValueError('Position selection differs from the complete validation grid')
                checkpoint = selected['best_snapshot']
                digest = validate_checkpoint_file(checkpoint, require_checksum=True, require_metadata=True,
                                                  expected_step=selected['best']['iter'])
                if digest != selected['best_snapshot_sha256']:
                    raise ValueError('Position selected checkpoint SHA mismatch')
                from utils.checkpointing import load_checkpoint_file
                content = load_checkpoint_file(checkpoint)
                if content['global_step'] != selected['best']['iter']:
                    raise ValueError('Position checkpoint step mismatch')
                if cfg_to_plain(content['config']) != actual:
                    raise ValueError('Position checkpoint actual configuration mismatch')
                for model_key, init_key in (('model_state_dict', 'change_detector'), ('speaker_state_dict', 'speaker')):
                    state = content[model_key]
                    expected_keys = set(params[init_key])
                    if model_key == 'speaker_state_dict':
                        expected_keys.add('core.position_enc.pe')
                    if set(state) != expected_keys:
                        raise ValueError('Position checkpoint learned parameter/buffer schema mismatch')
                    if model_key == 'speaker_state_dict':
                        pe = state['core.position_enc.pe']
                        dec = actual['model']['transformer_decoder']
                        if list(pe.shape) != [dec['seq_length'], dec['word_dim']] or str(pe.dtype) != 'torch.float32':
                            raise ValueError('Position decoder buffer shape/dtype mismatch')
                    import torch
                    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
                        raise ValueError('Position checkpoint has nonfinite tensors')
                    if any(n not in state or list(state[n].shape) != v['shape'] or
                            str(state[n].dtype) != v['dtype'] for n, v in params[init_key].items()):
                        raise ValueError('Position checkpoint parameter set/shape/dtype mismatch')
                training_evidence = run/'training_completion_evidence.json'
                if training_evidence.exists():
                    if read(training_evidence) != summary:
                        raise ValueError('Position training completion evidence changed')
                else:
                    write(training_evidence, summary, immutable=True)
                evidence = ['request_config.json', 'resolved_config.json', 'actual_model_config.json',
                    'controls_provenance.json', 'initial_parameter_summary.json', 'training_completion_evidence.json',
                    'command.txt', 'environment.txt', 'val_metrics.csv',
                    'validation_prediction_identity.json', 'best_snapshot_p1.json']
                mechanism = {'layout_passed': True, 'bypass_passed': True}
                if arm != 'E0':
                    mechanism = read(run/'position_mechanism.json')
                    if mechanism.get('checkpoint_sha256') != digest or mechanism.get('finite') is not True:
                        raise ValueError('Position mechanism checkpoint mismatch or nonfinite output')
                    if not all(isinstance(mechanism.get(k), bool) for k in ('layout_passed', 'bypass_passed')):
                        raise ValueError('Position mechanism result incomplete')
                    evidence.append('position_mechanism.json')
                # Bind selected bytes and sidecars; validation CSV binds all candidate hashes.
                for path in (checkpoint, checkpoint+'.sha256', checkpoint+'.metadata.json'):
                    files[str(path)] = sha256_file(path)
                for name in evidence:
                    files[str(run/name)] = sha256_file(run/name)
                rows.append({'dataset': dataset, 'seed': seed, 'arm': arm,
                             'metrics': selected['best']['metrics'], 'checkpoint_sha256': digest,
                             'selected_step': selected['best']['iter'], 'mechanism': mechanism})
            except (OSError, KeyError, ValueError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
                errors.append(key+': '+str(exc))
        for seed in SEEDS:
            try:
                protocol.verify_initialization(initial.get(seed, {}))
                protocol.verify_pair_configuration(configs[seed, 'E1'], configs[seed, 'E2'])
            except (ValueError, KeyError) as exc:
                errors.append('%s/%d: %s' % (dataset, seed, exc))
        complete = not errors and len(rows) == 9
        report['datasets'][dataset] = {'technical_complete': complete, 'errors': errors,
            'runs': rows, 'advancement': protocol.advancement(rows) if complete else None}
        if complete:
            report['files'].update(files)
        else:
            report['protocol_audit_passed'] = False
            report['errors'].extend(errors)
    return report


def freeze(root, lock):
    base.assert_unfrozen(root)
    verify_inputs(root, lock)
    audit = run_audit(root, lock)
    write(root/'audit.json', audit)
    if not any(d['technical_complete'] for d in audit['datasets'].values()):
        raise ValueError('No complete nine-run dataset can be technically frozen: '+str(audit['errors']))
    files = audit['files']
    for path in (root/'protocol.json', *(root/('inputs_'+d+'.json') for d in DATASETS)):
        files[str(path)] = sha256_file(path)
    write(root/'frozen.json', {'protocol_id': PROTOCOL, 'source': lock['source'],
        'files': files, 'datasets': audit['datasets'], 'validation_admitted': True}, immutable=True)
    write(root/'frozen_identity.json', {'sha256': sha256_file(root/'frozen.json')}, immutable=True)


def load_frozen(root, lock):
    if read(root/'frozen_identity.json') != {'sha256': sha256_file(root/'frozen.json')}:
        raise ValueError('Position frozen manifest changed')
    frozen = read(root/'frozen.json')
    if frozen['protocol_id'] != PROTOCOL or frozen['source'] != lock['source']:
        raise ValueError('Position frozen source/protocol mismatch')
    verify_frozen_files(frozen)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('preflight', 'train', 'select', 'audit', 'freeze', 'test', 'summary'))
    parser.add_argument('--root', default='experiments/'+PROTOCOL)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--semantic-source', action='append', default=[])
    args = parser.parse_args()
    os.chdir(PROJECT)
    root = Path(args.root).resolve()
    if root.name != PROTOCOL or (PROJECT/'experiments').resolve() not in root.parents:
        raise ValueError('Position outputs require their independent named protocol directory under experiments')
    runs = list(protocol.matrix(root, args.datasets, args.arms, args.seeds))
    if args.dry_run:
        import json
        print(json.dumps([cfg_to_plain(expected(root, d, s, a)) for d, s, a, _ in runs], indent=2))
        return
    if args.stage == 'preflight':
        sources = dict(item.split('=', 1) for item in args.semantic_source)
        if set(sources)-set(DATASETS):
            raise ValueError('Unknown position dataset source')
        preflight(root, sources)
        return
    lock = ensure_lock(root)
    if args.stage == 'train':
        verify_inputs(root, lock)
        for d, s, a, run in runs:
            base.train_one(root, lock, d, s, a, run)
    elif args.stage == 'select':
        base.assert_unfrozen(root)
        for d, s, a, run in runs:
            base.execute([sys.executable, 'scripts/select_best_snapshot_p1.py', '--exp-dir', str(run), '--protocol-id', PROTOCOL], run)
    elif args.stage == 'audit':
        verify_inputs(root, lock)
        report = load_frozen(root, lock) if (root/'frozen.json').exists() else run_audit(root, lock)
        write(root/'audit.json', report)
        print(report)
    elif args.stage == 'freeze':
        freeze(root, lock)
    elif args.stage == 'test':
        frozen = load_frozen(root, lock)
        for d, s, a, run in runs:
            if frozen['datasets'][d]['technical_complete']:
                base.test_one(root, frozen, d, s, a, run)
            else:
                print('Dataset not technically frozen: '+d)
    else:
        frozen = load_frozen(root, lock)
        rows = []
        for d, s, a, run in runs:
            if (run/'test_metrics.json').exists():
                cfg = expected(root, d, s, a)
                selected = read(run/'best_snapshot_p1.json')
                if protocol.enforce_position_admission(PROJECT, cfg, selected['best_snapshot'],
                        run/'test_output/captions/controls_locked/sc_results.json') != 'complete':
                    raise ValueError('Incomplete position scoring')
                rows.append({'dataset': d, 'seed': s, 'arm': a, 'metrics': read(run/'test_metrics.json')['metrics']})
        write(root/'summary.json', {'frozen_datasets': frozen['datasets'], 'test_runs': rows})


if __name__ == '__main__':
    main()
