"""Registered validation-only follow-up to the frozen semantic controls.

The old runner remains the authority for its old lock and test admission. This
module reuses its configurations, scorer, initialization and snapshot selector;
it never promotes a development run to a frozen test experiment.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar

from utils.semantic_controls import (ARMS, DATASETS, SEEDS, METRICS, PROTOCOL,
                                      FOLLOWUP_PROTOCOL, configuration)
from utils.semantic_control_audit import read, write, input_manifest, metrics
from utils.experiment_tracking import cfg_to_plain, stable_hash
from utils.checkpoint_integrity import sha256_file

PROJECT = Path(__file__).resolve().parents[1]
STAGES = ('P0', 'P1a', 'P1b', 'P1c', 'P1d', 'P2', 'P3')
FACTORS = {'semantic_fusion_dense_local_access': (False, True),
           'semantic_fusion_global_token_mode': ('all_mean', 'changed_mean')}
_INPUTS = ContextVar('followup_inputs', default=None)


def code_identity():
    # Include new/untracked implementation files as well as tracked execution
    # code; git revision alone does not identify a dirty working directory.
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                                    cwd=str(PROJECT), text=True).splitlines()
    files = {name: sha256_file(str(PROJECT / name)) for name in sorted(set(names))
             if Path(name).suffix in ('.py', '.yaml', '.yml', '.sh')
             and not name.startswith(('tests/', 'experiments/')) and (PROJECT/name).is_file()}
    return {'files': files, 'sha256': stable_hash(files)}


def runtime_identity():
    from utils.semantic_diagnostics import capture_runtime_identity, _java_identity
    result = capture_runtime_identity()
    import torch
    result.update(python_version=platform.python_version(), cuda=torch.version.cuda,
                  executable_sha256=sha256_file(sys.executable))
    java = _java_identity()
    if java.get('executable'):
        java['sha256'] = sha256_file(java['executable'])
    result['java'] = java
    result['determinism_environment'] = {key: os.environ.get(key, '') for key in
        ('PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES')}
    return result


def _verified_artifact(spec, name):
    """Return both identity values from one fresh file traversal."""
    record = spec['artifacts'][name]
    path = Path(record['path'])
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Missing/changed registered artifact: ' + name)
    digest = sha256_file(str(path))
    if digest != record['sha256']:
        raise ValueError('Missing/changed registered artifact: ' + name)
    if _INPUTS.get() is not None:
        _INPUTS.get()[name] = record['sha256']
    return path, digest


def artifact(spec, name):
    return _verified_artifact(spec, name)[0]


def _project_path(value):
    """Resolve configuration paths using the real subprocess working directory."""
    path = Path(value)
    return (path if path.is_absolute() else PROJECT/path).resolve()


def _compare_reference(reference, cfg):
    if reference.resolve() != _project_path(cfg.data.eval_anno_path):
        raise ValueError('Inference/scoring and comparison references differ')


def _verify_reference(spec, name, cfg):
    reference = artifact(spec, name)
    _compare_reference(reference, cfg)
    return reference


@contextmanager
def _dataset_working_directory():
    # The runner is a synchronous, single-threaded CLI. Limit the process-wide
    # cwd change to the legacy input resolver and restore it even on failure;
    # cfg/manifest path strings must retain their registered representation.
    previous = Path.cwd()
    try:
        os.chdir(PROJECT)
        yield
    finally:
        os.chdir(previous)


def _number(value, name, low=0, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Register a finite number for ' + name)
    if value < low or (high is not None and value > high):
        raise ValueError('Out-of-range ' + name)
    return value


def _ids(spec, name):
    ids = read(artifact(spec, name))
    if not isinstance(ids, list) or not ids or len(set(map(str, ids))) != len(ids):
        raise ValueError('Expected IDs must be a nonempty unique list')
    return list(map(str, ids))


def register(spec_path, output, execute=False):
    spec = read(spec_path)
    if spec.get('protocol_id') != FOLLOWUP_PROTOCOL or spec.get('validation_only') is not True:
        raise ValueError('Follow-up registration must be validation-only')
    if set(spec.get('stages', {})) != set(STAGES):
        raise ValueError('Declare every stage; unready parameters may be null')
    root = Path(spec['output_root'])
    if not root.is_absolute():
        raise ValueError('output_root must be absolute')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh follow-up output_root')
    for name in spec.get('artifacts', {}):
        path = artifact(spec, name)
        if root == path.parent or root in path.parents:
            raise ValueError('Inputs must be outside output_root')
    result = {'schema': 1, 'spec': spec, 'spec_sha256': stable_hash(spec), 'code': code_identity()}
    if execute:
        if Path(output).resolve() == root or root not in Path(output).resolve().parents:
            raise ValueError('Registration must be inside its new output_root')
        write(output, result, immutable=True)
    return result


def load_registration(path):
    lock = read(path)
    if lock.get('schema') != 1 or stable_hash(lock['spec']) != lock['spec_sha256']:
        raise ValueError('Registration content changed')
    if lock['spec']['protocol_id'] != FOLLOWUP_PROTOCOL or not lock['spec']['validation_only']:
        raise ValueError('Wrong follow-up protocol')
    if lock['code'] != code_identity():
        raise ValueError('Execution code changed since registration')
    return lock


def _stage_runtime(spec, stage, actual):
    label = spec['stages'][stage]['environment']
    if actual != spec['environments'].get(label):
        raise ValueError('Runtime differs from registered environment: ' + str(label))


def _receipt_status(stage, result):
    # Recompute eligibility from executed outcomes; a hand-written status is
    # never a dependency. Differences are completed evidence, not a pass.
    if result.get('error'):
        return 'FAILED'
    if stage == 'P1a':
        states = [job.get('status') for job in result.get('jobs', [])]
        if not states or any(state not in ('match', 'different') for state in states):
            return 'FAILED'
        return 'PASS' if all(state == 'match' for state in states) else 'DIFFERENT'
    if stage == 'P1b':
        return 'PASS' if result.get('jobs') and all(j['matched'] for j in result['jobs']) else 'DIFFERENT'
    if stage == 'P1c' and result.get('derived_not_required'):
        return 'NOT_REQUIRED'
    if stage == 'P1d':
        return 'READY_FOR_ANNOTATION' if result.get('status') == 'ok' else 'FAILED'
    return 'PASS' if result.get('verified') is True else 'FAILED'


def receipt(lock, stage, required=('PASS',)):
    spec = lock['spec']
    path = Path(spec['output_root']) / stage / 'receipt.json'
    record = read(path)
    if (record.get('spec_sha256') != lock['spec_sha256'] or record.get('code') != lock['code']
            or record.get('stage') != stage or not record.get('executed')):
        raise ValueError('Receipt identity/execution mismatch: ' + stage)
    _stage_runtime(spec, stage, record['runtime'])
    for name, digest in record['inputs'].items():
        _, actual_digest = _verified_artifact(spec, name)
        if actual_digest != digest:
            raise ValueError('Receipt input changed')
    for name, digest in record['outputs'].items():
        output = Path(spec['output_root']) / stage / name
        if not output.is_file() or sha256_file(str(output)) != digest:
            raise ValueError('Receipt output changed: ' + name)
    if not record['outputs'] or record['result'] != read(path.parent/'result.json'):
        raise ValueError('Receipt must bind its actual result output')
    expected_deps = [] if stage in ('P0', 'P1d') else ['P0']
    if stage in ('P1c', 'P2', 'P3'):
        expected_deps += ['P1a', 'P1b']
    if stage == 'P3':
        expected_deps += ['P2']
    if set(record.get('dependencies', {})) != set(expected_deps):
        raise ValueError('Receipt dependency set is incomplete')
    actual = _receipt_status(stage, record['result'])
    if actual != record['status'] or actual not in required:
        raise ValueError('Dependency did not pass: ' + stage + ' (' + actual + ')')
    for dep, digest in record.get('dependencies', {}).items():
        dep_path = Path(spec['output_root']) / dep / 'receipt.json'
        if sha256_file(str(dep_path)) != digest:
            raise ValueError('Dependency receipt changed')
        receipt(lock, dep, ('PASS', 'DIFFERENT', 'NOT_REQUIRED'))
    return record


def dependencies(lock, stage):
    root = Path(lock['spec']['output_root'])
    needed = []
    if stage in ('P1a', 'P1b', 'P1c', 'P2', 'P3'):
        receipt(lock, 'P0')
        needed.append('P0')
    if stage in ('P1c', 'P2', 'P3'):
        # P1c is conditional; structural development requires a reproduced
        # baseline. Training completion alone cannot resolve a discrepancy.
        allowed = ('PASS', 'DIFFERENT') if stage == 'P1c' else ('PASS',)
        for dep in ('P1a', 'P1b'):
            receipt(lock, dep, allowed)
            needed.append(dep)
    if stage == 'P3':
        receipt(lock, 'P2')
        needed.append('P2')
    return {dep: sha256_file(str(root/dep/'receipt.json')) for dep in needed}


def _cfg(path):
    import yaml
    from utils.attr_dict import AttrDict
    def convert(value):
        return AttrDict({k: convert(v) for k, v in value.items()}) if isinstance(value, dict) else value
    return convert(yaml.safe_load(Path(path).read_text(encoding='utf-8-sig')))


def verify_dataset(spec, dataset, cfg):
    if cfg.data.dataset != dataset:
        raise ValueError('Registered condition and resolved configuration differ')
    item = spec['datasets'][dataset]
    expected = read(artifact(spec, item['manifest']))
    with _dataset_working_directory():
        actual = input_manifest(cfg, item['source_kind'])
    if not cfg.data.use_semantic_maps:
        expected = copy.deepcopy(expected)
        for row in expected['samples']:
            row['semantic'] = []
    if actual['missing'] or actual != expected:
        raise ValueError('Resolved input bytes/mapping differ: ' + dataset)
    return actual


def candidate_configuration(project, root, dataset, seed, arm, environment, candidate,
                            constant_gate=1.0):
    """Derive matched G from C; never reuse an unmatched historical G."""
    changes = candidate.get('model', {})
    if len(changes) > 1 or any(k not in FACTORS or v not in FACTORS[k] for k, v in changes.items()):
        raise ValueError('Candidate must change at most one registered supported factor')
    _number(constant_gate, 'constant_gate', 0, 1)
    cfg = configuration(project, root, dataset, seed, arm, environment)
    cfg.train.protocol_id = FOLLOWUP_PROTOCOL
    if arm in ('rsaca', 'fixed_gate'):
        cfg.model.update(changes)
    if arm == 'fixed_gate':
        cfg.model.semantic_fusion_fixed_gate_value = float(constant_gate)
    return cfg


def _training_specs(lock, stage):
    spec, settings = lock['spec'], lock['spec']['stages'][stage]
    if settings.get('budget_steps') != 10000 or settings.get('selection_rule') != 'five_metric_equal_weight_log':
        raise ValueError('Explicitly register the existing 10000-step/grid and selection rule')
    datasets, seeds = settings.get('datasets'), settings.get('seeds')
    if not datasets or not seeds or len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds):
        raise ValueError('Register unique datasets and seeds')
    if not set(datasets) <= set(DATASETS) or not set(seeds) <= set(SEEDS):
        raise ValueError('Unknown registered condition/seed')
    candidates = settings.get('candidates')
    if stage == 'P3':
        if tuple(datasets) != DATASETS or tuple(seeds) != SEEDS:
            raise ValueError('P3 requires all three conditions and seeds in protocol order')
        best = receipt(lock, 'P2')['result']['selected_candidate']
        candidates = [best]
    if not candidates or len({c['name'] for c in candidates}) != len(candidates):
        raise ValueError('Register named unique candidates')
    if stage == 'P1c' and (len(candidates) != 1 or candidates[0]['model']):
        raise ValueError('Environment replay must retain the unmodified candidate')
    constants = settings.get('constant_gates')
    if stage == 'P2' and (not isinstance(constants, list) or not constants):
        raise ValueError('Register constant-g comparisons before development')
    if stage != 'P2':
        constants = [1.0]
    if len(set(constants)) != len(constants):
        raise ValueError('Duplicate constant gates')
    for value in constants:
        _number(value, 'constant_gates', 0, 1)
    for candidate in candidates:
        if not candidate['name'] or not all(c.isalnum() or c in '-_' for c in candidate['name']):
            raise ValueError('Unsafe candidate name')
        root = Path(spec['output_root']) / stage / candidate['name']
        for dataset in datasets:
            for seed in seeds:
                arms = ('card', 'rsaca') if stage == 'P1c' else ARMS
                for arm in arms:
                    gates = constants if arm == 'fixed_gate' else [1.0]
                    for gate in gates:
                        group = root / ('constant_' + str(gate)) if arm == 'fixed_gate' and stage == 'P2' else root
                        cfg = candidate_configuration(PROJECT, group, dataset, seed, arm,
                                                      spec['data_environment'], candidate, gate)
                        cfg.train.followup_stage = stage
                        yield candidate, dataset, seed, arm, gate, cfg


def training_configs(lock, registration, stage):
    for candidate, dataset, seed, arm, gate, cfg in _training_specs(lock, stage):
        cfg.train.followup_registration = str(Path(registration).resolve())
        cfg.train.followup_config_sha256 = ''
        cfg.train.followup_config_sha256 = stable_hash(cfg_to_plain(cfg))
        yield candidate, dataset, seed, arm, gate, cfg


def validate_training_registration(cfg):
    """Low-level pre-write guard, also applies when train is invoked directly."""
    lock = load_registration(cfg.train.followup_registration)
    stage = cfg.train.followup_stage
    if stage not in ('P1c', 'P2', 'P3'):
        raise ValueError('Unregistered follow-up training stage')
    dependencies(lock, stage)
    _stage_runtime(lock['spec'], stage, runtime_identity())
    actual = cfg_to_plain(cfg)
    for _, dataset, _, _, _, expected in training_configs(lock, cfg.train.followup_registration, stage):
        if actual == cfg_to_plain(expected):
            verify_dataset(lock['spec'], dataset, cfg)
            return
    raise ValueError('Training configuration is not derived from the sealed registration')


def validate_initialization_registration(cfg, summary):
    """Compare actual initialized tensors before the first optimizer update."""
    if cfg.train.followup_stage != 'P1c':
        return
    lock = load_registration(cfg.train.followup_registration)
    key = '%s/%s/%s' % (cfg.data.dataset, cfg.train.seed,
                       'card' if cfg.model.semantic_input_mode == 'none' else 'rsaca')
    name = lock['spec']['stages']['P1c']['initialization_references'][key]
    reference = read(artifact(lock['spec'], name))
    if (reference['seed'] != summary['seed'] or
            reference['change_detector'] != summary['change_detector'] or
            reference['speaker'] != summary['speaker']):
        raise ValueError('NOT_COMPARABLE: actual initial tensors differ from environment reference')


def _run(command, output):
    output.mkdir(parents=True, exist_ok=False)
    process = subprocess.run(command, cwd=str(PROJECT), text=True, capture_output=True)
    (output/'stdout.log').write_text(process.stdout, encoding='utf-8')
    (output/'stderr.log').write_text(process.stderr, encoding='utf-8')
    write(output/'command.json', {'command': command, 'returncode': process.returncode})
    if process.returncode:
        raise RuntimeError('Stage child process failed: ' + str(process.returncode))


def _score_command():
    return [sys.executable, str(PROJECT/'tools/score_predictions.py'),
            '--annotation', '{reference}', '--predictions', '{prediction}',
            '--stage', 'val', '--run-dir', '{run_dir}', '--output-json', '{output}']


def _decode_command(config_path, checkpoint, predictions):
    return [sys.executable, str(PROJECT/'test_card_spot.py'), '--cfg', str(config_path),
            '--checkpoint', str(checkpoint), '--split', 'val', '--result_json', str(predictions)]


def preview_stage(lock, stage):
    """Pure planning: inspect fields/receipts and render commands, never models."""
    spec, settings = lock['spec'], lock['spec']['stages'][stage]
    output = Path(spec['output_root'])/stage
    issues, commands, prerequisites = [], [], {}
    def require(key):
        value = settings.get(key)
        if value is None or value == '' or value == [] or value == {}:
            issues.append('Fill stages.%s.%s' % (stage, key))
        return value
    def number(key, high=None):
        try:
            _number(settings.get(key), key, 0, high)
        except ValueError as exc:
            issues.append(str(exc))
    def source(name):
        try:
            return artifact(spec, name)
        except (KeyError, TypeError, OSError, ValueError) as exc:
            issues.append('Artifact %r: %s' % (name, exc))
            return None
    if not isinstance(spec.get('environments', {}).get(settings.get('environment')), dict):
        issues.append('Fill the full registered environment for ' + stage)
    names = [] if stage in ('P0', 'P1d') else ['P0']
    if stage in ('P1c', 'P2', 'P3'):
        names += ['P1a', 'P1b']
    if stage == 'P3':
        names += ['P2']
    for name in names:
        try:
            allowed = ('PASS', 'DIFFERENT') if stage == 'P1c' and name != 'P0' else ('PASS',)
            prerequisites[name] = receipt(lock, name, allowed)['status']
        except (KeyError, OSError, TypeError, ValueError) as exc:
            prerequisites[name] = 'UNAVAILABLE: ' + str(exc)
            issues.append('Required receipt %s is not ready' % name)
    skip_training = stage == 'P1c' and all(prerequisites.get(n) == 'PASS' for n in ('P0', 'P1a', 'P1b'))
    if stage == 'P0':
        for key in ('required_artifacts', 'initialization_summaries'):
            for name in require(key) or []:
                source(name)
        for item in require('checkpoints') or []:
            source(item.get('artifact'))
            if type(item.get('step')) is not int or item['step'] not in range(1000, 10001, 1000):
                issues.append('Register a checkpoint step from the validation grid')
        if not spec.get('datasets'):
            issues.append('Register datasets and actual input manifests')
        for item in spec.get('datasets', {}).values():
            source(item.get('config')); source(item.get('manifest'))
            if item.get('source_kind') not in ('annotation', 'prediction', 'unknown'):
                issues.append('Register a valid semantic source_kind')
        for pair in settings.get('protocol_pairs', []):
            source(pair.get('first')); source(pair.get('second'))
    elif stage in ('P1a', 'P1b'):
        number('metric_tolerance' if stage == 'P1a' else 'max_changed_fraction', None if stage == 'P1a' else 1)
        for i, job in enumerate(require('jobs') or []):
            keys = ['prediction', 'reference', 'expected_ids']
            keys += ['original_metrics'] if stage == 'P1a' else ['config', 'checkpoint']
            paths = {key: source(job.get(key)) for key in keys}
            if job.get('dataset') not in spec.get('datasets', {}):
                issues.append('Register job dataset and its input manifest')
            else:
                item = spec['datasets'][job['dataset']]
                manifest_path = source(item.get('manifest'))
                dataset_config = source(item.get('config'))
                reference_config = paths.get('config') if stage == 'P1b' else dataset_config
                if reference_config and paths['reference']:
                    try:
                        _compare_reference(paths['reference'], _cfg(reference_config))
                    except (KeyError, TypeError, OSError, ValueError) as exc:
                        issues.append(str(exc))
                if manifest_path and paths['expected_ids']:
                    try:
                        ids = _ids(spec, job['expected_ids'])
                        expected = [r['sample_id'] for r in read(manifest_path)['samples'] if r['split'] == 'val']
                        if sorted(ids) != sorted(expected):
                            issues.append('Job %d IDs do not cover the complete validation split' % i)
                    except (KeyError, TypeError, ValueError) as exc:
                        issues.append(str(exc))
            if stage == 'P1a':
                if not job.get('scorer_resources'):
                    issues.append('Register scorer_resources for job %d' % i)
                for name in job.get('scorer_resources', []):
                    source(name)
                if all(paths.values()):
                    replacements = {'{prediction}': str(paths['prediction']), '{reference}': str(paths['reference']),
                                    '{run_dir}': str(output/('score_%d' % i)),
                                    '{output}': str(output/('score_%d' % i)/'metrics.json')}
                    commands.append([replacements.get(token, token) for token in _score_command()])
            elif all(paths.values()):
                run = output/('decode_%d' % i)
                commands.append(_decode_command(run/'decode.yaml', paths['checkpoint'],
                                               run/'evaluation/captions/val/predictions.json'))
    elif stage == 'P1d':
        if type(require('sample_count')) is not int or settings['sample_count'] < 1:
            issues.append('sample_count must be a positive integer')
        if type(require('sample_seed')) is not int:
            issues.append('sample_seed must be an integer')
        for key in ('expected_ids', 'reference', 'image_manifest'):
            source(require(key))
        for name in (require('predictions') or {}).values():
            source(name)
    elif not skip_training:
        if require('budget_steps') != 10000:
            issues.append('budget_steps must explicitly equal 10000')
        if require('selection_rule') != 'five_metric_equal_weight_log':
            issues.append('selection_rule must use the fixed five-metric log rule')
        datasets, seeds = require('datasets'), require('seeds')
        datasets = datasets if isinstance(datasets, list) else []
        seeds = seeds if isinstance(seeds, list) else []
        if len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
            issues.append('Invalid or duplicate datasets')
        if len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
            issues.append('Invalid or duplicate seeds')
        datasets = [value for value in datasets if value in DATASETS]
        if any(type(value) is not int for value in seeds):
            issues.append('Seeds must be integers')
        seeds = [value for value in seeds if type(value) is int and value in SEEDS]
        candidates = settings.get('candidates')
        if stage == 'P3':
            if tuple(datasets) != DATASETS or tuple(seeds) != SEEDS:
                issues.append('P3 requires all three conditions and seeds')
            candidates = [receipt(lock, 'P2')['result']['selected_candidate']] if prerequisites.get('P2') == 'PASS' else []
        elif not candidates:
            issues.append('Register named candidates')
        if candidates is not None and not isinstance(candidates, list):
            issues.append('candidates must be a list'); candidates = []
        gates = settings.get('constant_gates') if stage == 'P2' else [1.0]
        if not isinstance(gates, list) or not gates:
            issues.append('Register constant_gates'); gates = []
        valid_gates = []
        for gate in gates:
            try:
                _number(gate, 'constant_gates', 0, 1)
                valid_gates.append(gate)
            except ValueError as exc:
                issues.append(str(exc))
        gates = valid_gates
        if len(set(gates)) != len(gates):
            issues.append('Duplicate constant gates')
        if stage == 'P1c' and (not candidates or len(candidates) != 1 or candidates[0].get('model')):
            issues.append('P1c must keep one unmodified candidate')
        index = 0
        candidate_names = []
        for candidate in candidates or []:
            if not isinstance(candidate, dict) or not isinstance(candidate.get('model'), dict):
                issues.append('Candidates must contain a model mapping'); continue
            changes = candidate.get('model', {})
            name = candidate.get('name', '')
            if (not isinstance(name, str) or not name or name in candidate_names or not all(c.isalnum() or c in '-_' for c in name) or len(changes) > 1
                    or any(k not in FACTORS or v not in FACTORS[k] for k, v in changes.items())):
                issues.append('Invalid candidate: ' + str(candidate)); continue
            candidate_names.append(name)
            for dataset in datasets:
                item = spec.get('datasets', {}).get(dataset, {})
                source(item.get('manifest')); source(item.get('config'))
                for seed in seeds:
                    for arm in (('card', 'rsaca') if stage == 'P1c' else ARMS):
                        if stage == 'P1c':
                            key = '%s/%s/%s' % (dataset, seed, arm)
                            source(settings.get('initialization_references', {}).get(key))
                        for gate in (gates if arm == 'fixed_gate' else [1.0]):
                            group = output/name
                            if arm == 'fixed_gate' and stage == 'P2':
                                group = group/('constant_' + str(gate))
                            run = group/arm/('%s_%s_seed%d' % (arm, dataset, seed))
                            commands += [[sys.executable, str(PROJECT/'train_card_spot.py'), '--cfg', str(output/('config_%d.yaml' % index))],
                                         [sys.executable, str(PROJECT/'scripts/select_best_snapshot_p1.py'), '--exp-dir', str(run), '--protocol-id', FOLLOWUP_PROTOCOL]]
                            index += 1
    if output.exists():
        issues.append('Output directory already exists; preserve it and use a new registration/root')
    return {'stage': stage, 'status': 'NOT_RUN', 'output': str(output), 'issues': sorted(set(issues)),
            'prerequisites': prerequisites, 'commands': commands,
            'readiness': 'BLOCKED' if issues else ('NOT_REQUIRED' if skip_training else 'READY_FOR_EXECUTION_CHECKS'),
            'remaining_checks': 'Actual runtime, resolved datasets, checkpoint payload and scientific outputs are checked only on execution'}


def _p0(spec, settings):
    if not settings.get('required_artifacts') or not spec.get('datasets'):
        raise ValueError('Register source artifacts and dataset manifests')
    for name in settings['required_artifacts']:
        artifact(spec, name)
    manifests = {}
    for dataset, item in spec['datasets'].items():
        manifests[dataset] = stable_hash(verify_dataset(spec, dataset, _cfg(artifact(spec, item['config']))))
    # Full source files are required, not only audit-declared hashes. Summaries
    # remain summaries; no claim that they prove missing initialization tensors.
    if not settings.get('initialization_summaries') or not settings.get('checkpoints'):
        raise ValueError('Register available initialization summaries and checkpoint files')
    from utils.checkpoint_integrity import validate_checkpoint_file
    for name in settings['initialization_summaries']:
        value = read(artifact(spec, name))
        if not value.get('change_detector') or not value.get('speaker'):
            raise ValueError('Incomplete initialization summary')
    for item in settings['checkpoints']:
        validate_checkpoint_file(str(artifact(spec, item['artifact'])), require_checksum=True,
                                 require_metadata=True, expected_step=item['step'])
    comparisons = []
    for pair in settings.get('protocol_pairs', []):
        first, second = (read(artifact(spec, pair[k])) for k in ('first', 'second'))
        def changes(a, b, prefix=''):
            if isinstance(a, dict) and isinstance(b, dict):
                return [row for key in sorted(set(a) | set(b)) for row in
                        changes(a.get(key), b.get(key), prefix + ('.' if prefix else '') + key)]
            return [] if a == b else [{'field': prefix, 'first': a, 'second': b}]
        a, b = first['source']['execution_files'], second['source']['execution_files']
        comparisons.append({'first': pair['first'], 'second': pair['second'],
            'evidence_level': 'record declarations; actual source bytes are verified only when separately registered',
            'common_source_files': len(set(a) & set(b)),
            'matching_source_hashes': sum(a[k] == b[k] for k in set(a) & set(b)),
            'source_differences': changes(a, b),
            'configuration_differences': changes(first['configurations'], second['configurations']),
            'runtime_differences': changes(first['runtime'], second['runtime']),
            'input_hash_declarations_equal': first['input_hashes'] == second['input_hashes'],
            'environment_causality': 'not established'})
    return {'verified': True, 'input_manifests': manifests, 'protocol_comparisons': comparisons,
            'initialization_evidence': 'actual summary files; equality of missing raw tensors is not asserted'}


def _p1a(spec, settings, output):
    from utils.semantic_diagnostics import reproduce_scores
    tolerance = _number(settings.get('metric_tolerance'), 'metric_tolerance')
    if not settings.get('jobs'):
        raise ValueError('Register score jobs')
    results = []
    for i, job in enumerate(settings['jobs']):
        dataset = job['dataset']
        manifest = read(artifact(spec, spec['datasets'][dataset]['manifest']))
        ids = _ids(spec, job['expected_ids'])
        if sorted(ids) != sorted(r['sample_id'] for r in manifest['samples'] if r['split'] == 'val'):
            raise ValueError('Scoring IDs must cover the registered validation split')
        config = _cfg(artifact(spec, spec['datasets'][dataset]['config']))
        reference = _verify_reference(spec, job['reference'], config)
        # Use the same interpreter and project scorer. Actual scorer dependencies
        # and declared resources are bound, rather than a user PASS or command.
        if not job.get('scorer_resources'):
            raise ValueError('Register scorer resources (e.g. Java/SPICE jars)')
        for name in job['scorer_resources']:
            artifact(spec, name)
        command = _score_command()
        result = reproduce_scores(artifact(spec, job['prediction']), reference,
                    expected_ids=ids, scorer_command=command,
                    original_score_path=artifact(spec, job['original_metrics']),
                    tolerance=tolerance, spice=True, output_dir=output/('score_%d' % i))
        results.append(result)
    return {'jobs': results}


def _p1b(spec, settings, output):
    import yaml
    from scripts.diagnose_semantic_controls import _load_checkpoint_for_diagnostic, _checkpoint_config_compatibility
    from utils.semantic_diagnostics import validate_prediction_ids, _prediction_rows, _caption_text
    limit = _number(settings.get('max_changed_fraction'), 'max_changed_fraction', 0, 1)
    if not settings.get('jobs'):
        raise ValueError('Register complete validation decoding jobs')
    results = []
    for i, job in enumerate(settings['jobs']):
        cfg = _cfg(artifact(spec, job['config']))
        manifest = verify_dataset(spec, job['dataset'], cfg)
        ids = _ids(spec, job['expected_ids'])
        if sorted(ids) != sorted(r['sample_id'] for r in manifest['samples'] if r['split'] == 'val'):
            raise ValueError('P1b IDs must cover the entire registered validation split')
        reference = _verify_reference(spec, job['reference'], cfg)
        checkpoint = artifact(spec, job['checkpoint'])
        loaded = _load_checkpoint_for_diagnostic(checkpoint, require_integrity=True)
        recorded = copy.deepcopy(loaded['payload']['config'])
        # These additive defaults have no state tensors and preserve old output.
        for item in (recorded, cfg):
            item['model'].setdefault('semantic_fusion_fixed_gate_value', 1.0)
            item['model'].setdefault('semantic_fusion_dense_local_access', False)
            for key in ('followup_registration', 'followup_stage', 'followup_config_sha256'):
                item['train'].setdefault(key, '')
        compatible = _checkpoint_config_compatibility(recorded, cfg)
        if compatible['status'] != 'match':
            raise ValueError('Checkpoint/config identity mismatch: ' + str(compatible['mismatches']))
        baseline = artifact(spec, job['prediction'])
        if validate_prediction_ids(baseline, reference, ids)['status'] != 'ok':
            raise ValueError('Historical prediction IDs are incomplete')
        run = output/('decode_%d' % i)
        run.mkdir()
        cfg.exp_dir, cfg.exp_name = str(run), 'isolated_validation'
        # The legacy entry derives eval_output_dir two parents above the
        # caption directory. Keep that derived path inside this isolated run.
        config_path, predictions = run/'decode.yaml', run/'evaluation/captions/val/predictions.json'
        config_path.write_text(yaml.safe_dump(cfg_to_plain(cfg)), encoding='utf-8')
        _run(_decode_command(config_path, checkpoint, predictions), run/'process')
        identity = validate_prediction_ids(predictions, reference, ids)
        if identity['status'] != 'ok':
            raise ValueError('Decoded validation IDs are incomplete')
        first = {str(r['image_id']): _caption_text(r) for r in _prediction_rows(read(baseline))}
        second = {str(r['image_id']): _caption_text(r) for r in _prediction_rows(read(predictions))}
        changed = sum(first[k] != second[k] for k in ids)
        results.append({'matched': changed/len(ids) <= limit, 'changed': changed,
                        'count': len(ids), 'identity': identity, 'checkpoint': loaded['identity'],
                        'config_compatibility': compatible})
    return {'jobs': results}


def _p1d(spec, settings, output):
    from utils.semantic_diagnostics import analyze_content, validate_prediction_ids
    count, seed = settings.get('sample_count'), settings.get('sample_seed')
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError('Register sample_count and sample_seed')
    ids, reference = _ids(spec, settings['expected_ids']), artifact(spec, settings['reference'])
    if count > len(ids):
        raise ValueError('Sample count exceeds registered population')
    images = read(artifact(spec, settings['image_manifest']))
    if set(images) != set(ids):
        raise ValueError('Images must cover expected IDs')
    for pair in images.values():
        if set(pair) != {'before', 'after'}:
            raise ValueError('Register both images per sample')
        for name in pair.values():
            artifact(spec, name)
    predictions = {arm: artifact(spec, name) for arm, name in settings['predictions'].items()}
    for path in predictions.values():
        if validate_prediction_ids(path, reference, ids)['status'] != 'ok':
            raise ValueError('Content review prediction IDs differ')
    result = analyze_content(predictions, sample_seed=seed, sample_count=count,
                             failure_ids=settings.get('failure_ids', []))
    # Blind model names in the review packet; retain the key separately for
    # analysis after human annotation. This is preparation, not completed labels.
    names = sorted(predictions)
    import random
    random.Random(seed).shuffle(names)
    aliases = {name: 'system_%d' % i for i, name in enumerate(names)}
    selected = sorted(set(result['samples']['failure_cases'] + result['samples']['overview_sample']))
    refs = read(reference)['annotations']
    packet = [{'sample_id': sid, 'images': {k: str(artifact(spec, v)) for k, v in images[sid].items()},
               'references': [r['caption'] for r in refs if str(r['image_id']) == sid],
               'captions': {aliases[a]: result['sample_captions'][a][sid] for a in names},
               'labels': None} for sid in selected]
    write(output/'blind_packet.json', packet)
    write(output/'blinding_key.json', aliases)
    result['annotation_status'] = 'NOT_RUN'
    return result


def _train(lock, registration, stage, output):
    import yaml
    from utils.semantic_control_audit import prediction_identity, statistics_report, verify_initialization
    spec = lock['spec']
    if stage == 'P1c':
        if all(receipt(lock, s, ('PASS', 'DIFFERENT'))['status'] == 'PASS' for s in ('P1a', 'P1b')):
            return {'derived_not_required': True, 'reason': 'Registered scoring and full validation decoding both matched'}
    rows, summaries = [], {}
    plans = list(training_configs(lock, registration, stage))
    # Resolve every source and every plan before creating the first training run.
    for _, dataset, _, _, _, cfg in plans:
        verify_dataset(spec, dataset, cfg)
        if stage == 'P1c':
            key = '%s/%s/%s' % (dataset, cfg.train.seed,
                'card' if cfg.model.semantic_input_mode == 'none' else 'rsaca')
            artifact(spec, spec['stages'][stage]['initialization_references'][key])
    for candidate, dataset, seed, arm, gate, cfg in plans:
        run = Path(cfg.exp_dir)/cfg.exp_name
        if run.exists():
            raise ValueError('Refusing to reuse an existing training run')
        config_path = output/('config_%d.yaml' % len(rows))
        config_path.write_text(yaml.safe_dump(cfg_to_plain(cfg)), encoding='utf-8')
        _run([sys.executable, str(PROJECT/'train_card_spot.py'), '--cfg', str(config_path)],
             output/('train_process_%d' % len(rows)))
        _run([sys.executable, str(PROJECT/'scripts/select_best_snapshot_p1.py'), '--exp-dir', str(run),
              '--protocol-id', FOLLOWUP_PROTOCOL], output/('select_process_%d' % len(rows)))
        selected = read(run/'best_snapshot_p1.json')
        identities = read(run/'validation_prediction_identity.json')
        if set(identities) != {str(s) for s in range(1000, 10001, 1000)}:
            raise ValueError('Missing validation prediction identities')
        manifest = read(artifact(spec, spec['datasets'][dataset]['manifest']))
        ids = [r['sample_id'] for r in manifest['samples'] if r['split'] == 'val']
        for item in identities.values():
            actual = prediction_identity(item['prediction'], cfg.data.eval_anno_path, ids)
            actual.update(prediction=item['prediction'], reference=str(Path(cfg.data.eval_anno_path).resolve()))
            if actual != item:
                raise ValueError('Validation prediction identity changed')
        summary = read(run/'initial_parameter_summary.json')
        if summary['protocol_id'] != FOLLOWUP_PROTOCOL or summary['seed'] != seed:
            raise ValueError('Wrong initialization identity')
        validate_initialization_registration(cfg, summary)
        summaries[candidate['name'], dataset, seed, arm, gate] = summary
        rows.append({'candidate': candidate, 'dataset': dataset, 'seed': seed, 'arm': arm,
                     'constant_gate': gate, 'metrics': metrics(selected['best']['metrics']),
                     'selected': selected})
    if stage != 'P1c':
        groups = {(r['candidate']['name'], r['dataset'], r['seed']) for r in rows}
        for name, dataset, seed in groups:
            fixed = [k[-1] for k in summaries if k[:3] == (name, dataset, seed) and k[3] == 'fixed_gate']
            for gate in fixed:
                values = {arm: copy.deepcopy(summaries[name, dataset, seed, arm, gate if arm == 'fixed_gate' else 1.0]) for arm in ARMS}
                for value in values.values():
                    value['protocol_id'] = PROTOCOL  # reuse tensor equality check, after new-ID validation above
                verify_initialization(values)
    result = {'verified': True, 'rows': rows, 'test_status': 'PROHIBITED',
              'causal_environment_explanation': 'not inferred from training completion'}
    if stage == 'P2':
        candidates = {r['candidate']['name']: r['candidate'] for r in rows}
        from utils.semantic_controls import REFERENCE
        scores = {name: sum(sum(math.log(max(r['metrics'][m], 1e-12)/REFERENCE[m]) for m in METRICS)/5
                           for r in rows if r['candidate']['name'] == name and r['arm'] == 'rsaca') /
                       sum(r['candidate']['name'] == name and r['arm'] == 'rsaca' for r in rows) for name in candidates}
        best = sorted(scores, key=lambda name: (-scores[name], name))[0]
        result.update(candidate_scores=scores, selected_candidate=candidates[best], tie_break='lexical candidate name')
    if stage == 'P3':
        result['statistics'] = statistics_report(rows)
        if len(rows) != 36:
            raise ValueError('P3 must have exactly 36 matched runs')
    return result


def execute_stage(registration, stage, execute=False):
    lock = load_registration(registration)
    spec, settings = lock['spec'], lock['spec']['stages'][stage]
    output = Path(spec['output_root']) / stage
    if not execute:
        return preview_stage(lock, stage)
    if output.exists():
        raise ValueError('Stage output already exists; preserve it and use a new registration/root')
    output.mkdir(parents=True)
    record = {'stage': stage, 'spec_sha256': lock['spec_sha256'], 'code': lock['code'],
              'executed': True, 'started_unix': time.time(), 'inputs': {}, 'dependencies': {}}
    tracker = _INPUTS.set(record['inputs'])
    try:
        record['runtime'] = runtime_identity()
        _stage_runtime(spec, stage, record['runtime'])
        # Bind only actual inputs used by this stage and its prerequisites;
        # P1d may run before unrelated scoring/training artifacts are available.
        record['dependencies'] = dependencies(lock, stage)
        if stage == 'P0':
            result = _p0(spec, settings)
        elif stage == 'P1a':
            result = _p1a(spec, settings, output)
        elif stage == 'P1b':
            result = _p1b(spec, settings, output)
        elif stage == 'P1d':
            result = _p1d(spec, settings, output)
        else:
            result = _train(lock, registration, stage, output)
        for name in record['inputs']:
            artifact(spec, name)
        if code_identity() != lock['code']:
            raise ValueError('Code changed during execution')
    except Exception as exc:
        result = {'error': type(exc).__name__ + ': ' + str(exc)}
    finally:
        _INPUTS.reset(tracker)
    write(output/'result.json', result)
    record.update(result=result, status=_receipt_status(stage, result), elapsed_seconds=time.time()-record['started_unix'])
    record['outputs'] = {str(p.relative_to(output)): sha256_file(str(p)) for p in output.rglob('*') if p.is_file()}
    write(output/'receipt.json', record, immutable=True)
    return record
