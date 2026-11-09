"""Immutable registration and admission for the E0/E1/E2 position experiment.

Technical completeness and validation advancement are separate decisions. Scores
and failed finite mechanism probes never change the selected seed/checkpoint.
"""
import copy
import math
from pathlib import Path
import statistics

from utils.semantic_controls import (POSITION_PROTOCOL, POSITION_ARMS, POSITION_DATASETS,
                                     SEEDS, METRICS, REFERENCE, position_configuration)
from utils.semantic_control_audit import (read, write, input_manifest, source_identity,
    prediction_identity, verify_frozen_files, metrics)
from utils.experiment_tracking import cfg_to_plain, stable_hash
from utils.checkpoint_integrity import sha256_file, validate_checkpoint_file


def runtime_identity():
    from utils.semantic_followup import runtime_identity as capture
    return capture()


def guard_output_route(cfg, split, result_path):
    run = (Path(str(cfg.exp_dir))/str(cfg.exp_name)).resolve()
    output = Path(str(result_path)).resolve()
    root = position_root(run) or position_root(output)
    if root is None and getattr(cfg.train, 'protocol_id', '') != POSITION_PROTOCOL:
        return
    if root is None or getattr(cfg.train, 'protocol_id', '') != POSITION_PROTOCOL:
        raise ValueError('Protected position outputs require registered protocol identity')
    if run not in output.parents:
        raise ValueError('Position output must remain inside its registered run')
    if split != 'test':
        validation_root = run/(str(split)+'_output')
        if validation_root not in output.parents:
            raise ValueError('Validation cannot write outside its isolated output directory')
        if output.exists():
            raise ValueError('Position validation output exists; refusing overwrite')


def matrix(root, datasets=POSITION_DATASETS, arms=POSITION_ARMS, seeds=SEEDS):
    for dataset in datasets:
        for seed in seeds:
            for arm in arms:
                yield dataset, seed, arm, Path(root)/arm/('%s_%s_seed%d' % (arm, dataset, seed))


def position_root(path):
    """Recognize protected paths even when a caller claims a legacy protocol."""
    path = Path(path).resolve()
    for candidate in (path, *path.parents):
        if candidate.name == POSITION_PROTOCOL:
            return candidate
        lock = candidate/'protocol.json'
        if lock.is_file() and read(lock).get('protocol_id') == POSITION_PROTOCOL:
            return candidate
    return None


def verify_selection_registration(lock):
    if (lock.get('reference') != REFERENCE
            or lock.get('selection_steps') != list(range(1000, 10001, 1000))
            or lock.get('selection_tie_break') != 'earlier_step'):
        raise ValueError('Position lock changes the registered selection rule')


def guard_position_selection(run, protocol_id, csv=None, reference=None):
    root = position_root(run)
    if root is None and protocol_id != POSITION_PROTOCOL:
        return
    if root is None or protocol_id != POSITION_PROTOCOL:
        raise ValueError('Position selection requires its registered protocol identity')
    if (root/'frozen.json').exists():
        raise ValueError('Position protocol is frozen; reselection is forbidden')
    if csv is not None or reference is not None:
        raise ValueError('Position protocol fixes the validation CSV and reference values')
    lock = read(root/'protocol.json')
    verify_selection_registration(lock)
    key = str(Path(run).resolve().relative_to(root))
    if key not in lock['configurations']:
        raise ValueError('Unregistered position selection run')


def common_manifest(manifest):
    """E0 binds every ID/visual/reference without requiring semantic columns."""
    return {'artifacts': manifest['artifacts'], 'samples': [
        {key: row[key] for key in ('split', 'index', 'sample_id', 'visual')}
        for row in manifest['samples']]}


def verify_manifest(cfg, registered):
    actual = input_manifest(cfg, registered.get('source_kind', 'unknown'))
    if actual.get('missing') or common_manifest(actual) != common_manifest(registered):
        raise ValueError('Common input manifest changed or missing')
    if cfg.data.use_semantic_maps and actual != registered:
        raise ValueError('Semantic input manifest changed or missing')
    return actual


def verify_initialization(summaries):
    if set(summaries) != set(POSITION_ARMS):
        raise ValueError('Position initialization requires all E0/E1/E2 seeds')
    public = summaries['E0']
    for arm, item in summaries.items():
        if item.get('protocol_id') != POSITION_PROTOCOL or item['seed'] != public['seed']:
            raise ValueError('Position initial identity mismatch: '+arm)
        if not public['speaker'] or item['speaker'] != public['speaker']:
            raise ValueError('Position decoder initialization mismatch: '+arm)
        if not public['change_detector'] or any(item['change_detector'].get(k) != v
                for k, v in public['change_detector'].items()):
            raise ValueError('Position public initialization mismatch: '+arm)
    if summaries['E1']['change_detector'] != summaries['E2']['change_detector']:
        raise ValueError('E1/E2 adapter initialization mismatch')
    if not set(summaries['E1']['change_detector']) - set(public['change_detector']):
        raise ValueError('Missing position adapter parameter evidence')


def verify_pair_configuration(first, second):
    a, b = copy.deepcopy(first), copy.deepcopy(second)
    for cfg in (a, b):
        cfg.pop('exp_name', None)
        cfg.pop('exp_dir', None)
        cfg['train'].pop('semantic_control_arm', None)
    if a['model'].pop('semantic_fusion_local_position') != 'none' or b['model'].pop(
            'semantic_fusion_local_position') != 'sincos2d_kv' or a != b:
        raise ValueError('E1/E2 differ beyond the registered position switch')


def advancement(rows):
    values = {(r['arm'], int(r['seed'])): metrics(r['metrics']) for r in rows}
    paired = {seed: {m: values['E2', seed][m]-values['E1', seed][m] for m in METRICS}
              for seed in SEEDS}
    means = {m: statistics.mean(paired[s][m] for s in SEEDS) for m in METRICS}
    mechanism = all(r['mechanism'].get('layout_passed') is True and
                    r['mechanism'].get('bypass_passed') is True for r in rows if r['arm'] == 'E2') and all(
                    r['mechanism'].get('bypass_passed') is True for r in rows if r['arm'] == 'E1')
    advance = mechanism and means['CIDEr'] > 0 and sum(paired[s]['CIDEr'] > 0 for s in SEEDS) >= 2 and means['SPICE'] >= 0
    return {'advance': advance, 'mechanism_passed': mechanism,
            'paired_E2_minus_E1': paired, 'mean': means,
            'sample_std': {m: statistics.stdev(paired[s][m] for s in SEEDS) for m in METRICS},
            'positive_seeds': {m: sum(paired[s][m] > 0 for s in SEEDS) for m in METRICS},
            'reason': 'registered candidate criterion passed' if advance else 'registered candidate criterion not passed',
            'formal_test': 'all technically frozen runs, including regressions'}


def guard_position_training(cfg):
    run = Path(str(cfg.exp_dir))/str(cfg.exp_name)
    root = position_root(run)
    protocol = str(getattr(cfg.train, 'protocol_id', 'legacy'))
    if root is None and protocol != POSITION_PROTOCOL:
        return
    if root is None or protocol != POSITION_PROTOCOL:
        raise ValueError('Position training requires registered protocol identity')
    if (root/'frozen.json').exists():
        raise ValueError('Position protocol is frozen; further training is forbidden')
    lock = read(root/'protocol.json')
    key = str(run.resolve().relative_to(root))
    if cfg_to_plain(cfg) != lock['configurations'].get(key):
        raise ValueError('Position training configuration is not registered')


def enforce_position_admission(project, cfg, checkpoint, result_path):
    run = (Path(str(cfg.exp_dir))/str(cfg.exp_name)).resolve()
    root = position_root(run)
    if root is None or str(getattr(cfg.train, 'protocol_id', 'legacy')) != POSITION_PROTOCOL:
        raise ValueError('Position test requires registered protocol identity')
    if not all((root/n).is_file() for n in ('protocol.json', 'frozen.json', 'frozen_identity.json')):
        raise ValueError('Position test requires protocol.json and a technical freeze')
    if read(root/'frozen_identity.json') != {'sha256': sha256_file(root/'frozen.json')}:
        raise ValueError('Position frozen identity changed')
    lock, frozen = read(root/'protocol.json'), read(root/'frozen.json')
    if lock.get('protocol_id') != POSITION_PROTOCOL or frozen.get('protocol_id') != POSITION_PROTOCOL:
        raise ValueError('Position protocol identity mismatch')
    if source_identity(project) != lock['source'] or lock['source'].get('dirty_execution'):
        raise ValueError('Position execution source identity changed')
    if runtime_identity() != lock['runtime']:
        raise ValueError('Position runtime/scorer identity changed')
    if frozen['source'] != lock['source']:
        raise ValueError('Position frozen source identity changed')
    verify_frozen_files(frozen)
    key = str(run.relative_to(root))
    if cfg_to_plain(cfg) != lock['configurations'].get(key):
        raise ValueError('Position configuration does not match frozen run identity')
    dataset, arm, seed = str(cfg.data.dataset), str(cfg.train.semantic_control_arm), int(cfg.train.seed)
    expected_run = root/arm/('%s_%s_seed%d' % (arm, dataset, seed))
    if run != expected_run or arm not in POSITION_ARMS or dataset not in POSITION_DATASETS or seed not in SEEDS:
        raise ValueError('Position dataset/arm/seed directory mismatch')
    if not frozen['datasets'][dataset].get('technical_complete'):
        raise ValueError('Position dataset is not technically frozen')
    mode = 'sincos2d_kv' if arm == 'E2' else 'none'
    if cfg.model.semantic_fusion_local_position != mode or bool(cfg.data.use_semantic_maps) != (arm != 'E0'):
        raise ValueError('Position switch or semantic input differs from arm')
    selected_path = run/'best_snapshot_p1.json'
    if str(selected_path) not in frozen['files']:
        raise ValueError('Position selection is not bound by the freeze')
    selected = read(selected_path)
    if selected.get('protocol_id') != POSITION_PROTOCOL or selected.get('selection_metric_split') != 'validation':
        raise ValueError('Position selection identity mismatch')
    requested = Path(str(checkpoint)).resolve()
    if requested != Path(selected['best_snapshot']).resolve():
        raise ValueError('Requested checkpoint is not the frozen selected checkpoint')
    digest = validate_checkpoint_file(str(requested), require_checksum=True, require_metadata=True,
                                      expected_step=selected['best']['iter'])
    if digest != selected['best_snapshot_sha256']:
        raise ValueError('Position selected checkpoint SHA mismatch')
    manifest = read(root/('inputs_'+dataset+'.json'))
    verify_manifest(cfg, manifest)
    expected = run/'test_output/captions/controls_locked/sc_results.json'
    if Path(str(result_path)).resolve() != expected:
        raise ValueError('Position test output conflicts with the fixed frozen test path')
    identity_path, score, receipt = (run/n for n in ('test_prediction_identity.json', 'test_metrics.json', 'test_score_identity.json'))
    if expected.exists():
        if not identity_path.is_file():
            raise ValueError('Position predictions lack identity; refusing overwrite')
        ids = [r['sample_id'] for r in manifest['samples'] if r['split'] == 'test']
        observed = prediction_identity(expected, cfg.data.eval_anno_path, ids)
        observed['checkpoint_sha256'] = digest
        observed['scorer_identity'] = lock['runtime']
        if read(identity_path) != observed:
            raise ValueError('Position prediction identity changed')
        if score.exists():
            if not receipt.is_file() or read(receipt) != {'score_sha256': sha256_file(score), 'identity': observed}:
                raise ValueError('Position score lacks matching identity receipt')
            metrics(read(score)['metrics'])
            return 'complete'
        if receipt.exists():
            raise ValueError('Position receipt without score; refusing regeneration')
        return 'prediction_only'
    if identity_path.exists() or score.exists() or receipt.exists():
        raise ValueError('Position partial artifacts without predictions; refusing regeneration')
    return 'new'


def record_position_prediction(cfg, prediction, checkpoint):
    """Publish the same immutable inference identity for wrapper and direct calls."""
    if getattr(cfg.train, 'protocol_id', '') != POSITION_PROTOCOL:
        return
    run = (Path(str(cfg.exp_dir))/str(cfg.exp_name)).resolve()
    root = position_root(run)
    lock = read(root/'protocol.json')
    manifest = read(root/('inputs_'+str(cfg.data.dataset)+'.json'))
    ids = [row['sample_id'] for row in manifest['samples'] if row['split'] == 'test']
    identity = prediction_identity(prediction, cfg.data.eval_anno_path, ids)
    identity.update(checkpoint_sha256=sha256_file(checkpoint), scorer_identity=lock['runtime'])
    path = run/'test_prediction_identity.json'
    if path.exists():
        if read(path) != identity:
            raise ValueError('Position prediction identity already exists and differs')
    else:
        write(path, identity, immutable=True)
    return identity
