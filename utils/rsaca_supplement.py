"""Read-only historical import and shared admission for the RSACA supplement."""
import copy
import math
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from scripts.select_best_snapshot_p1 import validation_rows
from utils.attr_dict import AttrDict
from utils.checkpoint_integrity import sha256_file, validate_checkpoint_file
from utils.experiment_tracking import cfg_to_plain, stable_hash
from utils.semantic_control_audit import (
    input_manifest, matrix, metrics, prediction_identity, read, source_identity, write,
)
from utils.semantic_controls import ABLATIONS, ARMS, PROTOCOL, REFERENCE, SUPPLEMENT_PROTOCOL

REVISION = '2026-11-25'
HISTORICAL = {
    'O': '443d9ca11aed443f8a66d8cf72088c349af2c673',
    'S': '534f22fcd69294890152e1ae69a56bdf4838405a',
}
COUNTS = {'levir_cc': 1929, 'levir_mci': 1929, 'second_cc': 1227}
SOURCES = {'levir_cc': 'prediction', 'levir_mci': 'annotation', 'second_cc': 'annotation'}
GRID = set(range(1000, 10001, 1000))
HEX = re.compile(r'[0-9a-f]{64}')


def now():
    return datetime.now(timezone.utc).isoformat()


def node(value):
    if isinstance(value, dict):
        return AttrDict({k: node(v) for k, v in value.items()})
    return value


def arms_for(series):
    return {
        'O': ARMS, 'S': ARMS, 'new-ablation': ('rsaca',) + ABLATIONS,
        'new-full': ARMS + ABLATIONS,
    }[series]


def entries(root, series):
    return list(matrix(root / 'runs', arms=arms_for(series)))


def runtime():
    import numpy
    import torch
    return {
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'numpy': numpy.__version__, 'cuda': torch.version.cuda,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
    }


def scorer_identity():
    from utils import eval_utils_spot
    import pycocoevalcap
    import pycocotools
    java = shutil.which('java')
    if not java:
        raise ValueError('Java is required for caption scoring')
    files = {}
    for package in (pycocoevalcap, pycocotools):
        for root in map(Path, package.__path__):
            files[str(root.resolve())] = {
                p.relative_to(root).as_posix(): sha256_file(p)
                for p in sorted(root.rglob('*')) if p.is_file()
                and not {'__pycache__', 'cache', 'tmp'}.intersection(p.parts) and p.suffix != '.pyc'}
    return {
        'packages': files, 'entry_sha256': sha256_file(eval_utils_spot.__file__), 'java': str(Path(java).resolve()),
        'java_sha256': sha256_file(java),
        'java_version': subprocess.check_output(
            [java, '-version'], stderr=subprocess.STDOUT, text=True).strip(),
    }


def validate_semantic_sample(ds, split, filename, paths):
    """Check the real reader output before a sample can reach optional collate."""
    import numpy as np
    from datasets.rcc_dataset_transformer_levir import _load_image_or_array
    phases = (ds.semantic_diff_phase,) if ds.semantic_diff_only else (
        ds.semantic_before_phase, ds.semantic_after_phase)
    root = Path(ds.semantic_diff_root if ds.semantic_diff_only else ds.semantic_map_root)
    if len(paths) == 2 and Path(paths[0]).samefile(paths[1]):
        raise ValueError('Temporal maps resolve to the same file: %s/%s' % (split, filename))
    sizes = []
    for phase, path in zip(phases, paths):
        allowed = ((root/split/phase).resolve(), (root/phase/split).resolve())
        if Path(path).parent.resolve() not in allowed:
            raise ValueError('Wrong split, source or temporal phase: %s/%s %s' % (split, filename, path))
        raw = _load_image_or_array(path)
        if (raw.ndim not in (2, 3) or min(raw.shape) < 1
                or (raw.ndim == 3 and raw.shape[-1] not in (1, 3, 4))
                or raw.dtype.kind not in 'buif'
                or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()):
            raise ValueError('Invalid map dimensions or encoding: %s/%s %s' % (split, filename, path))
        sizes.append(raw.shape[:2])
    if len(set(sizes)) != 1:
        raise ValueError('Missing or misaligned temporal maps: %s/%s' % (split, filename))
    before, after, diff, _ = ds._load_semantic_pair(split, filename)
    if not ds.semantic_diff_only and (before is None or after is None or before.shape != after.shape):
        raise ValueError('Missing or misaligned temporal maps: %s/%s' % (split, filename))
    if diff is None:
        raise ValueError('Missing change map: %s/%s' % (split, filename))
    for array in (before, after, diff):
        if array is not None:
            invalid = (array != ds.cfg.train.semantic_ignore_index) & (
                (array < 0) | (array > ds.cfg.data.num_semantic_classes))
            if array.ndim != 2 or min(array.shape) < 1 or invalid.any():
                raise ValueError('Invalid decoded map: %s/%s' % (split, filename))


def mapped_path(path, mappings):
    original = PurePosixPath(path)
    for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
        try:
            relative = original.relative_to(PurePosixPath(old))
        except ValueError:
            continue
        if '..' in relative.parts:
            raise ValueError('Path mapping may not traverse a parent: ' + path)
        return str(Path(new).joinpath(*relative.parts).resolve())
    raise ValueError('No explicit historical path mapping for ' + path)


def eval_config(request, root, dataset, seed, arm, mappings=None):
    cfg = node(copy.deepcopy(request))
    if mappings is not None:
        for key, value in cfg.data.items():
            if isinstance(value, str) and value.startswith('/'):
                cfg.data[key] = mapped_path(value, mappings)
    cfg.exp_dir = str(root / 'evaluations' / arm)
    cfg.exp_name = '%s_%s_seed%d' % (arm, dataset, seed)
    cfg.train.protocol_id = SUPPLEMENT_PROTOCOL
    # Explicit defaults added after S/O. Both switches preserve their forward.
    cfg.model.setdefault('semantic_fusion_dense_local_access', False)
    cfg.model.setdefault('semantic_fusion_fixed_gate_value', 1.0)
    return cfg


def get_inputs(cfg, old=None, mappings=None, cache=None, decode=True):
    actual = input_manifest(cfg, SOURCES[cfg.data.dataset], hash_cache=cache, validate_maps=decode)
    if actual['missing']:
        raise ValueError('Missing input files: ' + str(actual['missing'][:12]))
    if len(split_ids(actual, 'test')) != COUNTS[cfg.data.dataset]:
        raise ValueError('Official test split count mismatch for ' + cfg.data.dataset)
    if old is not None:
        expected = copy.deepcopy(old)
        for artifact in expected['artifacts'].values():
            artifact['path'] = mapped_path(artifact['path'], mappings)
        for row in expected['samples']:
            if not cfg.data.use_semantic_maps:
                row['semantic'] = []
            for entry in row['visual'] + row['semantic']:
                entry['path'] = mapped_path(entry['path'], mappings)
        if (actual['artifacts'], actual['samples']) != (expected['artifacts'], expected['samples']):
            raise ValueError('Current inputs disagree with the mapped historical manifest')
    return actual


def split_ids(manifest, split):
    return [row['sample_id'] for row in manifest['samples'] if row['split'] == split]


def run_evidence(run, request, series, manifest, historical_prediction=None):
    """Recalculate selection without recreating or writing historical artifacts."""
    dataset, seed = request['data']['dataset'], request['train']['seed']
    proto = PROTOCOL if series in HISTORICAL else SUPPLEMENT_PROTOCOL
    names = ['request_config.json', 'resolved_config.json', 'actual_model_config.json',
             'run_summary.json', 'best_snapshot_p1.json', 'val_metrics.csv',
             'validation_prediction_identity.json', 'initial_parameter_summary.json']
    if series in HISTORICAL:
        names += ['controls_provenance.json', 'environment.txt', 'command.txt']
    files = {str(run/name): sha256_file(run/name) for name in names}
    if read(run/'request_config.json') != request or read(run/'resolved_config.json') != request:
        raise ValueError('Request/resolved configuration differs from training lock')
    summary = read(run/'run_summary.json')
    if (summary.get('status'), summary.get('final_global_step'), summary.get('dataset'),
            summary.get('seed'), summary.get('experiment_name')) != (
            'completed', 10000, dataset, seed, request['exp_name']):
        raise ValueError('Training completion/key mismatch')
    record = read(run/'best_snapshot_p1.json')
    if (record['protocol_id'], record['selection_metric_split'], record['selection_strategy'],
            record['selection']['reference'], record['selection']['tie_break']) != (
            proto, 'validation', 'five_metric_equal_weight_log', REFERENCE, 'earlier_step'):
        raise ValueError('Selection protocol mismatch')
    rows = validation_rows(run/'val_metrics.csv', REFERENCE)
    candidates = record['candidates']
    by_step = {c['iter']: c for c in candidates}
    if len(candidates) != len(GRID) or set(by_step) != GRID:
        raise ValueError('Recorded selection grid differs')
    for row in rows:
        step = row['iter']
        observed = by_step[step]
        if (any(observed[k] != row[k] for k in ('row', 'iter', 'metrics'))
                or not math.isclose(observed['score'], row['score'], rel_tol=0, abs_tol=5e-15)):
            raise ValueError('Recorded validation metrics or selection score differs')
        snapshot = PurePosixPath(observed['snapshot_path'].replace('\\', '/'))
        expected_name = '%s_checkpoint_%d.pt' % (request['exp_name'], step)
        registered_path = PurePosixPath(row['snapshot_path'].replace('\\', '/'))
        if (snapshot.name != expected_name or snapshot.parent.name != 'snapshots'
                or snapshot != registered_path
                or snapshot.parent.parent.name != request['exp_name']):
            raise ValueError('Candidate snapshot filename/key mismatch')
        sidecar = run/'snapshots'/(snapshot.name + '.metadata.json')
        checksum = run/'snapshots'/(snapshot.name + '.sha256')
        digest = observed['snapshot_sha256']
        size = observed['snapshot_size_bytes']
        if not HEX.fullmatch(digest) or size < 1 or read(sidecar) != dict(
                filename=snapshot.name, sha256=digest, size_bytes=size, step=step):
            raise ValueError('Snapshot sidecar differs from selection')
        if checksum.read_text(encoding='utf-8').split() != [digest, snapshot.name]:
            raise ValueError('Snapshot checksum sidecar differs')
        files.update({str(p): sha256_file(p) for p in (sidecar, checksum)})
    best = by_step[min(rows, key=lambda value: (-value['score'], value['iter']))['iter']]
    if (record['best'] != best or record['best_snapshot'] != best['snapshot_path']
            or record['best_snapshot_sha256'] != best['snapshot_sha256']
            or record['best_snapshot_size_bytes'] != best['snapshot_size_bytes']):
        raise ValueError('Selected checkpoint identity differs from validation grid')
    identity = read(run/'validation_prediction_identity.json')
    ids = split_ids(manifest, 'val')
    reference = manifest['artifacts']['eval_anno_path']
    if set(identity) != set(map(str, GRID)):
        raise ValueError('Validation prediction identity grid incomplete')
    for item in identity.values():
        if (item['count'], item['sample_ids_sha256'], item['reference_sha256']) != (
                len(ids), stable_hash(sorted(ids)), reference['sha256']):
            raise ValueError('Validation IDs or reference differ from training inputs')
    if series not in HISTORICAL:
        for entry in identity.values():
            path = Path(entry['prediction']).resolve()
            if run.resolve() not in path.parents:
                raise ValueError('Validation prediction is outside the training run')
            actual = prediction_identity(path, reference['path'], ids)
            if any(entry.get(k) != v for k, v in actual.items()):
                raise ValueError('Validation prediction content differs from recorded identity')
            files[str(path)] = actual['prediction_sha256']
    chosen = identity[str(best['iter'])]
    prediction = historical_prediction or Path(chosen['prediction'])
    if sha256_file(prediction) != chosen['prediction_sha256']:
        raise ValueError('Selected validation prediction differs')
    observed_ids = [str(row['image_id']) for row in read(prediction)]
    if len(observed_ids) != len(ids) or set(observed_ids) != set(ids):
        raise ValueError('Selected validation predictions do not cover the split')
    files[str(prediction)] = chosen['prediction_sha256']
    initial = read(run/'initial_parameter_summary.json')
    if initial['protocol_id'] != proto or initial['seed'] != seed:
        raise ValueError('Initialization protocol/key mismatch')
    return {
        'step': best['iter'], 'checkpoint_sha256': best['snapshot_sha256'],
        'checkpoint_size': best['snapshot_size_bytes'], 'original_checkpoint': best['snapshot_path'],
        'selection': str(run/'best_snapshot_p1.json'), 'selection_sha256': files[str(run/'best_snapshot_p1.json')],
        'validation_metrics': metrics(best['metrics']), 'validation_prediction': str(prediction),
        'training_seconds': summary.get('total_training_seconds'), 'training_config': read(run/'actual_model_config.json'),
        'files': files, 'initial': initial,
    }


def training_run(cfg):
    return (Path(cfg.exp_dir) / cfg.exp_name).resolve()


def validate_training(project, cfg, output):
    run = training_run(cfg)
    root = run.parents[2]
    if (root/'frozen.json').exists():
        raise ValueError('Supplement training matrix already frozen')
    lock = read(root/'training.json')
    key = run.relative_to(root).as_posix()
    if (lock.get('protocol_id') != SUPPLEMENT_PROTOCOL
            or lock['configurations'].get(key) != cfg_to_plain(cfg)
            or Path(output or run).resolve() != run
            or lock['execution'] != execution_identity(project)):
        raise ValueError('Training command differs from its recorded matrix')
    path = lock['inputs'][cfg.data.dataset]['fusion' if cfg.data.use_semantic_maps else 'card']
    if sha256_file(path) != lock['input_digests'][path] or get_inputs(cfg) != read(path):
        raise ValueError('Training input identity changed')
    write(run/'supplement_training_provenance.json', {
        'key': key, 'training_sha256': sha256_file(root/'training.json'),
        'manifest_sha256': lock['input_digests'][path],
    }, immutable=True)


def execution_identity(project):
    source = source_identity(project)
    if source['dirty_execution']:
        raise ValueError('Execution source differs from the checked-in code')
    return {'source': source, 'runtime': runtime()}


def check_checkpoint(item, cfg, original_request):
    from models.CARD import CARD
    from models.transformer_decoder import DynamicSpeaker
    from utils.checkpointing import load_checkpoint_file, split_model_states

    path = item['checkpoint']
    if validate_checkpoint_file(path, require_metadata=True, expected_step=item['step']) != item['checkpoint_sha256']:
        raise ValueError('Weight digest differs from validated selection')
    if Path(path).stat().st_size != item['checkpoint_size']:
        raise ValueError('Weight size differs from validated selection')
    checkpoint = load_checkpoint_file(path)
    if checkpoint['global_step'] != item['step'] or cfg_to_plain(checkpoint['config']) != item['training_config']:
        raise ValueError('Checkpoint internal step or configuration differs from training')
    actual = node(copy.deepcopy(cfg_to_plain(cfg)))
    adapted = adapted_configuration(actual)
    expected_training = copy.deepcopy(original_request)
    expected_training['model']['transformer_decoder'] = adapted['model']['transformer_decoder']
    expected_training['model']['num_semantic_classes'] = adapted['model']['num_semantic_classes']
    expected_training['train']['content_word_token_ids'] = []
    for section in ('data', 'train', 'model'):
        if item['training_config'][section] != expected_training[section]:
            raise ValueError('Actual training configuration differs from selected run: '+section)
    initial = item['initial']
    for model, state, name in zip((CARD(actual), DynamicSpeaker(actual)),
                                  split_model_states(checkpoint), ('change_detector', 'speaker')):
        expected = model.state_dict()
        if (set(state) != set(expected) or any(
                state[key].dtype != target.dtype or state[key].shape != target.shape
                for key, target in expected.items())):
            raise ValueError('Strict checkpoint tensor mismatch: '+name)
        params = dict(model.named_parameters())
        if set(initial[name]) != set(params) or any(
                initial[name][key]['shape'] != list(value.shape)
                or initial[name][key]['dtype'] != str(value.dtype)
                or not HEX.fullmatch(initial[name][key]['sha256'])
                for key, value in params.items()):
            raise ValueError('Initialization parameter schema mismatch: '+name)
        model.load_state_dict(state, strict=True)


def adapted_configuration(cfg):
    import h5py
    cfg.model.transformer_decoder.vocab_size = len(read(cfg.data.vocab_json))
    with h5py.File(cfg.data.h5_label_file, 'r') as handle:
        cfg.model.transformer_decoder.seq_length = handle['labels'].shape[1]
    if cfg.data.use_semantic_maps:
        cfg.model.num_semantic_classes = cfg.data.num_semantic_classes+1
    cfg.train.content_word_token_ids = []
    return cfg_to_plain(cfg)


def freeze_payload(root, payload):
    if set(payload['runs']) != {
        run.relative_to(root).as_posix()
        for _, _, _, run in entries(root, payload['series'])
    } or payload['errors'] or not all(x['strict_load'] for x in payload['runs'].values()):
        raise ValueError('Test blocked: every matrix key must pass the technical audit')
    return dict(payload, protocol_id=SUPPLEMENT_PROTOCOL, admission_rule_revision=REVISION, frozen_at=now())


def read_frozen(root):
    frozen = read(root/'frozen.json')
    if (frozen['protocol_id'] != SUPPLEMENT_PROTOCOL
            or frozen['series'] != root.name
            or frozen['admission_rule_revision'] != REVISION
            or set(frozen['runs']) != {p.relative_to(root).as_posix() for _, _, _, p in entries(root, frozen['series'])}
            or frozen['errors'] or not all(x['strict_load'] for x in frozen['runs'].values())):
        raise ValueError('Frozen matrix identity or coverage mismatch')
    if read(root/'frozen_identity.json')['sha256'] != sha256_file(root/'frozen.json'):
        raise ValueError('Frozen manifest changed')
    if 'replay' in frozen:
        replay = frozen['replay']
        if sha256_file(replay['path']) != replay['sha256']:
            raise ValueError('Frozen historical validation replay changed')
    elif frozen['series'] in HISTORICAL:
        raise ValueError('Frozen historical validation replay is missing')
    return frozen


def admission_details(project, cfg, checkpoint, result_path):
    run = training_run(cfg)
    root = run.parents[2]
    frozen = read_frozen(root)
    if frozen['execution'] != execution_identity(project) or frozen['scorer'] != scorer_identity():
        raise ValueError('Evaluation source, runtime, or scorer identity changed')
    if frozen['series'] in HISTORICAL:
        primary = read(root.parent/'primary.json')
        primary_root = root.parent/primary['series']
        if (primary['series'] not in HISTORICAL
                or primary['sha256'] != sha256_file(primary_root/'frozen.json')):
            raise ValueError('Historical primary series must be registered before decoding')
        if read_frozen(primary_root)['series'] != primary['series']:
            raise ValueError('Historical primary series identity differs')
    key = 'runs/%s/%s' % (run.parent.name, run.name)
    item = frozen['runs'][key]
    if (cfg_to_plain(cfg) != item['config'] or Path(checkpoint).resolve() != Path(item['checkpoint'])
            or Path(result_path).resolve() != run/'captions'/'test.json'):
        raise ValueError('Configuration, checkpoint or output differs from frozen supplemental run')
    from utils.semantic_control_audit import verify_frozen_files
    verify_frozen_files({'files': item['files']})
    manifest_path = Path(item['manifest'])
    if sha256_file(manifest_path) != item['manifest_sha256']:
        raise ValueError('Input manifest changed')
    manifest = read(manifest_path)
    if get_inputs(cfg, decode=False) != manifest:
        raise ValueError('Frozen input content or resolution changed')
    digest = validate_checkpoint_file(checkpoint, require_metadata=True, expected_step=item['step'])
    if digest != item['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    return frozen, item, manifest


def admit_replay(project, cfg, checkpoint, result_path):
    output = Path(result_path).resolve()
    attempt = output.parent.parent
    plan = read(attempt/'replay_request.json')
    if (output != attempt/'captions'/'val.json'
            or Path(checkpoint).resolve() != Path(plan['checkpoint'])
            or cfg_to_plain(cfg) != plan['config']
            or plan['execution'] != execution_identity(project)
            or sha256_file(checkpoint) != plan['checkpoint_sha256']
            or sha256_file(plan['manifest']) != plan['manifest_sha256']
            or get_inputs(cfg, decode=False) != read(plan['manifest'])):
        raise ValueError('Validation replay differs from approved input or run identity')


def observed_prediction(run, cfg, item, manifest):
    path = run/'captions'/'test.json'
    ids = set(split_ids(manifest, 'test'))
    reference = read(cfg.data.eval_anno_path)
    annotations = [a for a in reference['annotations'] if str(a['image_id']) in ids]
    annotation_ids = [a['id'] for a in annotations]
    refs_per_id = {key: 0 for key in ids}
    for annotation in annotations:
        refs_per_id[str(annotation['image_id'])] += 1
    if len(set(annotation_ids)) != len(annotation_ids) or not all(refs_per_id.values()):
        raise ValueError('Duplicate reference IDs or missing captions in test split')
    return dict(prediction_identity(path, cfg.data.eval_anno_path, ids),
                reference_count=len(annotations), reference_counts_sha256=stable_hash(refs_per_id),
                checkpoint_sha256=item['checkpoint_sha256'], step=item['step'],
                selection_sha256=item['selection_sha256'], split='test',
                frozen_sha256=sha256_file(run.parents[2]/'frozen.json'))


def admit(project, cfg, checkpoint, result_path):
    frozen, item, manifest = admission_details(project, cfg, checkpoint, result_path)
    run = training_run(cfg)
    prediction, receipt = Path(result_path), run/'prediction_identity.json'
    score = run/'test_score.json'
    score_receipt = run/'test_score_identity.json'
    if not any(p.exists() for p in (prediction, receipt, score, score_receipt)):
        return 'new'
    if not prediction.is_file() or not receipt.is_file():
        raise ValueError('Incomplete output. Preserve the evaluation directory before retrying')
    identity = observed_prediction(run, cfg, item, manifest)
    if read(receipt)['identity'] != identity:
        raise ValueError('Prediction, reference, split or selected weight changed')
    if score.exists() != score_receipt.exists():
        raise ValueError('Incomplete score output. Preserve it before retrying')
    if score.is_file():
        verified_score(run, identity, frozen['scorer'])
        return 'complete'
    return 'prediction_only'


def verified_score(run, identity, scorer):
    path = run/'test_score.json'
    if read(run/'test_score_identity.json') != {'score_sha256': sha256_file(path)}:
        raise ValueError('Score file hash changed')
    record = read(path)
    metrics(record['metrics'])
    if record['identity'] != identity or record['scorer'] != scorer or record['split'] != 'test':
        raise ValueError('Score identity differs from frozen prediction/scorer')
    return record


def publish_prediction(cfg, result_path, predictions):
    run = training_run(cfg)
    item = read_frozen(run.parents[2])['runs']['runs/%s/%s' % (run.parent.name, run.name)]
    manifest = read(item['manifest'])
    if set(predictions) != set(split_ids(manifest, 'test')):
        raise ValueError('Decoded IDs do not cover the frozen test split')
    write(result_path, [{'image_id': key, 'caption': text} for key, text in predictions.items()],
          immutable=True)
    write(run/'prediction_identity.json', {
        'identity': observed_prediction(run, cfg, item, manifest), 'decoded_at': now(),
    }, immutable=True)
