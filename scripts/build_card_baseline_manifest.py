import argparse
import datetime
import hashlib
import json
import math
import os


DATASET_ORDER = ('levir_cc', 'levir_mci', 'second_cc')
DEFAULT_EXPERIMENTS = {
    'levir_cc': 'card_levir_cc_baseline',
    'levir_mci': 'levir_mci_card_baseline',
    'second_cc': 'second_cc_card_rgb_baseline',
}
REQUIRED_METRICS = (
    'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
    'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE',
)


def load_json(path):
    with open(path, encoding='utf-8-sig') as handle:
        return json.load(handle)


def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def is_within(path, parent):
    try:
        return os.path.commonpath([canonical(path), canonical(parent)]) == canonical(parent)
    except ValueError:
        return False


def nested(payload, dotted, default=None):
    value = payload
    for part in dotted.split('.'):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def require_false(payload, dotted):
    value = nested(payload, dotted, False)
    if value not in (False, 0, None, ''):
        raise ValueError('%s must be false for original CARD, got %r' % (dotted, value))


def require_zero(payload, dotted):
    value = nested(payload, dotted, 0.0)
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        raise ValueError('%s must be zero for original CARD, got %r' % (dotted, value))
    if not math.isfinite(numeric) or abs(numeric) > 1e-12:
        raise ValueError('%s must be zero for original CARD, got %r' % (dotted, value))


def require_numeric(payload, dotted, expected):
    value = nested(payload, dotted)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be %s, got %r' % (dotted, expected, value))
    if not math.isfinite(numeric) or abs(numeric - expected) > 1e-12:
        raise ValueError('%s must be %s, got %r' % (dotted, expected, value))


def audit_resolved_config(path, dataset):
    if not os.path.isfile(path):
        raise FileNotFoundError('Missing resolved baseline config: %s' % path)
    cfg = load_json(path)
    if nested(cfg, 'model.type') != 'card':
        raise ValueError('model.type must be card in %s' % path)
    if nested(cfg, 'data.dataset') != dataset:
        raise ValueError('data.dataset must be %s in %s' % (dataset, path))
    if str(nested(cfg, 'model.semantic_input_mode', 'none')).lower() != 'none':
        raise ValueError('model.semantic_input_mode must be none in %s' % path)
    if str(nested(cfg, 'train.optim.type', '')).lower() != 'adam':
        raise ValueError('train.optim.type must be adam in %s' % path)
    init_checkpoint = nested(cfg, 'train.init_checkpoint', '')
    if init_checkpoint not in (None, '', 'None'):
        raise ValueError('train.init_checkpoint must be empty in %s' % path)
    start_from = nested(cfg, 'train.start_from', None)
    if start_from not in (None, '', 'None'):
        raise ValueError('train.start_from must be empty in %s' % path)

    for dotted in (
        'model.enable_aux_mask', 'model.use_aux_mask',
        'data.use_change_mask', 'data.use_semantic_maps',
        'data.allow_missing_pseudo_mask',
        'train.use_aux_mask', 'train.use_semantic_aux',
        'train.use_aux_semantic', 'train.use_semantic_detach',
        'train.use_semantic_partial_detach', 'train.use_partial_detach',
        'train.use_semantic_cross_attention', 'train.use_semantic_hard_gate',
        'train.use_feature_reweight', 'train.detach_reweight_mask',
        'train.use_content_word_weight', 'train.use_content_word_weighted_ce',
        'train.normalize_content_word_weights', 'train.use_relation_aux',
        'train.use_weak_mask_prior', 'train.use_aux_warmup',
        'train.use_mask_conf_filter', 'train.use_mask_warmup',
        'train.use_semantic_warmup', 'train.semantic_late_start',
        'train.paper_selection_mode', 'train.finetune_decoder_only',
    ):
        require_false(cfg, dotted)
    for dotted in (
        'train.finetune_steps', 'train.lambda_mask', 'train.lambda_semantic',
        'train.semantic_detach_ratio', 'train.reweight_alpha',
        'train.aux_warmup_start_ratio', 'train.aux_warmup_end_ratio',
        'model.semantic_fusion_gamma_init', 'model.semantic_fusion_gamma_max',
    ):
        require_zero(cfg, dotted)
    for dotted, expected in (
        ('train.max_iter', 10000), ('train.total_steps', 10000),
        ('train.snapshot_interval', 1000), ('train.save_interval', 1000),
        ('train.eval_interval', 1000), ('train.optim.lr', 0.0002),
        ('train.optim.weight_decay', 0.0), ('train.optim.step_size', 17),
        ('train.optim.gamma', 0.1),
        ('train.seed', 1111), ('data.train.batch_size', 32),
    ):
        require_numeric(cfg, dotted, expected)
    return cfg


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def complete_metrics(payload):
    metrics = payload.get('selected_val_metrics')
    if not isinstance(metrics, dict):
        raise ValueError('selection is missing selected_val_metrics')
    result = {}
    for metric in REQUIRED_METRICS:
        value = metrics.get(metric)
        if isinstance(value, bool):
            raise ValueError('invalid validation metric %s=%r' % (metric, value))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError('missing/invalid validation metric %s' % metric)
        if not math.isfinite(value):
            raise ValueError('non-finite validation metric %s' % metric)
        result[metric] = value
    return result


def build_entry(exp_root, dataset, exp_name):
    exp_dir = canonical(os.path.join(exp_root, exp_name))
    selection_path = canonical(os.path.join(exp_dir, 'baseline_best_checkpoint.json'))
    resolved_path = canonical(os.path.join(exp_dir, 'resolved_config.json'))
    cfg = audit_resolved_config(resolved_path, dataset)
    resolved_yaml = canonical(os.path.join(exp_dir, 'resolved_config.yaml'))
    if not os.path.isfile(resolved_yaml):
        raise FileNotFoundError('Missing resolved baseline YAML: %s' % resolved_yaml)
    if not os.path.isfile(selection_path):
        raise FileNotFoundError('Missing baseline selection: %s' % selection_path)
    selection = load_json(selection_path)
    if selection.get('status') != 'done':
        raise ValueError('baseline selection status is not done: %s' % selection_path)
    if selection.get('selection_strategy') != 'validation_best_cider':
        raise ValueError('baseline must use validation_best_cider: %s' % selection_path)
    if selection.get('selection_uses_test_metrics') is not False:
        raise ValueError('baseline selection is not explicitly validation-only: %s' % selection_path)
    if selection.get('selection_metric_split') != 'validation':
        raise ValueError('baseline selection split is not validation: %s' % selection_path)
    source_dir = canonical(selection.get('selected_source_exp_dir') or exp_dir)
    if source_dir != exp_dir:
        raise ValueError('baseline selection came from a different experiment: %s' % source_dir)
    metrics_file = canonical(selection.get('metrics_file') or '')
    if os.path.basename(metrics_file) not in ('val_metrics.csv', 'eval_snapshots.csv'):
        raise ValueError('baseline selection did not use a validation metrics file: %s' % metrics_file)
    if not is_within(metrics_file, exp_dir) or not os.path.isfile(metrics_file):
        raise ValueError('validation metrics file is missing/outside baseline experiment: %s' % metrics_file)
    checkpoint = canonical(
        selection.get('selected_checkpoint_path')
        or selection.get('selected_checkpoint')
        or ''
    )
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError('Locked baseline checkpoint is missing: %s' % checkpoint)
    if not is_within(checkpoint, exp_dir):
        raise ValueError('Locked baseline checkpoint is outside its experiment: %s' % checkpoint)
    metrics = complete_metrics(selection)
    test_exp_dir = canonical(os.path.join(exp_root, 'card_baseline_locked_tests', dataset))
    return {
        'status': 'done',
        'exp_name': exp_name,
        'exp_dir': exp_dir,
        'selection_json': selection_path,
        'selection_strategy': 'validation_best_cider',
        'selection_uses_test_metrics': False,
        'selected_checkpoint': checkpoint,
        'selected_checkpoint_sha256': sha256_file(checkpoint),
        'selected_val_metrics': metrics,
        'resolved_config': resolved_path,
        'resolved_config_sha256': sha256_file(resolved_path),
        'resolved_config_yaml': resolved_yaml,
        'resolved_config_yaml_sha256': sha256_file(resolved_yaml),
        'data_root': canonical(nested(cfg, 'data.data_root')),
        'feature_root': canonical(nested(cfg, 'data.default_feature_dir')),
        'eval_anno_path': canonical(nested(cfg, 'data.eval_anno_path')),
        'test_exp_dir': test_exp_dir,
        'test_result': canonical(os.path.join(test_exp_dir, 'test_card_baseline_locked_result.json')),
    }


def verify_manifest(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('Missing locked baseline manifest: %s' % path)
    manifest = load_json(path)
    if manifest.get('status') != 'validation_locked':
        raise ValueError('baseline manifest status is not validation_locked')
    if manifest.get('selection_uses_test_metrics') is not False:
        raise ValueError('baseline manifest is not explicitly validation-only')
    datasets = manifest.get('datasets')
    if not isinstance(datasets, dict) or set(datasets) != set(DATASET_ORDER):
        raise ValueError('baseline manifest must contain exactly three datasets')
    for dataset in DATASET_ORDER:
        entry = datasets[dataset]
        if entry.get('status') != 'done':
            raise ValueError('%s baseline lock status is not done' % dataset)
        if entry.get('selection_strategy') != 'validation_best_cider':
            raise ValueError('%s baseline selection strategy changed' % dataset)
        if entry.get('selection_uses_test_metrics') is not False:
            raise ValueError('%s baseline selection is not validation-only' % dataset)
        checkpoint = canonical(entry.get('selected_checkpoint') or '')
        resolved = canonical(entry.get('resolved_config') or '')
        resolved_yaml = canonical(entry.get('resolved_config_yaml') or '')
        source_exp_dir = canonical(entry.get('exp_dir') or '')
        if os.path.dirname(resolved) != source_exp_dir:
            raise ValueError('%s resolved config is outside the source experiment' % dataset)
        if entry.get('exp_name') != os.path.basename(source_exp_dir):
            raise ValueError('%s source experiment name/path mismatch' % dataset)
        if not is_within(checkpoint, source_exp_dir):
            raise ValueError('%s locked checkpoint is outside the source experiment' % dataset)
        test_exp_dir = canonical(entry.get('test_exp_dir') or '')
        expected_test_dir = canonical(os.path.join(os.path.dirname(source_exp_dir), 'card_baseline_locked_tests', dataset))
        expected_result = canonical(os.path.join(expected_test_dir, 'test_card_baseline_locked_result.json'))
        if test_exp_dir != expected_test_dir or canonical(entry.get('test_result') or '') != expected_result:
            raise ValueError('%s locked test target is not the isolated baseline-test directory' % dataset)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('%s locked checkpoint is missing: %s' % (dataset, checkpoint))
        if sha256_file(checkpoint) != entry.get('selected_checkpoint_sha256'):
            raise ValueError('%s locked checkpoint hash changed' % dataset)
        if not os.path.isfile(resolved):
            raise FileNotFoundError('%s resolved config is missing: %s' % (dataset, resolved))
        if sha256_file(resolved) != entry.get('resolved_config_sha256'):
            raise ValueError('%s resolved config hash changed' % dataset)
        if not os.path.isfile(resolved_yaml):
            raise FileNotFoundError('%s resolved YAML is missing: %s' % (dataset, resolved_yaml))
        if sha256_file(resolved_yaml) != entry.get('resolved_config_yaml_sha256'):
            raise ValueError('%s resolved YAML hash changed' % dataset)
        cfg = audit_resolved_config(resolved, dataset)
        for field, dotted in (
            ('data_root', 'data.data_root'),
            ('feature_root', 'data.default_feature_dir'),
            ('eval_anno_path', 'data.eval_anno_path'),
        ):
            expected = canonical(nested(cfg, dotted) or '')
            if canonical(entry.get(field) or '') != expected:
                raise ValueError('%s locked %s does not match resolved config' % (dataset, field))
        complete_metrics({'selected_val_metrics': entry.get('selected_val_metrics')})
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description='Audit and lock original CARD baselines using validation only.')
    parser.add_argument('--exp_root', default='./experiments')
    parser.add_argument('--output', default=None)
    parser.add_argument('--levir_cc_exp', default=DEFAULT_EXPERIMENTS['levir_cc'])
    parser.add_argument('--levir_mci_exp', default=DEFAULT_EXPERIMENTS['levir_mci'])
    parser.add_argument('--second_cc_exp', default=DEFAULT_EXPERIMENTS['second_cc'])
    parser.add_argument('--only_dataset', choices=DATASET_ORDER, default=None)
    parser.add_argument('--validate_only', action='store_true')
    parser.add_argument('--verify_manifest', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_manifest:
        manifest = verify_manifest(args.verify_manifest)
        print(json.dumps({'status': 'verified', 'datasets': list(manifest['datasets'])}, indent=2))
        return
    names = {
        'levir_cc': args.levir_cc_exp,
        'levir_mci': args.levir_mci_exp,
        'second_cc': args.second_cc_exp,
    }
    datasets = (args.only_dataset,) if args.only_dataset else DATASET_ORDER
    entries = {dataset: build_entry(args.exp_root, dataset, names[dataset]) for dataset in datasets}
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'datasets': entries}, indent=2, ensure_ascii=False))
        return
    if args.only_dataset:
        raise SystemExit('--only_dataset requires --validate_only; partial manifests are forbidden')
    output = canonical(args.output or os.path.join(args.exp_root, 'card_baseline_locked_manifest.json'))
    manifest = {
        'status': 'validation_locked',
        'selection_uses_test_metrics': False,
        'selection_policy': 'highest validation CIDEr with all 8 validation metrics present',
        'locked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'datasets': entries,
    }
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    temporary = output + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
