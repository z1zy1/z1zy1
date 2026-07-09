import argparse
import csv
import glob
import json
import os
import time


CAPTION_COLUMNS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
SUMMARY_COLUMNS = [
    'dataset',
    'exp_name',
    'method_group',
    'status',
    'config_hash',
    'checkpoint',
    'selection_strategy',
    'Bleu_1',
    'Bleu_2',
    'Bleu_3',
    'Bleu_4',
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'SPICE',
    'Mask_Precision',
    'Mask_Recall',
    'Mask_F1',
    'Mask_IoU',
    'Mask_mIoU',
    'IoU_road',
    'IoU_building',
    'Semantic_mIoU',
    'Semantic_IoU',
    'Semantic_F1',
    'Change_Bleu_4',
    'Change_CIDEr',
    'Change_SPICE',
    'NoChange_Bleu_4',
    'NoChange_ROUGE_L',
    'NoChange_SPICE',
    'notes',
    'timestamp',
]

GROUP_METRIC_COLUMNS = {
    'change': {
        'Bleu_4': 'Change_Bleu_4',
        'CIDEr': 'Change_CIDEr',
        'SPICE': 'Change_SPICE',
    },
    'nochange': {
        'Bleu_4': 'NoChange_Bleu_4',
        'ROUGE_L': 'NoChange_ROUGE_L',
        'SPICE': 'NoChange_SPICE',
    },
}

GROUP_RESULT_BASENAMES = {
    'test_change_result.json',
    'test_nochange_result.json',
    'val_change_result.json',
    'val_nochange_result.json',
}

ALIASES = {
    'BLEU_1': 'Bleu_1',
    'BLEU_2': 'Bleu_2',
    'BLEU_3': 'Bleu_3',
    'BLEU_4': 'Bleu_4',
    'bleu_1': 'Bleu_1',
    'bleu_2': 'Bleu_2',
    'bleu_3': 'Bleu_3',
    'bleu_4': 'Bleu_4',
    'B1': 'Bleu_1',
    'B2': 'Bleu_2',
    'B3': 'Bleu_3',
    'B4': 'Bleu_4',
    'ROUGE-L': 'ROUGE_L',
    'rouge_l': 'ROUGE_L',
    'meteor': 'METEOR',
    'cider': 'CIDEr',
    'spice': 'SPICE',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Append or update one paper-required experiment summary row.')
    parser.add_argument('--summary_csv', default=os.path.join('experiments', 'paper_required_experiments_summary.csv'))
    parser.add_argument('--exp_dir', default=None)
    parser.add_argument('--experiments_root', default='experiments')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--method_group', default=None)
    parser.add_argument('--status', default=None)
    parser.add_argument('--notes', default='')
    parser.add_argument('--test_metrics', default=None)
    parser.add_argument('--test_metrics_change', default=None)
    parser.add_argument('--test_metrics_nochange', default=None)
    parser.add_argument('--test_result_json', default=None)
    parser.add_argument('--best_checkpoint_json', default=None)
    parser.add_argument('--config_hash', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--selection_strategy', default=None)
    parser.add_argument('--set', dest='sets', action='append', default=[], help='Additional key=value override for the row.')
    return parser.parse_args()


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def read_csv_rows(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def latest(paths):
    existing = [path for path in paths if path and os.path.exists(path)]
    if not existing:
        return None
    return max(existing, key=lambda item: os.path.getmtime(item))


def latest_glob(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return latest(paths)


def normalize_metric_name(name):
    return ALIASES.get(name, name)


def normalize_metrics(raw):
    result = {}
    for key, value in (raw or {}).items():
        norm = normalize_metric_name(key)
        if norm in SUMMARY_COLUMNS:
            result[norm] = value
    return result


def read_config_hash(exp_dir):
    path = os.path.join(exp_dir, 'config_hash.txt') if exp_dir else None
    if not path or not os.path.exists(path):
        return ''
    values = {}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                values[key] = value
    return values.get('full_config_hash') or values.get('comparable_config_hash') or ''


def read_resolved_config(exp_dir):
    return load_json(os.path.join(exp_dir, 'resolved_config.json')) if exp_dir else None


def infer_exp_name(exp_dir, args_exp_name, resolved):
    if args_exp_name:
        return args_exp_name
    if isinstance(resolved, dict) and resolved.get('exp_name'):
        return str(resolved['exp_name'])
    if exp_dir:
        return os.path.basename(os.path.normpath(exp_dir))
    return ''


def infer_dataset(exp_name, args_dataset, resolved):
    if args_dataset:
        return args_dataset
    if isinstance(resolved, dict):
        data = resolved.get('data', {})
        if isinstance(data, dict) and data.get('dataset'):
            return str(data['dataset'])
    if exp_name.startswith('levir_mci'):
        return 'levir_mci'
    if exp_name.startswith('second_cc'):
        return 'second_cc'
    return ''


def infer_method_group(exp_name, explicit):
    if explicit:
        return explicit
    if 'external' in exp_name or 'mmodalcc' in exp_name.lower():
        return 'external_comparison'
    if 'config_check' in exp_name:
        return 'config_check'
    if 'sanity' in exp_name:
        return 'sanity_check'
    if exp_name.endswith('finetune_from_weak_best') or exp_name.startswith('levir_mci_short_') or exp_name.startswith('levir_mci_ultrashort_'):
        return 'caption_finetune'
    if 'reselect' in exp_name:
        if exp_name.startswith('levir_mci'):
            return 'levir_mci_checkpoint_reselect'
        if exp_name.startswith('second_cc'):
            return 'second_cc_checkpoint_reselect'
        return 'checkpoint_reselect'
    if exp_name.startswith('levir_mci_weak'):
        return 'levir_mci_weak_aux_pd_noreweight'
    if exp_name.startswith('second_cc_crossattn'):
        return 'second_cc_cross_attention_partial_detach'
    return 'reimplemented'


def load_best_checkpoint(path):
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    return payload


def read_metrics_csv(path, group=None):
    if not path or not os.path.exists(path):
        return {}
    rows, _ = read_csv_rows(path)
    if not rows:
        return {}
    selected = None
    if group:
        wanted = str(group).lower()
        for row in rows:
            row_group = str(row.get('group', '')).lower().replace('-', '')
            if row_group == wanted:
                selected = row
                break
    else:
        selected = rows[-1]
    return normalize_metrics(dict(selected or {}))


def load_test_payload(path):
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def metrics_from_payload(path, group=None):
    payload = load_test_payload(path)
    if not payload:
        return {}, {}
    payload_group = str(payload.get('group', '')).lower().replace('-', '')
    if group:
        if payload_group != group:
            return {}, payload
    elif payload_group in GROUP_METRIC_COLUMNS:
        return {}, payload
    metrics = normalize_metrics(payload.get('metrics', {}))
    if not group:
        metrics.update(normalize_metrics(payload.get('aux_metrics', {})))
    return metrics, payload


def first_metrics_source(sources):
    for source_type, path, group in sources:
        if not path or not os.path.exists(path):
            continue
        if source_type == 'csv':
            metrics = read_metrics_csv(path, group=group)
            payload = {}
        elif source_type == 'json':
            metrics, payload = metrics_from_payload(path, group=group)
        else:
            raise ValueError('Unknown metrics source type: %s' % source_type)
        if metrics:
            return metrics, payload, path
    return {}, {}, ''


def non_group_test_result_jsons(exp_dir):
    if not exp_dir:
        return []
    paths = []
    for path in glob.glob(os.path.join(exp_dir, 'test_*_result.json')):
        name = os.path.basename(path)
        if name in GROUP_RESULT_BASENAMES:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: os.path.getmtime(item), reverse=True)


def validate_overall_metrics_path(path):
    if not path:
        return path
    name = os.path.basename(path).lower()
    if '_change' in name or '_nochange' in name or '_no_change' in name:
        raise ValueError('Refusing to read grouped metrics as overall metrics: %s' % path)
    return path


def overall_metric_sources(exp_dir, args):
    sources = []
    if args.test_metrics:
        sources.append(('csv', validate_overall_metrics_path(args.test_metrics), None))
    if exp_dir:
        sources.extend([
            ('csv', os.path.join(exp_dir, 'test_metrics.csv'), None),
            ('csv', os.path.join(exp_dir, 'test_metrics_overall.csv'), None),
            ('json', os.path.join(exp_dir, 'test_overall_result.json'), None),
            ('json', os.path.join(exp_dir, 'test_paper_best_result.json'), None),
        ])
    if args.test_result_json:
        sources.append(('json', args.test_result_json, None))
    for path in non_group_test_result_jsons(exp_dir):
        sources.append(('json', path, None))
    return sources


def group_metric_sources(exp_dir, group, args):
    file_group = 'nochange' if group == 'nochange' else group
    explicit = args.test_metrics_nochange if group == 'nochange' else args.test_metrics_change
    sources = []
    if explicit:
        sources.append(('csv', explicit, None))
    if not exp_dir:
        return sources
    sources.extend([
        ('csv', os.path.join(exp_dir, 'test_metrics_%s.csv' % file_group), None),
        ('json', os.path.join(exp_dir, 'test_%s_result.json' % file_group), group),
        ('csv', os.path.join(exp_dir, 'test_group_summary.csv'), group),
    ])
    return sources


def update_group_metrics(row, group, metrics):
    for metric_name, summary_column in GROUP_METRIC_COLUMNS[group].items():
        value = metrics.get(metric_name, '')
        if value not in (None, ''):
            row[summary_column] = value


def merge_summary_rows(existing, incoming):
    merged = dict(existing)
    for key, value in incoming.items():
        if value not in (None, ''):
            merged[key] = value
        elif key not in merged:
            merged[key] = ''
    return merged


def has_core_caption_metrics(metrics):
    return any(metrics.get(key) not in (None, '') for key in CAPTION_COLUMNS)


def apply_key_value_overrides(row, pairs):
    for item in pairs:
        if '=' not in item:
            raise ValueError('--set expects key=value, got %s' % item)
        key, value = item.split('=', 1)
        row[key] = value


def row_key(row):
    return row.get('exp_name') or row.get('experiment') or ''


def main():
    args = parse_args()
    exp_dir = os.path.normpath(args.exp_dir) if args.exp_dir else None
    resolved = read_resolved_config(exp_dir)
    exp_name = infer_exp_name(exp_dir, args.exp_name, resolved)
    if not exp_name:
        raise ValueError('Could not infer exp_name; pass --exp_name or --exp_dir.')
    dataset = infer_dataset(exp_name, args.dataset, resolved)
    best_path = args.best_checkpoint_json or (os.path.join(exp_dir, 'best_checkpoint.json') if exp_dir else None)
    best = load_best_checkpoint(best_path)
    row = {key: '' for key in SUMMARY_COLUMNS}
    row.update({
        'dataset': dataset,
        'exp_name': exp_name,
        'method_group': infer_method_group(exp_name, args.method_group),
        'status': args.status or 'done',
        'config_hash': args.config_hash or read_config_hash(exp_dir),
        'checkpoint': args.checkpoint or best.get('selected_checkpoint') or best.get('selected_checkpoint_path') or '',
        'selection_strategy': args.selection_strategy or best.get('selection_strategy') or '',
        'notes': args.notes,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    metrics, payload, _ = first_metrics_source(overall_metric_sources(exp_dir, args))
    if payload and not row['checkpoint']:
        row['checkpoint'] = payload.get('snapshot_path', '')
    row.update(metrics)
    for group in ('change', 'nochange'):
        group_metrics, _, _ = first_metrics_source(group_metric_sources(exp_dir, group, args))
        update_group_metrics(row, group, group_metrics)
    apply_key_value_overrides(row, args.sets)
    if not has_core_caption_metrics(row) and args.status in (None, 'done') and exp_dir:
        row['status'] = 'failed'
        row['notes'] = (row['notes'] + '; ' if row['notes'] else '') + 'No test metrics found.'

    existing_rows, existing_fields = read_csv_rows(args.summary_csv)
    fieldnames = []
    for key in SUMMARY_COLUMNS + existing_fields + list(row.keys()):
        if key not in fieldnames:
            fieldnames.append(key)

    updated = False
    for index, existing in enumerate(existing_rows):
        if row_key(existing) == exp_name:
            merged = merge_summary_rows(existing, row)
            if 'experiment' in existing_fields and not merged.get('experiment'):
                merged['experiment'] = exp_name
            existing_rows[index] = merged
            updated = True
            break
    if not updated:
        if 'experiment' in fieldnames and not row.get('experiment'):
            row['experiment'] = exp_name
        existing_rows.append(row)
    write_csv_rows(args.summary_csv, existing_rows, fieldnames)
    print(json.dumps({'summary_csv': args.summary_csv, 'exp_name': exp_name, 'updated': updated, 'status': row.get('status')}, indent=2))


if __name__ == '__main__':
    main()
