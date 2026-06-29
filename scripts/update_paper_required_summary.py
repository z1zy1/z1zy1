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
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_csv_rows(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline='', encoding='utf-8') as f:
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
    with open(path, encoding='utf-8') as f:
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
    if exp_name.endswith('finetune_from_weak_best'):
        return 'caption_finetune'
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


def read_metrics_csv(path):
    if not path or not os.path.exists(path):
        return {}
    rows, _ = read_csv_rows(path)
    return dict(rows[-1]) if rows else {}


def load_test_payload(path):
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def update_from_group_payload(row, payload):
    groups = payload.get('group_metrics', {}) if isinstance(payload, dict) else {}
    change = groups.get('change', {}).get('metrics', {}) if isinstance(groups.get('change'), dict) else {}
    nochange = groups.get('nochange', {}).get('metrics', {}) if isinstance(groups.get('nochange'), dict) else {}
    if change:
        row['Change_Bleu_4'] = change.get('Bleu_4', row.get('Change_Bleu_4', ''))
        row['Change_CIDEr'] = change.get('CIDEr', row.get('Change_CIDEr', ''))
        row['Change_SPICE'] = change.get('SPICE', row.get('Change_SPICE', ''))
    if nochange:
        row['NoChange_Bleu_4'] = nochange.get('Bleu_4', row.get('NoChange_Bleu_4', ''))
        row['NoChange_ROUGE_L'] = nochange.get('ROUGE_L', row.get('NoChange_ROUGE_L', ''))
        row['NoChange_SPICE'] = nochange.get('SPICE', row.get('NoChange_SPICE', ''))


def update_from_group_csv(row, exp_dir):
    if not exp_dir:
        return
    change_csv = os.path.join(exp_dir, 'test_metrics_change.csv')
    nochange_csv = os.path.join(exp_dir, 'test_metrics_nochange.csv')
    change = read_metrics_csv(change_csv)
    nochange = read_metrics_csv(nochange_csv)
    if change:
        row['Change_Bleu_4'] = change.get('Bleu_4', row.get('Change_Bleu_4', ''))
        row['Change_CIDEr'] = change.get('CIDEr', row.get('Change_CIDEr', ''))
        row['Change_SPICE'] = change.get('SPICE', row.get('Change_SPICE', ''))
    if nochange:
        row['NoChange_Bleu_4'] = nochange.get('Bleu_4', row.get('NoChange_Bleu_4', ''))
        row['NoChange_ROUGE_L'] = nochange.get('ROUGE_L', row.get('NoChange_ROUGE_L', ''))
        row['NoChange_SPICE'] = nochange.get('SPICE', row.get('NoChange_SPICE', ''))


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
    test_csv = args.test_metrics or (os.path.join(exp_dir, 'test_metrics.csv') if exp_dir else None)
    test_json = args.test_result_json or latest_glob([os.path.join(exp_dir, 'test_*_result.json')]) if exp_dir else args.test_result_json

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

    metrics = normalize_metrics(read_metrics_csv(test_csv))
    payload = load_test_payload(test_json)
    if payload:
        metrics.update(normalize_metrics(payload.get('metrics', {})))
        metrics.update(normalize_metrics(payload.get('aux_metrics', {})))
        if not row['checkpoint']:
            row['checkpoint'] = payload.get('snapshot_path', '')
        update_from_group_payload(row, payload)
    update_from_group_csv(row, exp_dir)
    row.update(metrics)
    if not metrics and args.status is None and exp_dir:
        row['status'] = 'failed'
        row['notes'] = (row['notes'] + '; ' if row['notes'] else '') + 'No test metrics found.'
    apply_key_value_overrides(row, args.sets)

    existing_rows, existing_fields = read_csv_rows(args.summary_csv)
    fieldnames = []
    for key in SUMMARY_COLUMNS + existing_fields + list(row.keys()):
        if key not in fieldnames:
            fieldnames.append(key)

    updated = False
    for index, existing in enumerate(existing_rows):
        if row_key(existing) == exp_name:
            merged = dict(existing)
            merged.update(row)
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
