import argparse
import csv
import glob
import json
import os
import time


CAPTION_COLUMNS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
AUX_COLUMNS = [
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
]
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
GROUP_RESULT_BASENAMES = {
    'test_change_result.json',
    'test_nochange_result.json',
    'val_change_result.json',
    'val_nochange_result.json',
}
REQUIRED_EXPERIMENTS = {
    'levir_mci': [
        'levir_mci_card_baseline',
        'levir_mci_card_mask_loss',
        'levir_mci_card_semantic_loss',
        'levir_mci_card_mask_semantic',
        'levir_mci_card_mask_semantic_pd05',
        'levir_mci_card_mask_semantic_pd05_noreweight',
        'levir_mci_card_mask_semantic_pd05_reweight',
        'levir_mci_ours_weak_coupled_final',
    ],
    'second_cc': [
        'second_cc_card_rgb_baseline',
        'second_cc_card_semantic_aux',
        'second_cc_card_semantic_crossattn',
        'second_cc_card_semantic_hardgate',
        'second_cc_ours_weak_coupled_final',
        'second_cc_mmodalcc_comparison',
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Upsert LEVIR-MCI / SECOND-CC paper experiment outputs into the paper summary.')
    parser.add_argument('--experiments_root', default='experiments')
    parser.add_argument('--output_csv', default=None)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--dataset', choices=['all', 'levir_mci', 'second_cc'], default='all')
    parser.add_argument('--include_unlisted', action='store_true')
    parser.add_argument('--external_mmodalcc_csv', default=os.path.join('external_results', 'second_cc_mmodalcc_results.csv'))
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


def latest_existing(paths):
    existing = [path for path in paths if path and os.path.exists(path)]
    if not existing:
        return None
    return max(existing, key=lambda path: os.path.getmtime(path))


def find_test_result(exp_dir):
    candidates = [
        os.path.join(exp_dir, 'test_paper_best_result.json'),
        os.path.join(exp_dir, 'test_overall_result.json'),
        os.path.join(exp_dir, 'val_overall_result.json'),
    ]
    for path in glob.glob(os.path.join(exp_dir, 'test_*_result.json')):
        if os.path.basename(path) not in GROUP_RESULT_BASENAMES:
            candidates.append(path)
    return latest_existing(candidates)


def find_selection_json(exp_dir):
    return latest_existing([
        os.path.join(exp_dir, 'best_checkpoint.json'),
        os.path.join(exp_dir, 'best_snapshot_for_paper.json'),
        os.path.join(exp_dir, 'best_snapshot_v2.json'),
        os.path.join(exp_dir, 'best_snapshot.json'),
    ])


def infer_dataset(exp_name):
    if exp_name.startswith('levir_mci'):
        return 'levir_mci'
    if exp_name.startswith('second_cc'):
        return 'second_cc'
    return 'unknown'


def infer_method_group(exp_name):
    if exp_name.startswith('levir_mci_short_') or exp_name.endswith('finetune_from_weak_best'):
        return 'caption_finetune'
    if exp_name.startswith('levir_mci_weak'):
        return 'levir_mci_weak_aux_pd_noreweight'
    if exp_name.startswith('second_cc_crossattn'):
        return 'second_cc_cross_attention_partial_detach'
    if 'external' in exp_name or 'mmodalcc' in exp_name.lower():
        return 'external_comparison'
    return 'reimplemented'


def read_metrics_csv(path, group=None):
    rows, _ = read_csv_rows(path)
    if not rows:
        return {}
    if group:
        wanted = group.lower().replace('-', '')
        for row in rows:
            if str(row.get('group', '')).lower().replace('-', '') == wanted:
                return row
        return {}
    return rows[-1]


def read_config_hash(exp_dir):
    path = os.path.join(exp_dir, 'config_hash.txt')
    if not os.path.exists(path):
        return ''
    values = {}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                values[key] = value
    return values.get('full_config_hash') or values.get('comparable_config_hash') or ''


def row_from_experiment(root, dataset, exp_name):
    exp_dir = os.path.join(root, exp_name)
    selection_path = find_selection_json(exp_dir)
    test_path = find_test_result(exp_dir)
    selection = load_json(selection_path) or {}
    test_result = load_json(test_path) or {}
    metrics = dict(read_metrics_csv(os.path.join(exp_dir, 'test_metrics.csv')))
    if not metrics:
        metrics.update(test_result.get('metrics', {}))
        metrics.update(test_result.get('aux_metrics', {}))
    change = read_metrics_csv(os.path.join(exp_dir, 'test_metrics_change.csv'))
    nochange = read_metrics_csv(os.path.join(exp_dir, 'test_metrics_nochange.csv'))
    checkpoint = (
        selection.get('selected_checkpoint')
        or selection.get('selected_checkpoint_path')
        or selection.get('best_snapshot')
        or test_result.get('snapshot_path', '')
    )
    has_caption_metrics = any(metrics.get(key) not in (None, '') for key in CAPTION_COLUMNS)
    status = 'done' if test_path and has_caption_metrics else 'failed'
    notes = ''
    if not os.path.isdir(exp_dir):
        status = 'failed'
        notes = 'Experiment directory missing; not summarized.'
    elif status == 'failed':
        notes = 'No overall test metrics found.'
    row = {key: '' for key in SUMMARY_COLUMNS}
    row.update({
        'dataset': dataset,
        'exp_name': exp_name,
        'method_group': infer_method_group(exp_name),
        'status': status,
        'config_hash': read_config_hash(exp_dir),
        'checkpoint': checkpoint,
        'selection_strategy': selection.get('selection_strategy', ''),
        'notes': notes,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    for key in CAPTION_COLUMNS + AUX_COLUMNS:
        row[key] = metrics.get(key, '')
    row['Change_Bleu_4'] = change.get('Bleu_4', '')
    row['Change_CIDEr'] = change.get('CIDEr', '')
    row['Change_SPICE'] = change.get('SPICE', '')
    row['NoChange_Bleu_4'] = nochange.get('Bleu_4', '')
    row['NoChange_ROUGE_L'] = nochange.get('ROUGE_L', '')
    row['NoChange_SPICE'] = nochange.get('SPICE', '')
    return row


def read_external_mmodalcc(path):
    if not os.path.exists(path):
        row = {key: '' for key in SUMMARY_COLUMNS}
        row.update({
            'dataset': 'second_cc',
            'exp_name': 'second_cc_mmodalcc_comparison',
            'method_group': 'external_comparison',
            'status': 'failed',
            'notes': 'Provide external_results/second_cc_mmodalcc_results.csv to include MModalCC; no result is fabricated.',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        return [row]
    rows, _ = read_csv_rows(path)
    normalized = []
    for raw in rows:
        exp_name = raw.get('exp_name') or raw.get('experiment') or 'second_cc_mmodalcc_comparison'
        row = {key: '' for key in SUMMARY_COLUMNS}
        row.update({
            'dataset': raw.get('dataset', 'second_cc'),
            'exp_name': exp_name,
            'method_group': 'external_comparison',
            'status': raw.get('status', 'done'),
            'checkpoint': raw.get('checkpoint') or raw.get('best_snapshot', ''),
            'notes': raw.get('notes', ''),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        for key in CAPTION_COLUMNS + AUX_COLUMNS:
            row[key] = raw.get(key, '')
        normalized.append(row)
    return normalized


def wanted_experiments(args):
    datasets = ['levir_mci', 'second_cc'] if args.dataset == 'all' else [args.dataset]
    names = []
    for dataset in datasets:
        names.extend((dataset, exp) for exp in REQUIRED_EXPERIMENTS[dataset])
    if args.include_unlisted:
        listed = {exp for _, exp in names}
        for path in sorted(glob.glob(os.path.join(args.experiments_root, '*'))):
            if not os.path.isdir(path):
                continue
            exp = os.path.basename(path)
            if exp in listed:
                continue
            dataset = infer_dataset(exp)
            if args.dataset != 'all' and dataset != args.dataset:
                continue
            names.append((dataset, exp))
    return names


def row_key(row):
    return row.get('exp_name') or row.get('experiment') or ''


def merge_summary_rows(existing, incoming):
    merged = dict(existing)
    for key, value in incoming.items():
        if value not in (None, ''):
            merged[key] = value
        elif key not in merged:
            merged[key] = ''
    return merged


def write_outputs(rows, csv_path, json_path):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    existing_rows, existing_fields = read_csv_rows(csv_path)
    ordered = list(existing_rows)
    index_by_name = {row_key(row): index for index, row in enumerate(ordered) if row_key(row)}
    for row in rows:
        name = row_key(row)
        if name in index_by_name:
            ordered[index_by_name[name]] = merge_summary_rows(ordered[index_by_name[name]], row)
        else:
            index_by_name[name] = len(ordered)
            ordered.append(row)
    fieldnames = []
    for key in SUMMARY_COLUMNS + existing_fields + [key for row in ordered for key in row.keys()]:
        if key not in fieldnames:
            fieldnames.append(key)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, '') for field in fieldnames})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    root = os.path.normpath(args.experiments_root)
    output_csv = os.path.normpath(args.output_csv or os.path.join(root, 'paper_required_experiments_summary.csv'))
    output_json = os.path.normpath(args.output_json or os.path.join(root, 'paper_required_experiments_summary.json'))

    rows = []
    for dataset, exp_name in wanted_experiments(args):
        if exp_name == 'second_cc_mmodalcc_comparison':
            rows.extend(read_external_mmodalcc(args.external_mmodalcc_csv))
        else:
            rows.append(row_from_experiment(root, dataset, exp_name))

    write_outputs(rows, output_csv, output_json)
    print(json.dumps({'output_csv': output_csv, 'output_json': output_json, 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()