import argparse
import csv
import glob
import json
import os


CAPTION_COLUMNS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
AUX_COLUMNS = ['Mask_Precision', 'Mask_Recall', 'Mask_F1', 'Mask_IoU', 'Mask_mIoU', 'IoU_road', 'IoU_building', 'Semantic_mIoU']
FIELDNAMES = [
    'dataset',
    'experiment',
    'source',
    'status',
    'best_snapshot',
    'selection_json',
    'test_result_json',
] + CAPTION_COLUMNS + AUX_COLUMNS + ['notes']

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
    parser = argparse.ArgumentParser(description='Summarize LEVIR-MCI / SECOND-CC paper experiment outputs.')
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
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def latest_existing(paths):
    existing = [path for path in paths if path and os.path.exists(path)]
    if not existing:
        return None
    return max(existing, key=lambda path: os.path.getmtime(path))


def find_test_result(exp_dir):
    candidates = glob.glob(os.path.join(exp_dir, 'test_*_result.json'))
    candidates += [
        os.path.join(exp_dir, 'test_overall_result.json'),
        os.path.join(exp_dir, 'val_overall_result.json'),
    ]
    return latest_existing(candidates)


def find_selection_json(exp_dir):
    return latest_existing([
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


def row_from_experiment(root, dataset, exp_name):
    exp_dir = os.path.join(root, exp_name)
    selection_path = find_selection_json(exp_dir)
    test_path = find_test_result(exp_dir)
    selection = load_json(selection_path) or {}
    test_result = load_json(test_path) or {}
    metrics = test_result.get('metrics', {})
    aux = test_result.get('aux_metrics', {})
    best_snapshot = selection.get('best_snapshot') or test_result.get('snapshot_path', '')
    status = 'complete' if test_path and metrics else 'missing_test_result'
    if not os.path.isdir(exp_dir):
        status = 'missing_experiment_dir'
    row = {
        'dataset': dataset,
        'experiment': exp_name,
        'source': 'reimplemented',
        'status': status,
        'best_snapshot': best_snapshot,
        'selection_json': selection_path or '',
        'test_result_json': test_path or '',
        'notes': '',
    }
    for key in CAPTION_COLUMNS:
        row[key] = metrics.get(key, '')
    for key in AUX_COLUMNS:
        row[key] = aux.get(key, '')
    return row


def read_external_mmodalcc(path):
    if not os.path.exists(path):
        return [{
            'dataset': 'second_cc',
            'experiment': 'second_cc_mmodalcc_comparison',
            'source': 'N/A',
            'status': 'missing_external_results',
            'best_snapshot': '',
            'selection_json': '',
            'test_result_json': '',
            **{key: '' for key in CAPTION_COLUMNS + AUX_COLUMNS},
            'notes': 'Provide external_results/second_cc_mmodalcc_results.csv to include MModalCC; no result is fabricated.',
        }]
    with open(path, newline='', encoding='utf-8') as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    normalized = []
    for index, raw in enumerate(rows):
        row = {
            'dataset': raw.get('dataset', 'second_cc'),
            'experiment': raw.get('experiment', 'second_cc_mmodalcc_comparison'),
            'source': raw.get('source', 'external_reported'),
            'status': raw.get('status', 'complete'),
            'best_snapshot': raw.get('best_snapshot', ''),
            'selection_json': '',
            'test_result_json': path,
            'notes': raw.get('notes', ''),
        }
        for key in CAPTION_COLUMNS + AUX_COLUMNS:
            row[key] = raw.get(key, '')
        if index == 0 and not row['source']:
            row['source'] = 'external_reported'
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


def write_outputs(rows, csv_path, json_path):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in FIELDNAMES})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)


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
