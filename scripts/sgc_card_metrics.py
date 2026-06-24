import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath('.'))


METRIC_COLUMNS = [
    'Bleu_1',
    'Bleu_2',
    'Bleu_3',
    'Bleu_4',
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'SPICE',
]

AUX_METRIC_COLUMNS = [
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

BASELINE_EVAL = {
    'Bleu_4': 0.4375,
    'METEOR': 0.3377,
    'ROUGE_L': 0.6942,
    'CIDEr': 1.2299,
    'SPICE': 0.2607,
}

BASELINE_TEST = {
    'Bleu_1': 0.828,
    'Bleu_2': 0.728,
    'Bleu_3': 0.641,
    'Bleu_4': 0.565,
    'METEOR': 0.391,
    'ROUGE_L': 0.746,
    'CIDEr': 1.348,
    'SPICE': 0.336,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Score SGC-CARD caption JSON and write structured metrics.')
    parser.add_argument('--anno', required=True, help='COCO-format annotation JSON.')
    parser.add_argument('--result_json', required=True, help='Generated sc_results.json.')
    parser.add_argument('--snapshot_path', default='', help='Snapshot path to record in CSV/JSON.')
    parser.add_argument('--baseline', choices=['eval', 'test'], default='eval')
    parser.add_argument('--csv', default=None, help='Optional CSV output path.')
    parser.add_argument('--append', action='store_true', help='Append one row to --csv.')
    parser.add_argument('--output_json', default=None, help='Optional structured JSON output path.')
    parser.add_argument('--output_txt', default=None, help='Optional human-readable text output path.')
    parser.add_argument('--aux_metrics_json', default=None, help='Optional aux_metrics.json from test_card_spot.py.')
    parser.add_argument('--eval_change_nochange_split', action='store_true')
    parser.add_argument('--changeflag_json', default=None, help='Dataset JSON containing image changeflag fields.')
    parser.add_argument('--split', default=None, help='Optional split filter for change/no-change grouping.')
    parser.add_argument('--group_output_dir', default=None, help='Directory for group JSON/CSV outputs.')
    return parser.parse_args()


def balanced_score(metrics):
    return (
        0.25 * (metrics.get('CIDEr', 0.0) / BASELINE_EVAL['CIDEr'])
        + 0.25 * (metrics.get('SPICE', 0.0) / BASELINE_EVAL['SPICE'])
        + 0.20 * (metrics.get('Bleu_4', 0.0) / BASELINE_EVAL['Bleu_4'])
        + 0.15 * (metrics.get('METEOR', 0.0) / BASELINE_EVAL['METEOR'])
        + 0.15 * (metrics.get('ROUGE_L', 0.0) / BASELINE_EVAL['ROUGE_L'])
    )


def all_above(metrics, baseline):
    return all(metrics.get(metric, float('-inf')) > value for metric, value in baseline.items())


def metric_subset(metrics):
    return {key: float(metrics[key]) for key in metrics if key in METRIC_COLUMNS}


def load_metrics(anno, result_json):
    from utils.eval_utils_spot import score_generation

    metrics = score_generation(anno, result_json)
    return metric_subset(metrics)


def load_metrics_with_ids(anno, result_json, image_ids):
    from utils.eval_utils_spot import score_generation_with_ids

    if not image_ids:
        return {}
    metrics = score_generation_with_ids(anno, result_json, image_ids)
    return metric_subset(metrics)


def load_result_ids(result_json):
    with open(result_json, encoding='utf-8') as f:
        rows = json.load(f)
    ids = []
    for row in rows:
        image_id = row.get('image_id')
        if image_id is not None:
            ids.append(str(image_id))
    return ids


def normalize_id(value):
    return os.path.basename(str(value))


def load_changeflag_groups(path, split=None, valid_ids=None):
    if not path:
        raise ValueError('--changeflag_json is required when --eval_change_nochange_split is enabled.')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    items = data.get('images', data) if isinstance(data, dict) else data
    valid = set(valid_ids or [])
    valid_with_basename = set(valid) | {normalize_id(item) for item in valid}
    change_ids = []
    nochange_ids = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_split = item.get('split') or item.get('filepath')
        if split and item_split and str(item_split) != str(split):
            continue
        if 'changeflag' not in item:
            continue
        raw_id = item.get('filename') or item.get('image_id') or item.get('id') or item.get('imgid')
        if raw_id is None:
            continue
        image_id = normalize_id(raw_id)
        if valid_with_basename and image_id not in valid_with_basename and str(raw_id) not in valid_with_basename:
            continue
        try:
            flag = int(item.get('changeflag'))
        except (TypeError, ValueError):
            continue
        if flag == 1:
            change_ids.append(image_id)
        elif flag == 0:
            nochange_ids.append(image_id)
    return {
        'change': sorted(set(change_ids)),
        'nochange': sorted(set(nochange_ids)),
    }


def infer_aux_metrics_path(result_json, explicit_path=None):
    if explicit_path:
        return explicit_path
    candidate = os.path.join(os.path.dirname(result_json), 'aux_metrics.json')
    return candidate if os.path.exists(candidate) else None


def load_aux_metrics(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def write_csv_row(path, row, append):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fieldnames = [
        'snapshot_path',
        'Bleu_1',
        'Bleu_2',
        'Bleu_3',
        'Bleu_4',
        'METEOR',
        'ROUGE_L',
        'CIDEr',
        'SPICE',
        'balanced_score',
        'all_above_baseline',
    ] + AUX_METRIC_COLUMNS
    write_header = (not append) or (not os.path.exists(path)) or os.path.getsize(path) == 0
    mode = 'a' if append else 'w'
    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_group_outputs(output_dir, split_name, snapshot_path, overall_payload, group_payloads):
    os.makedirs(output_dir, exist_ok=True)
    overall_path = os.path.join(output_dir, '%s_overall_result.json' % split_name)
    with open(overall_path, 'w', encoding='utf-8') as f:
        json.dump(overall_payload, f, indent=2)

    rows = []
    for group_name, payload in group_payloads.items():
        file_group_name = 'no-change' if group_name == 'nochange' else group_name
        path = os.path.join(output_dir, '%s_%s_result.json' % (split_name, file_group_name))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        metrics = payload.get('metrics', {})
        rows.append({
            'group': group_name,
            'num_images': payload.get('num_images', 0),
            'note': payload.get('note', ''),
            **{metric: metrics.get(metric, '') for metric in METRIC_COLUMNS},
        })

    csv_path = os.path.join(output_dir, '%s_group_summary.csv' % split_name)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['group', 'num_images'] + METRIC_COLUMNS + ['note']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})
    return overall_path, csv_path


def write_text(path, payload):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if payload['baseline_name'] == 'test':
            f.write('metric, baseline, ours, delta\n')
            for metric in METRIC_COLUMNS:
                baseline = payload['baseline'].get(metric)
                ours = payload['metrics'].get(metric)
                if baseline is None or ours is None:
                    continue
                f.write('%s, %.6f, %.6f, %.6f\n' % (metric, baseline, ours, payload['deltas'][metric]))
            for metric in AUX_METRIC_COLUMNS:
                if metric in payload.get('aux_metrics', {}):
                    f.write('%s, , %.6f, \n' % (metric, payload['aux_metrics'][metric]))
            f.write('ALL_TEST_METRICS_ABOVE_BASELINE = %s\n' % payload['ALL_TEST_METRICS_ABOVE_BASELINE'])
        else:
            for metric in METRIC_COLUMNS:
                if metric in payload['metrics']:
                    f.write('%s: %.6f\n' % (metric, payload['metrics'][metric]))
            for metric in AUX_METRIC_COLUMNS:
                if metric in payload.get('aux_metrics', {}):
                    f.write('%s: %.6f\n' % (metric, payload['aux_metrics'][metric]))
            f.write('balanced_score: %.6f\n' % payload['balanced_score'])
            f.write('ALL_ABOVE_BASELINE = %s\n' % payload['all_above_baseline'])


def main():
    args = parse_args()
    metrics = load_metrics(args.anno, args.result_json)
    aux_metrics = load_aux_metrics(infer_aux_metrics_path(args.result_json, args.aux_metrics_json))

    payload = {
        'snapshot_path': args.snapshot_path,
        'result_json': args.result_json,
        'metrics': metrics,
        'aux_metrics': aux_metrics,
        'baseline_name': args.baseline,
    }

    if args.baseline == 'eval':
        score = balanced_score(metrics)
        is_all_above = all_above(metrics, BASELINE_EVAL)
        payload.update({
            'baseline': BASELINE_EVAL,
            'balanced_score': score,
            'all_above_baseline': is_all_above,
        })
        if args.csv:
            row = {'snapshot_path': args.snapshot_path, 'balanced_score': score, 'all_above_baseline': is_all_above}
            row.update(metrics)
            row.update(aux_metrics)
            write_csv_row(args.csv, row, args.append)
    else:
        deltas = {
            metric: metrics.get(metric, 0.0) - baseline
            for metric, baseline in BASELINE_TEST.items()
        }
        payload.update({
            'baseline': BASELINE_TEST,
            'deltas': deltas,
            'ALL_TEST_METRICS_ABOVE_BASELINE': all_above(metrics, BASELINE_TEST),
        })

    if args.eval_change_nochange_split:
        result_ids = load_result_ids(args.result_json)
        groups = load_changeflag_groups(args.changeflag_json, args.split, result_ids)
        group_payloads = {}
        for group_name, ids in groups.items():
            group_metrics = load_metrics_with_ids(args.anno, args.result_json, ids)
            note = ''
            if group_name == 'nochange':
                note = 'No-change CIDEr is reported for completeness and is not used as a core selection criterion.'
            group_payloads[group_name] = {
                'snapshot_path': args.snapshot_path,
                'result_json': args.result_json,
                'group': group_name,
                'num_images': len(ids),
                'metrics': group_metrics,
                'note': note,
            }
        payload['group_metrics'] = group_payloads
        split_name = args.split or ('test' if args.baseline == 'test' else 'val')
        output_dir = args.group_output_dir or os.path.dirname(args.output_json or args.result_json) or '.'
        overall_path, group_csv = write_group_outputs(output_dir, split_name, args.snapshot_path, payload, group_payloads)
        payload['group_output_json'] = overall_path
        payload['group_summary_csv'] = group_csv

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    if args.output_txt:
        write_text(args.output_txt, payload)

    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
