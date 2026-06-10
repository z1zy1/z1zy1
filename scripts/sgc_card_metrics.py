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


def load_metrics(anno, result_json):
    from utils.eval_utils_spot import score_generation

    metrics = score_generation(anno, result_json)
    return {key: float(metrics[key]) for key in metrics if key in METRIC_COLUMNS}


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
    ]
    write_header = (not append) or (not os.path.exists(path)) or os.path.getsize(path) == 0
    mode = 'a' if append else 'w'
    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})


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
            f.write('ALL_TEST_METRICS_ABOVE_BASELINE = %s\n' % payload['ALL_TEST_METRICS_ABOVE_BASELINE'])
        else:
            for metric in METRIC_COLUMNS:
                if metric in payload['metrics']:
                    f.write('%s: %.6f\n' % (metric, payload['metrics'][metric]))
            f.write('balanced_score: %.6f\n' % payload['balanced_score'])
            f.write('ALL_ABOVE_BASELINE = %s\n' % payload['all_above_baseline'])


def main():
    args = parse_args()
    metrics = load_metrics(args.anno, args.result_json)

    payload = {
        'snapshot_path': args.snapshot_path,
        'result_json': args.result_json,
        'metrics': metrics,
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

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    if args.output_txt:
        write_text(args.output_txt, payload)

    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
