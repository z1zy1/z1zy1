import argparse
import csv
import math
import os
import re
import statistics


METRICS = [
    'Bleu_1',
    'Bleu_2',
    'Bleu_3',
    'Bleu_4',
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'SPICE',
]

ALIASES = {
    'BLEU1': 'Bleu_1',
    'BLEU2': 'Bleu_2',
    'BLEU3': 'Bleu_3',
    'BLEU4': 'Bleu_4',
    'B1': 'Bleu_1',
    'B2': 'Bleu_2',
    'B3': 'Bleu_3',
    'B4': 'Bleu_4',
    'ROUGE-L': 'ROUGE_L',
    'ROUGE_L': 'ROUGE_L',
}

NUMBER_RE = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
METRIC_RE = re.compile(r'^\s*([A-Za-z0-9_-]+)\s*:\s*' + NUMBER_RE)


def normalize_metric(name):
    canonical = ALIASES.get(name.upper())
    if canonical:
        return canonical
    for metric in METRICS:
        if name.lower() == metric.lower():
            return metric
    return None


def read_text(path):
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'cp936'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, 'r', errors='replace') as f:
        return f.read()


def parse_metrics(path):
    metrics = {}
    text = read_text(path)
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        metric = normalize_metric(match.group(1))
        if metric is not None and metric not in metrics:
            metrics[metric] = float(match.group(2))
    missing = [metric for metric in METRICS if metric not in metrics]
    if missing:
        raise RuntimeError('Missing metrics in %s: %s' % (path, ', '.join(missing)))
    return metrics


def infer_default_inputs(results_root):
    names = [
        'lmask002_lsem01_semantic_detach_seed1',
        'lmask002_lsem01_semantic_detach_seed2',
        'lmask002_lsem01_semantic_detach_seed3',
    ]
    return [os.path.join(results_root, name, 'test_results.txt') for name in names]


def compute_summary(rows):
    summary = []
    for metric in METRICS:
        values = [row[metric] for row in rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append({
            'Metric': metric,
            'Mean': mean,
            'Std': std,
            'Count': len(values),
        })
    return summary


def save_csv(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'Mean', 'Std', 'Count'])
        writer.writeheader()
        for row in summary:
            writer.writerow({
                'Metric': row['Metric'],
                'Mean': '%.6f' % row['Mean'],
                'Std': '%.6f' % row['Std'],
                'Count': row['Count'],
            })


def save_markdown(path, summary, inputs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# lmask002_lsem01_semantic_detach seed summary\n\n')
        f.write('Inputs:\n')
        for input_path in inputs:
            f.write('- `%s`\n' % input_path)
        f.write('\n| Metric | Mean | Std |\n')
        f.write('|---|---:|---:|\n')
        for row in summary:
            f.write('| %s | %.6f | %.6f |\n' % (row['Metric'], row['Mean'], row['Std']))


def print_summary(summary):
    print('Metric\tMean\tStd')
    for row in summary:
        print('%s\t%.6f\t%.6f' % (row['Metric'], row['Mean'], row['Std']))


def main():
    parser = argparse.ArgumentParser(description='Summarize test metrics across semantic-detach seeds.')
    parser.add_argument('--results_root', default='./results')
    parser.add_argument('--inputs', nargs='*', default=None)
    parser.add_argument('--output_csv', default='./results/lmask002_lsem01_semantic_detach_seed_summary.csv')
    parser.add_argument('--output_md', default='./results/lmask002_lsem01_semantic_detach_seed_summary.md')
    args = parser.parse_args()

    inputs = args.inputs or infer_default_inputs(args.results_root)
    missing = [path for path in inputs if not os.path.exists(path)]
    if missing:
        raise SystemExit('Missing input test result file(s): %s' % ', '.join(missing))

    rows = [parse_metrics(path) for path in inputs]
    summary = compute_summary(rows)
    save_csv(args.output_csv, summary)
    save_markdown(args.output_md, summary, inputs)
    print_summary(summary)
    print('Saved CSV: %s' % args.output_csv)
    print('Saved Markdown: %s' % args.output_md)


if __name__ == '__main__':
    main()
