import argparse
import csv
import glob
import json
import os
import re
import shutil


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

STRICT_FILTER = {
    'Bleu_4': 0.56,
    'CIDEr': 1.31,
    'METEOR': 0.386,
}

RELAXED_FILTER = {
    'Bleu_4': 0.54,
    'CIDEr': 1.28,
    'METEOR': 0.37,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Select SGC-CARD best snapshot with v2 two-stage filtering.')
    parser.add_argument('--csv', default=None, help='Path to eval_snapshots.csv.')
    parser.add_argument('--exp_dir', default=os.path.join('experiments', 'sgc_card_lm003_ls005_pd05_rw02_warmup'))
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--copy_path', default=None)
    parser.add_argument('--strict_bleu4', type=float, default=STRICT_FILTER['Bleu_4'])
    parser.add_argument('--strict_cider', type=float, default=STRICT_FILTER['CIDEr'])
    parser.add_argument('--strict_meteor', type=float, default=STRICT_FILTER['METEOR'])
    parser.add_argument('--relaxed_bleu4', type=float, default=RELAXED_FILTER['Bleu_4'])
    parser.add_argument('--relaxed_cider', type=float, default=RELAXED_FILTER['CIDEr'])
    parser.add_argument('--relaxed_meteor', type=float, default=RELAXED_FILTER['METEOR'])
    return parser.parse_args()


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    return float(value)


def score_v2(row):
    return (
        0.30 * (to_float(row.get('CIDEr')) / BASELINE_EVAL['CIDEr'])
        + 0.25 * (to_float(row.get('Bleu_4')) / BASELINE_EVAL['Bleu_4'])
        + 0.20 * (to_float(row.get('SPICE')) / BASELINE_EVAL['SPICE'])
        + 0.15 * (to_float(row.get('METEOR')) / BASELINE_EVAL['METEOR'])
        + 0.10 * (to_float(row.get('ROUGE_L')) / BASELINE_EVAL['ROUGE_L'])
    )


def all_above_eval_baseline(row):
    return all(to_float(row.get(metric), -float('inf')) > baseline for metric, baseline in BASELINE_EVAL.items())


def not_above_eval_baseline_metrics(row):
    return [
        metric for metric, baseline in BASELINE_EVAL.items()
        if to_float(row.get(metric), -float('inf')) <= baseline
    ]


def checkpoint_number(path):
    basename = os.path.basename(path or '')
    matches = re.findall(r'(\d+)', basename)
    return matches[-1] if matches else None


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return os.path.normpath(path)
    return None


def resolve_snapshot_path(raw_path, exp_dir):
    raw_path = (raw_path or '').strip()
    candidates = []
    if raw_path:
        candidates.extend([
            raw_path,
            os.path.abspath(raw_path),
            os.path.join(exp_dir, raw_path),
            os.path.join(exp_dir, os.path.basename(raw_path)),
            os.path.join(exp_dir, 'snapshots', os.path.basename(raw_path)),
            os.path.join(exp_dir, 'checkpoints', os.path.basename(raw_path)),
        ])
        root, ext = os.path.splitext(raw_path)
        if ext.lower() == '.pth':
            candidates.append(root + '.pt')
        elif ext.lower() == '.pt':
            candidates.append(root + '.pth')
    found = first_existing(candidates)
    if found is not None:
        return found

    number = checkpoint_number(raw_path)
    patterns = []
    if raw_path:
        basename_no_ext = os.path.splitext(os.path.basename(raw_path))[0]
        patterns.extend([
            os.path.join(exp_dir, 'snapshots', basename_no_ext + '.*'),
            os.path.join(exp_dir, 'checkpoints', basename_no_ext + '.*'),
            os.path.join(exp_dir, basename_no_ext + '.*'),
        ])
    if number:
        patterns.extend([
            os.path.join(exp_dir, 'snapshots', '*%s*.pt' % number),
            os.path.join(exp_dir, 'snapshots', '*%s*.pth' % number),
            os.path.join(exp_dir, 'checkpoints', '*%s*.pt' % number),
            os.path.join(exp_dir, 'checkpoints', '*%s*.pth' % number),
            os.path.join(exp_dir, '*%s*.pt' % number),
            os.path.join(exp_dir, '*%s*.pth' % number),
        ])
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted({os.path.normpath(path) for path in matches if os.path.exists(path)})
    return matches[0] if matches else os.path.normpath(raw_path)


def load_rows(csv_path, exp_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('eval CSV does not exist: %s' % csv_path)
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('No snapshot rows found in %s' % csv_path)

    normalized = []
    for row in rows:
        row = dict(row)
        row['snapshot_path'] = resolve_snapshot_path(row.get('snapshot_path', ''), exp_dir)
        row['score_v2'] = score_v2(row)
        row['all_above_eval_baseline'] = all_above_eval_baseline(row)
        row['not_above_eval_baseline_metrics'] = not_above_eval_baseline_metrics(row)
        normalized.append(row)
    return normalized


def passes_filter(row, thresholds):
    return all(to_float(row.get(metric), -float('inf')) >= threshold for metric, threshold in thresholds.items())


def select_candidates(rows, strict_filter, relaxed_filter):
    strict = [row for row in rows if passes_filter(row, strict_filter)]
    if strict:
        return strict, 'strict', 'Selected from strict quality filter: Bleu_4 >= %.3f, CIDEr >= %.3f, METEOR >= %.3f.' % (
            strict_filter['Bleu_4'], strict_filter['CIDEr'], strict_filter['METEOR']
        )

    relaxed = [row for row in rows if passes_filter(row, relaxed_filter)]
    if relaxed:
        return relaxed, 'relaxed', 'Selected from relaxed quality filter: Bleu_4 >= %.3f, CIDEr >= %.3f, METEOR >= %.3f.' % (
            relaxed_filter['Bleu_4'], relaxed_filter['CIDEr'], relaxed_filter['METEOR']
        )

    return rows, 'fallback', 'WARNING: no snapshot satisfies quality filters, fallback to all snapshots.'


def row_metrics(row):
    return {
        metric: to_float(row.get(metric))
        for metric in METRIC_COLUMNS
        if row.get(metric) not in (None, '')
    }


def candidate_summary(row):
    return {
        'snapshot_path': row['snapshot_path'],
        'Bleu_4': to_float(row.get('Bleu_4')),
        'METEOR': to_float(row.get('METEOR')),
        'ROUGE_L': to_float(row.get('ROUGE_L')),
        'CIDEr': to_float(row.get('CIDEr')),
        'SPICE': to_float(row.get('SPICE')),
        'score_v2': float(row['score_v2']),
    }


def copy_or_link_snapshot(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError('Selected snapshot does not exist: %s' % src)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    exp_dir = os.path.normpath(args.exp_dir)
    csv_path = os.path.normpath(args.csv or os.path.join(exp_dir, 'eval_snapshots.csv'))
    output_json = os.path.normpath(args.output_json or os.path.join(exp_dir, 'best_snapshot_v2.json'))
    copy_path = os.path.normpath(args.copy_path or os.path.join(exp_dir, 'best_balanced_v2.pth'))

    strict_filter = {
        'Bleu_4': args.strict_bleu4,
        'CIDEr': args.strict_cider,
        'METEOR': args.strict_meteor,
    }
    relaxed_filter = {
        'Bleu_4': args.relaxed_bleu4,
        'CIDEr': args.relaxed_cider,
        'METEOR': args.relaxed_meteor,
    }

    rows = load_rows(csv_path, exp_dir)
    candidates, filter_level, selection_reason = select_candidates(rows, strict_filter, relaxed_filter)
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (
            row['score_v2'],
            to_float(row.get('Bleu_4')),
            to_float(row.get('CIDEr')),
            to_float(row.get('METEOR')),
        ),
        reverse=True,
    )
    best = sorted_candidates[0]

    payload = {
        'best_snapshot': best['snapshot_path'],
        'selection_version': 'v2_two_stage_filter_score',
        'selection_reason': selection_reason,
        'metrics': row_metrics(best),
        'score_v2': float(best['score_v2']),
        'all_above_eval_baseline': bool(best['all_above_eval_baseline']),
        'not_above_eval_baseline_metrics': best['not_above_eval_baseline_metrics'],
        'filter_level': filter_level,
        'candidate_count': len(candidates),
        'top_candidates': [candidate_summary(row) for row in sorted_candidates[:5]],
    }

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    copy_or_link_snapshot(best['snapshot_path'], copy_path)

    print(json.dumps(payload, indent=2))
    print('Saved v2 selection JSON: %s' % output_json)
    print('Saved v2 balanced snapshot: %s' % copy_path)


if __name__ == '__main__':
    main()
