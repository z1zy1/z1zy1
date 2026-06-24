import argparse
import csv
import glob
import json
import os
import re
import shutil


CAPTION_BASELINE = {
    'Bleu_4': 0.4375,
    'METEOR': 0.3377,
    'ROUGE_L': 0.6942,
    'CIDEr': 1.2299,
    'SPICE': 0.2607,
}

CAPTION_COLUMNS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
AUX_COLUMNS = ['Mask_F1', 'Mask_IoU', 'Mask_mIoU', 'IoU_road', 'IoU_building', 'Semantic_mIoU']
NEGATIVE_ABLATION_TOKENS = ('hard_gate', 'hardgate', 'reweight', 'rw')


def parse_args():
    parser = argparse.ArgumentParser(description='Select the paper snapshot from dataset-aware eval_snapshots.csv.')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--csv', default=None)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--copy_path', default=None)
    parser.add_argument('--allow_negative_ablation', action='store_true')
    parser.add_argument('--metric', default='paper_balanced', choices=['paper_balanced', 'CIDEr', 'Bleu_4', 'SPICE', 'METEOR'])
    return parser.parse_args()


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_bool(value):
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def checkpoint_number(path):
    matches = re.findall(r'(\d+)', os.path.basename(path or ''))
    return matches[-1] if matches else None


def resolve_snapshot(raw_path, exp_dir):
    raw_path = raw_path or ''
    candidates = [
        raw_path,
        os.path.abspath(raw_path),
        os.path.join(exp_dir, raw_path),
        os.path.join(exp_dir, os.path.basename(raw_path)),
        os.path.join(exp_dir, 'snapshots', os.path.basename(raw_path)),
        os.path.join(exp_dir, 'checkpoints', os.path.basename(raw_path)),
    ]
    root, ext = os.path.splitext(raw_path)
    if ext.lower() == '.pth':
        candidates.append(root + '.pt')
    elif ext.lower() == '.pt':
        candidates.append(root + '.pth')
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.normpath(path)

    number = checkpoint_number(raw_path)
    patterns = []
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    if stem:
        patterns.extend([
            os.path.join(exp_dir, 'snapshots', stem + '.*'),
            os.path.join(exp_dir, 'checkpoints', stem + '.*'),
        ])
    if number:
        patterns.extend([
            os.path.join(exp_dir, 'snapshots', '*%s*.pt' % number),
            os.path.join(exp_dir, 'snapshots', '*%s*.pth' % number),
            os.path.join(exp_dir, 'checkpoints', '*%s*.pt' % number),
            os.path.join(exp_dir, 'checkpoints', '*%s*.pth' % number),
        ])
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted({os.path.normpath(path) for path in matches if os.path.exists(path)})
    return matches[0] if matches else os.path.normpath(raw_path)


def is_negative_ablation(exp_dir, snapshot_path):
    text = ('%s %s' % (exp_dir, snapshot_path)).lower()
    return any(token in text for token in NEGATIVE_ABLATION_TOKENS)


def caption_score(row):
    return (
        0.30 * (to_float(row.get('CIDEr')) / CAPTION_BASELINE['CIDEr'])
        + 0.25 * (to_float(row.get('SPICE')) / CAPTION_BASELINE['SPICE'])
        + 0.20 * (to_float(row.get('Bleu_4')) / CAPTION_BASELINE['Bleu_4'])
        + 0.15 * (to_float(row.get('METEOR')) / CAPTION_BASELINE['METEOR'])
        + 0.10 * (to_float(row.get('ROUGE_L')) / CAPTION_BASELINE['ROUGE_L'])
    )


def aux_tiebreak(row):
    values = []
    for key in ('Mask_F1', 'Mask_IoU', 'Mask_mIoU', 'Semantic_mIoU'):
        if row.get(key) not in (None, ''):
            values.append(max(0.0, min(1.0, to_float(row.get(key)))))
    return sum(values) / float(len(values)) if values else 0.0


def score_row(row, metric):
    if metric != 'paper_balanced':
        return to_float(row.get(metric), -float('inf'))
    return caption_score(row) + 0.03 * aux_tiebreak(row)


def load_rows(csv_path, exp_dir, metric):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('eval snapshot CSV not found: %s' % csv_path)
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if not rows:
        raise RuntimeError('No rows in %s' % csv_path)
    normalized = []
    for row in rows:
        row['snapshot_path'] = resolve_snapshot(row.get('snapshot_path', ''), exp_dir)
        row['paper_score'] = score_row(row, metric)
        row['caption_score'] = caption_score(row)
        row['aux_tiebreak'] = aux_tiebreak(row)
        row['negative_ablation'] = is_negative_ablation(exp_dir, row['snapshot_path'])
        row['all_above_baseline'] = parse_bool(row.get('all_above_baseline'))
        normalized.append(row)
    return normalized


def copy_or_link(src, dst):
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


def row_payload(row):
    payload = {
        'snapshot_path': row.get('snapshot_path', ''),
        'paper_score': to_float(row.get('paper_score')),
        'caption_score': to_float(row.get('caption_score')),
        'aux_tiebreak': to_float(row.get('aux_tiebreak')),
        'negative_ablation': bool(row.get('negative_ablation')),
        'all_above_baseline': bool(row.get('all_above_baseline')),
    }
    payload['metrics'] = {key: to_float(row.get(key)) for key in CAPTION_COLUMNS if row.get(key) not in (None, '')}
    payload['aux_metrics'] = {key: to_float(row.get(key)) for key in AUX_COLUMNS if row.get(key) not in (None, '')}
    return payload


def main():
    args = parse_args()
    exp_dir = os.path.normpath(args.exp_dir)
    csv_path = os.path.normpath(args.csv or os.path.join(exp_dir, 'eval_snapshots.csv'))
    output_json = os.path.normpath(args.output_json or os.path.join(exp_dir, 'best_snapshot_for_paper.json'))
    copy_path = os.path.normpath(args.copy_path or os.path.join(exp_dir, 'best_for_paper.pth'))

    rows = load_rows(csv_path, exp_dir, args.metric)
    eligible = rows if args.allow_negative_ablation else [row for row in rows if not row['negative_ablation']]
    if not eligible:
        eligible = rows
    ranked = sorted(
        eligible,
        key=lambda row: (
            row['paper_score'],
            to_float(row.get('CIDEr')),
            to_float(row.get('Bleu_4')),
            to_float(row.get('SPICE')),
        ),
        reverse=True,
    )
    best = ranked[0]
    copy_or_link(best['snapshot_path'], copy_path)

    payload = {
        'exp_dir': exp_dir,
        'csv': csv_path,
        'selection_metric': args.metric,
        'allow_negative_ablation': args.allow_negative_ablation,
        'best_snapshot': best['snapshot_path'],
        'copy_path': copy_path,
        'best': row_payload(best),
        'top_candidates': [row_payload(row) for row in ranked[:10]],
        'negative_ablation_policy': 'hard_gate/reweight experiments are excluded unless --allow_negative_ablation is set.',
    }
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
