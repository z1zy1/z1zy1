import argparse
import csv
import os
import sys
from decimal import Decimal, InvalidOperation


SUMMARY_METRICS = [
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
]

DONE_REQUIRED_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr', 'SPICE']
OVERALL_CHANGE_PAIRS = [
    ('Bleu_4', 'Change_Bleu_4'),
    ('CIDEr', 'Change_CIDEr'),
    ('SPICE', 'Change_SPICE'),
]
NOCHANGE_COLUMNS = ['NoChange_Bleu_4', 'NoChange_ROUGE_L', 'NoChange_SPICE']


def parse_args():
    parser = argparse.ArgumentParser(description='Check paper-required summary CSV consistency.')
    parser.add_argument('--summary_csv', default=os.path.join('experiments', 'paper_required_experiments_summary.csv'))
    parser.add_argument('--output', default=os.path.join('experiments', 'summary_check_report.txt'))
    parser.add_argument('--baseline_csv', default=os.path.join('experiments', 'paper_required_experiments_summary.baseline.csv'))
    parser.add_argument('--strict', action='store_true', help='Exit non-zero when consistency issues are found.')
    return parser.parse_args()


def read_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError('Summary CSV does not exist: %s' % path)
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def row_name(row):
    return row.get('exp_name') or row.get('experiment') or ''


def is_blank(value):
    return value is None or str(value).strip() == ''


def normalized_value(value):
    text = '' if value is None else str(value).strip()
    if text == '':
        return ''
    try:
        return Decimal(text).normalize()
    except (InvalidOperation, ValueError):
        return text


def values_equal(left, right):
    return normalized_value(left) == normalized_value(right)


def line_for(index):
    return index + 2


def check_duplicate_exp_names(rows):
    seen = {}
    issues = []
    for index, row in enumerate(rows):
        name = row_name(row)
        if not name:
            continue
        seen.setdefault(name, []).append(line_for(index))
    for name, lines in sorted(seen.items()):
        if len(lines) > 1:
            issues.append('duplicate exp_name %s at CSV lines %s' % (name, ', '.join(str(item) for item in lines)))
    return issues


def check_done_missing_metrics(rows):
    issues = []
    for index, row in enumerate(rows):
        status = str(row.get('status', '')).strip().lower()
        if status != 'done':
            continue
        missing = [metric for metric in DONE_REQUIRED_METRICS if is_blank(row.get(metric))]
        if missing:
            issues.append('line %d exp_name=%s status=done but missing metrics: %s' % (
                line_for(index), row_name(row), ', '.join(missing)
            ))
    return issues


def check_overall_change_identical(rows):
    issues = []
    for index, row in enumerate(rows):
        nochange_nonempty = any(not is_blank(row.get(column)) for column in NOCHANGE_COLUMNS)
        if not nochange_nonempty:
            continue
        all_present = all(
            not is_blank(row.get(overall)) and not is_blank(row.get(change))
            for overall, change in OVERALL_CHANGE_PAIRS
        )
        if not all_present:
            continue
        if all(values_equal(row.get(overall), row.get(change)) for overall, change in OVERALL_CHANGE_PAIRS):
            issues.append(
                'line %d exp_name=%s overall metrics exactly match change-only metrics while no-change metrics are non-empty'
                % (line_for(index), row_name(row))
            )
    return issues


def baseline_by_exp_name(path):
    if not os.path.exists(path):
        return None
    rows, _ = read_rows(path)
    result = {}
    for row in rows:
        name = row_name(row)
        if name and name not in result:
            result[name] = row
    return result


def check_skipped_existing_changes(rows, baseline_csv):
    baseline = baseline_by_exp_name(baseline_csv)
    warnings = []
    issues = []
    skipped = [row for row in rows if str(row.get('status', '')).strip().lower() == 'skipped_existing']
    if not skipped:
        return issues, warnings
    if baseline is None:
        warnings.append(
            'skipped_existing metric-change check needs baseline snapshot: %s' % baseline_csv
        )
        return issues, warnings
    for row in skipped:
        name = row_name(row)
        old = baseline.get(name)
        if not old:
            warnings.append('skipped_existing exp_name=%s has no matching baseline row; metric-change check skipped' % name)
            continue
        changed = []
        for metric in SUMMARY_METRICS:
            if not values_equal(row.get(metric), old.get(metric)):
                changed.append('%s: %s -> %s' % (metric, old.get(metric, ''), row.get(metric, '')))
        if changed:
            issues.append('skipped_existing exp_name=%s has metric changes compared with baseline: %s' % (name, '; '.join(changed)))
    return issues, warnings


def write_report(path, summary_csv, rows, fieldnames, issues, warnings):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Summary consistency report\n')
        f.write('summary_csv: %s\n' % summary_csv)
        f.write('rows: %d\n' % len(rows))
        f.write('columns: %d\n' % len(fieldnames))
        f.write('issues: %d\n' % len(issues))
        f.write('warnings: %d\n\n' % len(warnings))
        if issues:
            f.write('[issues]\n')
            for item in issues:
                f.write('- %s\n' % item)
            f.write('\n')
        else:
            f.write('[issues]\n- none\n\n')
        if warnings:
            f.write('[warnings]\n')
            for item in warnings:
                f.write('- %s\n' % item)
        else:
            f.write('[warnings]\n- none\n')


def main():
    args = parse_args()
    rows, fieldnames = read_rows(args.summary_csv)
    issues = []
    warnings = []
    issues.extend(check_duplicate_exp_names(rows))
    issues.extend(check_overall_change_identical(rows))
    issues.extend(check_done_missing_metrics(rows))
    skipped_issues, skipped_warnings = check_skipped_existing_changes(rows, args.baseline_csv)
    issues.extend(skipped_issues)
    warnings.extend(skipped_warnings)
    write_report(args.output, args.summary_csv, rows, fieldnames, issues, warnings)
    print('Wrote summary consistency report: %s' % args.output)
    if args.strict and issues:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())