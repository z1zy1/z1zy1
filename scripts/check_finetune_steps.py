import argparse
import csv
import glob
import hashlib
import json
import os
import re
from collections import defaultdict


DEFAULT_EXPERIMENTS = [
    'levir_mci_short_caption_ft_500_lr01',
    'levir_mci_short_caption_ft_1000_lr01',
]
RELEVANT_KEYS = [
    'exp_name',
    'data.dataset',
    'train.init_checkpoint',
    'train.finetune_steps',
    'train.max_iter',
    'train.total_steps',
    'train.max_epochs',
    'train.optim.lr',
    'train.snapshot_interval',
    'train.save_interval',
    'train.eval_interval',
    'train.selection_strategy',
    'model.enable_aux_mask',
    'train.use_aux_mask',
    'train.use_semantic_aux',
    'train.lambda_mask',
    'train.lambda_semantic',
    'train.use_feature_reweight',
    'train.use_semantic_hard_gate',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Audit fine-tune step configs, logs, checkpoints, and selection results.')
    parser.add_argument('exp_dirs', nargs='*', help='Experiment directories or experiment names under --experiments_root.')
    parser.add_argument('--experiments_root', default='experiments')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def stable_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def read_config_hash(exp_dir, resolved):
    path = os.path.join(exp_dir, 'config_hash.txt')
    values = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    values[key] = value
    if not values and resolved:
        values['computed_resolved_config_hash'] = stable_hash(resolved)
    return values


def nested_get(payload, dotted, default=''):
    cur = payload
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def flatten(payload, prefix=''):
    result = {}
    if not isinstance(payload, dict):
        return result
    for key, value in payload.items():
        full = '%s.%s' % (prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten(value, full))
        else:
            result[full] = value
    return result


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def checkpoint_number(text):
    matches = re.findall(r'(\d+)', os.path.basename(str(text or '')))
    return int(matches[-1]) if matches else None


def sorted_checkpoints(exp_dir):
    patterns = []
    for folder in ('snapshots', 'checkpoints', ''):
        patterns.extend([
            os.path.join(exp_dir, folder, '*.pt'),
            os.path.join(exp_dir, folder, '*.pth'),
        ])
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted({os.path.normpath(path) for path in paths if os.path.isfile(path)})
    return sorted(paths, key=lambda item: (checkpoint_number(item) is None, checkpoint_number(item) or -1, item))


def selected_checkpoint(exp_dir):
    payload = load_json(os.path.join(exp_dir, 'best_checkpoint.json')) or {}
    checkpoint = payload.get('selected_checkpoint') or payload.get('selected_checkpoint_path') or ''
    if not checkpoint:
        txt = os.path.join(exp_dir, 'best_checkpoint.txt')
        if os.path.exists(txt):
            with open(txt, encoding='utf-8-sig') as f:
                checkpoint = f.readline().strip()
    return checkpoint, payload


def val_metric_rows(exp_dir):
    for name in ('val_metrics.csv', 'eval_snapshots.csv'):
        path = os.path.join(exp_dir, name)
        rows = read_csv_rows(path)
        if rows:
            return path, rows
    for name in ('val_metrics.json', 'eval_snapshots.json'):
        path = os.path.join(exp_dir, name)
        payload = load_json(path)
        if isinstance(payload, list):
            return path, payload
        if isinstance(payload, dict):
            rows = payload.get('checkpoints') or payload.get('rows') or []
            if rows:
                return path, rows
    return '', []


def row_checkpoint(row):
    if not isinstance(row, dict):
        return ''
    return row.get('checkpoint_path') or row.get('snapshot_path') or row.get('checkpoint') or ''


def row_step(row):
    if not isinstance(row, dict):
        return None
    for key in ('step', 'iter', 'epoch'):
        value = row.get(key)
        if value not in (None, ''):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                pass
    return checkpoint_number(row_checkpoint(row))


def last_train_jsonl_step(path):
    last = None
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ('iter', 'step', 'global_step'):
                if key in row:
                    try:
                        last = max(last or 0, int(float(row[key])))
                    except (TypeError, ValueError):
                        pass
    return last


def last_train_log_step(path):
    if not os.path.exists(path):
        return None
    last = None
    patterns = [
        re.compile(r'iter\s*[:= ]\s*(\d+)', re.IGNORECASE),
        re.compile(r'\bt\s*[:= ]\s*(\d+)', re.IGNORECASE),
        re.compile(r'Running eval at iter\s+(\d+)', re.IGNORECASE),
        re.compile(r'checkpoint[_-](\d+)\.(?:pt|pth)', re.IGNORECASE),
    ]
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            for pattern in patterns:
                for match in pattern.findall(line):
                    try:
                        last = max(last or 0, int(match))
                    except ValueError:
                        pass
    return last


def inspect_experiment(exp_dir):
    resolved = load_json(os.path.join(exp_dir, 'resolved_config.json')) or {}
    hashes = read_config_hash(exp_dir, resolved)
    checkpoints = sorted_checkpoints(exp_dir)
    selected, selected_payload = selected_checkpoint(exp_dir)
    metrics_path, metrics_rows = val_metric_rows(exp_dir)
    val_checkpoints = [row_checkpoint(row) for row in metrics_rows if row_checkpoint(row)]
    val_steps = [row_step(row) for row in metrics_rows if row_step(row) is not None]
    train_jsonl_step = last_train_jsonl_step(os.path.join(exp_dir, 'train_log.jsonl'))
    train_text_step = last_train_log_step(os.path.join(exp_dir, 'train.log'))
    checkpoint_steps = [checkpoint_number(path) for path in checkpoints if checkpoint_number(path) is not None]
    actual_candidates = [item for item in [train_jsonl_step, train_text_step] + val_steps + checkpoint_steps if item is not None]
    return {
        'exp_dir': os.path.normpath(exp_dir),
        'exists': os.path.isdir(exp_dir),
        'resolved_config': resolved,
        'flat_config': flatten(resolved),
        'config_hash': hashes,
        'init_checkpoint': nested_get(resolved, 'train.init_checkpoint'),
        'finetune_steps': nested_get(resolved, 'train.finetune_steps'),
        'max_iter': nested_get(resolved, 'train.max_iter'),
        'max_epochs': nested_get(resolved, 'train.max_epochs'),
        'total_steps': nested_get(resolved, 'train.total_steps'),
        'snapshot_interval': nested_get(resolved, 'train.snapshot_interval'),
        'save_interval': nested_get(resolved, 'train.save_interval'),
        'eval_interval': nested_get(resolved, 'train.eval_interval'),
        'train_log_last_step': max([item for item in (train_jsonl_step, train_text_step) if item is not None], default=None),
        'actual_step_count': max(actual_candidates) if actual_candidates else None,
        'checkpoints': checkpoints,
        'checkpoint_steps': checkpoint_steps,
        'selected_checkpoint': selected,
        'selected_payload': selected_payload,
        'val_metrics_path': metrics_path,
        'val_metric_rows': metrics_rows,
        'val_checkpoints': val_checkpoints,
        'val_steps': val_steps,
    }


def resolve_exp_dirs(args):
    items = args.exp_dirs or DEFAULT_EXPERIMENTS
    result = []
    for item in items:
        has_path_separator = os.sep in item or (os.altsep is not None and os.altsep in item)
        if os.path.isabs(item) or has_path_separator or os.path.exists(item):
            result.append(item)
        else:
            result.append(os.path.join(args.experiments_root, item))
    return result


def format_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def diff_configs(reports):
    if len(reports) < 2:
        return []
    base = reports[0]
    base_flat = base['flat_config']
    keys = sorted(set().union(*(set(report['flat_config'].keys()) for report in reports)))
    changed = []
    for key in keys:
        values = [report['flat_config'].get(key, '<missing>') for report in reports]
        if any(value != values[0] for value in values[1:]):
            changed.append((key, values))
    priority = [item for item in changed if item[0] in RELEVANT_KEYS]
    rest = [item for item in changed if item[0] not in RELEVANT_KEYS]
    return priority + rest


def selected_duplicate_groups(reports):
    groups = defaultdict(list)
    for report in reports:
        selected = report.get('selected_checkpoint') or ''
        if selected:
            key = os.path.basename(os.path.normpath(selected))
            groups[key].append(report['exp_dir'])
    return {key: value for key, value in groups.items() if len(value) > 1}


def diagnose(reports):
    lines = []
    duplicates = selected_duplicate_groups(reports)
    if duplicates:
        for checkpoint, dirs in sorted(duplicates.items()):
            lines.append('selected_same_checkpoint: %s selected by %s' % (checkpoint, ', '.join(dirs)))
    for report in reports:
        name = os.path.basename(report['exp_dir'])
        finetune_steps = report.get('finetune_steps')
        max_iter = report.get('max_iter')
        checkpoint_steps = sorted(set(report.get('checkpoint_steps') or []))
        val_steps = sorted(set(report.get('val_steps') or []))
        selected_step = checkpoint_number(report.get('selected_checkpoint'))
        if finetune_steps in ('', None, 0, '0') and max_iter not in ('', None):
            lines.append('%s: resolved config has no explicit train.finetune_steps; training length is controlled by train.max_iter=%s.' % (name, max_iter))
        if checkpoint_steps:
            lines.append('%s: saved checkpoint steps=%s.' % (name, ','.join(str(item) for item in checkpoint_steps)))
        else:
            lines.append('%s: no saved checkpoint files found in this workspace.' % name)
        if val_steps:
            lines.append('%s: validation metric steps=%s.' % (name, ','.join(str(item) for item in val_steps)))
        else:
            lines.append('%s: no val_metrics/eval_snapshots rows found in this workspace.' % name)
        if selected_step is not None and val_steps and selected_step in val_steps and selected_step != max(val_steps):
            lines.append('%s: selected an earlier evaluated checkpoint (%s) even though later val checkpoints exist; this indicates checkpoint selection, not necessarily an ignored step limit.' % (name, selected_step))
        if max_iter not in ('', None) and report.get('actual_step_count') is not None:
            try:
                if int(max_iter) != int(report['actual_step_count']):
                    lines.append('%s: actual max observed step %s differs from train.max_iter %s; inspect interruption/log completeness.' % (name, report['actual_step_count'], max_iter))
            except (TypeError, ValueError):
                pass
    return lines


def write_report(path, reports):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    diff = diff_configs(reports)
    duplicates = selected_duplicate_groups(reports)
    lines = []
    lines.append('LEVIR-MCI fine-tune step audit')
    lines.append('experiments: %d' % len(reports))
    lines.append('')
    lines.append('[per experiment]')
    for report in reports:
        lines.append('')
        lines.append('- exp_dir: %s' % report['exp_dir'])
        lines.append('  exists: %s' % report['exists'])
        lines.append('  config_hash: %s' % format_value(report['config_hash']))
        lines.append('  init_checkpoint: %s' % report['init_checkpoint'])
        lines.append('  finetune_steps: %s' % report['finetune_steps'])
        lines.append('  max_iter: %s' % report['max_iter'])
        lines.append('  total_steps: %s' % report['total_steps'])
        lines.append('  max_epochs: %s' % report['max_epochs'])
        lines.append('  save_interval: %s' % report['save_interval'])
        lines.append('  eval_interval: %s' % report['eval_interval'])
        lines.append('  snapshot_interval: %s' % report['snapshot_interval'])
        lines.append('  train_log_last_step: %s' % report['train_log_last_step'])
        lines.append('  actual_step_count: %s' % report['actual_step_count'])
        lines.append('  checkpoint_count: %d' % len(report['checkpoints']))
        lines.append('  checkpoints: %s' % (', '.join(os.path.basename(path) for path in report['checkpoints']) or '<none>'))
        lines.append('  selected_best_checkpoint: %s' % (report['selected_checkpoint'] or '<none>'))
        lines.append('  val_metrics_file: %s' % (report['val_metrics_path'] or '<none>'))
        lines.append('  val_metrics_checkpoints: %s' % (', '.join(os.path.basename(path) for path in report['val_checkpoints']) or '<none>'))
    lines.append('')
    lines.append('[same selected checkpoint]')
    if duplicates:
        for checkpoint, dirs in sorted(duplicates.items()):
            lines.append('- %s: %s' % (checkpoint, ', '.join(dirs)))
    else:
        lines.append('- none')
    lines.append('')
    lines.append('[resolved config diff]')
    if diff:
        names = [os.path.basename(report['exp_dir']) for report in reports]
        lines.append('columns: key | %s' % ' | '.join(names))
        for key, values in diff[:200]:
            lines.append('- %s | %s' % (key, ' | '.join(format_value(value) for value in values)))
        if len(diff) > 200:
            lines.append('- ... %d additional differing keys omitted' % (len(diff) - 200))
    else:
        lines.append('- no resolved_config differences found, or configs are missing')
    lines.append('')
    lines.append('[diagnosis]')
    diagnosis = diagnose(reports)
    if diagnosis:
        for item in diagnosis:
            lines.append('- %s' % item)
    else:
        lines.append('- no diagnosis available')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return lines


def main():
    args = parse_args()
    exp_dirs = resolve_exp_dirs(args)
    reports = [inspect_experiment(exp_dir) for exp_dir in exp_dirs]
    output = args.output or os.path.join(args.experiments_root, 'finetune_check', 'levir_mci_finetune_steps_report.txt')
    lines = write_report(output, reports)
    print('Wrote fine-tune audit report: %s' % output)
    for line in lines[-min(20, len(lines)):]:
        print(line)


if __name__ == '__main__':
    main()