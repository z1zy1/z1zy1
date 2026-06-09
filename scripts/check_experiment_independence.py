import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime


def sha256_file(path):
    if not path or not os.path.exists(path) or os.path.isdir(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_info(path):
    if not path or not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        'path': path,
        'size': stat.st_size,
        'mtime': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds'),
        'sha256': sha256_file(path),
    }


def read_text(path):
    if not os.path.exists(path):
        return ''
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'cp936'):
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, 'r', errors='replace') as f:
        return f.read()


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def nested_get(data, dotted):
    cur = data
    for key in dotted.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def parse_values(text, name):
    values = []
    patterns = [
        re.compile(r'%s\s*:\s*([A-Za-z0-9_.+-]+)' % re.escape(name)),
        re.compile(r'%s[\'"]?\s*,\s*[\'"]?([A-Za-z0-9_.+-]+)' % re.escape(name)),
        re.compile(r'%s[\'"]?\s+[\'"]?([A-Za-z0-9_.+-]+)' % re.escape(name)),
    ]
    for pattern in patterns:
        values.extend(match.group(1) for match in pattern.finditer(text))
    return values


def checkpoint_files(exp_dir):
    candidates = []
    for subdir in ('snapshots', 'checkpoints'):
        root = os.path.join(exp_dir, subdir)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.endswith(('.pt', '.pth', '.ckpt')):
                candidates.append(os.path.join(root, name))
    return candidates


def read_best_checkpoint(result_dir):
    paths = [
        os.path.join(result_dir, 'snapshot_selection', 'best_checkpoint.csv'),
        os.path.join(result_dir, 'snapshot_selection', 'selected_snapshot.csv'),
        os.path.join(result_dir, 'selected_snapshot.csv'),
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[0]
        selected = row.get('selected_checkpoint') or row.get('checkpoint') or row.get('snapshot')
        return {
            'csv_path': path,
            'selected_checkpoint': selected,
            'snapshot_name': row.get('snapshot_name'),
            'source_file': row.get('source_file'),
            'row': row,
        }
    return None


def infer_checkpoint_path(exp_dir, exp_name, best):
    if not best or not best.get('selected_checkpoint'):
        return None
    selected = str(best['selected_checkpoint'])
    explicit = best.get('snapshot_name')
    candidates = []
    if explicit:
        candidates.extend([
            os.path.join(exp_dir, 'snapshots', explicit),
            os.path.join(exp_dir, 'checkpoints', explicit),
            os.path.join(exp_dir, 'snapshots', explicit + '.pt'),
            os.path.join(exp_dir, 'checkpoints', explicit + '.pt'),
        ])
    candidates.extend([
        os.path.join(exp_dir, 'snapshots', '%s_checkpoint_%s.pt' % (exp_name, selected)),
        os.path.join(exp_dir, 'checkpoints', '%s_checkpoint_%s.pt' % (exp_name, selected)),
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def collect(exp_dir, result_dir):
    cfg = read_json(os.path.join(exp_dir, 'cfg.json'))
    exp_name = nested_get(cfg, 'exp_name') if cfg else os.path.basename(os.path.normpath(exp_dir))
    args_text = read_text(os.path.join(exp_dir, 'args.txt'))
    train_text = (
        read_text(os.path.join(exp_dir, 'train.log'))
        + '\n'
        + read_text(os.path.join(exp_dir, 'train_log.txt'))
        + '\n'
        + args_text
    )
    ckpts = checkpoint_files(exp_dir)
    best = read_best_checkpoint(result_dir)
    selected_path = infer_checkpoint_path(exp_dir, exp_name, best)
    eval_path = os.path.join(result_dir, 'eval_results.txt')
    test_path = os.path.join(result_dir, 'test_results.txt')
    if not os.path.exists(eval_path):
        eval_path = os.path.join(exp_dir, 'eval_sents', 'eval_results.txt')

    return {
        'exp_dir': exp_dir,
        'result_dir': result_dir,
        'exp_name': exp_name,
        'cfg_lambda_mask': nested_get(cfg, 'train.lambda_mask') if cfg else None,
        'cfg_lambda_semantic': nested_get(cfg, 'train.lambda_semantic') if cfg else None,
        'cfg_use_semantic_detach': nested_get(cfg, 'train.use_semantic_detach') if cfg else None,
        'log_lambda_mask': parse_values(train_text, 'lambda_mask'),
        'log_lambda_semantic': parse_values(train_text, 'lambda_semantic'),
        'log_use_semantic_detach': parse_values(train_text, 'use_semantic_detach'),
        'checkpoints': [file_info(path) for path in ckpts],
        'eval_results': file_info(eval_path),
        'best': best,
        'selected_checkpoint_path': selected_path,
        'selected_checkpoint': file_info(selected_path),
        'test_results': file_info(test_path),
    }


def same_nonempty(a, b, key):
    return bool(a and b and a.get(key) and a.get(key) == b.get(key))


def print_report(one, two):
    warnings = []
    for idx, item in enumerate((one, two), 1):
        print('Experiment %d' % idx)
        print('  exp_name: %s' % item['exp_name'])
        print('  output_dir: %s' % item['exp_dir'])
        print('  result_dir: %s' % item['result_dir'])
        print('  cfg lambda_mask: %s' % item['cfg_lambda_mask'])
        print('  cfg lambda_semantic: %s' % item['cfg_lambda_semantic'])
        print('  cfg use_semantic_detach: %s' % item['cfg_use_semantic_detach'])
        print('  train.log lambda_mask: %s' % item['log_lambda_mask'])
        print('  train.log lambda_semantic: %s' % item['log_lambda_semantic'])
        print('  train.log use_semantic_detach: %s' % item['log_use_semantic_detach'])
        print('  checkpoint files:')
        if item['checkpoints']:
            for ckpt in item['checkpoints']:
                print('    {path} size={size} mtime={mtime} sha256={sha256}'.format(**ckpt))
        else:
            print('    none')
        print('  eval_results: %s' % item['eval_results'])
        print('  selected_snapshot: %s' % item['best'])
        print('  selected_checkpoint_path: %s' % item['selected_checkpoint_path'])
        print('  selected_checkpoint_file: %s' % item['selected_checkpoint'])
        print('  test_results: %s' % item['test_results'])
        print('')

    if os.path.abspath(one['exp_dir']) == os.path.abspath(two['exp_dir']):
        warnings.append('Both experiments use the same output_dir.')
    if os.path.abspath(one['result_dir']) == os.path.abspath(two['result_dir']):
        warnings.append('Both experiments use the same result_dir.')
    if one['exp_name'] == two['exp_name']:
        warnings.append('Both experiments have the same exp_name.')
    for label, item in (('exp1', one), ('exp2', two)):
        if not item['checkpoints']:
            warnings.append('%s has no checkpoint files under snapshots/ or checkpoints/.' % label)
        if item['eval_results'] is None:
            warnings.append('%s has no eval_results.txt.' % label)
        if item['best'] is None:
            warnings.append('%s has no selected snapshot CSV.' % label)
        if item['test_results'] is None:
            warnings.append('%s has no test_results.txt.' % label)
    if one['selected_checkpoint_path'] and two['selected_checkpoint_path']:
        if os.path.abspath(one['selected_checkpoint_path']) == os.path.abspath(two['selected_checkpoint_path']):
            warnings.append('Both experiments selected the same checkpoint path.')
    if same_nonempty(one['selected_checkpoint'] or {}, two['selected_checkpoint'] or {}, 'sha256'):
        warnings.append('Selected checkpoint files have the same hash.')
    if same_nonempty(one['eval_results'] or {}, two['eval_results'] or {}, 'sha256'):
        warnings.append('eval_results.txt files have the same hash.')
    if same_nonempty(one['test_results'] or {}, two['test_results'] or {}, 'sha256'):
        warnings.append('test_results.txt files have the same hash.')

    print('Warnings')
    if warnings:
        for warning in warnings:
            print('  WARNING: %s' % warning)
    else:
        print('  none')


def main():
    parser = argparse.ArgumentParser(description='Check whether two experiment runs are independent.')
    parser.add_argument('--exp1', required=True)
    parser.add_argument('--exp2', required=True)
    parser.add_argument('--result1', required=True)
    parser.add_argument('--result2', required=True)
    args = parser.parse_args()

    one = collect(args.exp1, args.result1)
    two = collect(args.exp2, args.result2)
    print_report(one, two)


if __name__ == '__main__':
    main()
