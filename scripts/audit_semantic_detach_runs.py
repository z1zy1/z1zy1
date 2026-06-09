import argparse
import csv
import json
import os
import re


DEFAULT_EXPERIMENTS = [
    'lmask005_lsem005_semantic_detach',
    'lmask005_lsem0075_semantic_detach',
]

EXPECTED_LAMBDAS = {
    'lmask005_lsem005_semantic_detach': {'lambda_mask': 0.05, 'lambda_semantic': 0.05},
    'lmask005_lsem0075_semantic_detach': {'lambda_mask': 0.05, 'lambda_semantic': 0.075},
    'lmask002_lsem0075_semantic_detach': {'lambda_mask': 0.02, 'lambda_semantic': 0.075},
}


def nested_get(data, path):
    cur = data
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


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


def parse_float_from_text(text, name):
    patterns = [
        re.compile(r'%s\s*:\s*([0-9.]+)' % re.escape(name)),
        re.compile(r'%s[\'"]?\s*,\s*[\'"]?([0-9.]+)' % re.escape(name)),
        re.compile(r'%s[\'"]?\s+[\'"]?([0-9.]+)' % re.escape(name)),
    ]
    values = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return values


def parse_exp_name_from_text(text):
    match = re.search(r'exp_name\s*:\s*([A-Za-z0-9_.-]+)', text)
    return match.group(1) if match else None


def read_best_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    return {
        'selected_checkpoint': row.get('selected_checkpoint') or row.get('checkpoint'),
        'snapshot_name': row.get('snapshot_name'),
        'source_file': row.get('source_file'),
    }


def parse_test_template(path):
    text = read_text(path)
    if not text:
        return {}
    snapshot_match = re.search(r'--snapshot\s+"?([0-9]+)"?', text)
    exp_match = re.search(r'exp_name\s+"?([A-Za-z0-9_.-]+)"?', text)
    lambdas = parse_float_from_text(text, 'train.lambda_semantic')
    return {
        'snapshot': snapshot_match.group(1) if snapshot_match else None,
        'exp_name': exp_match.group(1) if exp_match else None,
        'lambda_semantic': lambdas[-1] if lambdas else None,
        'path': path,
    }


def audit_one(exp_name, exp_root, results_root):
    expected = EXPECTED_LAMBDAS.get(exp_name, {})
    out_dir = os.path.join(exp_root, exp_name)
    result_dir = os.path.join(results_root, exp_name)
    cfg_path = os.path.join(out_dir, 'cfg.json')
    args_path = os.path.join(out_dir, 'args.txt')
    train_log_path = os.path.join(out_dir, 'train.log')
    logger_train_log_path = os.path.join(out_dir, 'train_log.txt')
    best_csv_path = os.path.join(result_dir, 'snapshot_selection', 'best_checkpoint.csv')
    test_template_paths = [
        os.path.join(result_dir, 'test_command_template.sh'),
        os.path.join(result_dir, 'test_command_template.txt'),
    ]

    cfg_data = read_json(cfg_path)
    args_text = read_text(args_path)
    train_text = read_text(train_log_path) + '\n' + read_text(logger_train_log_path)
    best = read_best_checkpoint(best_csv_path)
    test_template = {}
    for path in test_template_paths:
        test_template = parse_test_template(path)
        if test_template:
            break

    cfg_exp_name = nested_get(cfg_data, 'exp_name') if cfg_data else None
    cfg_lambda_mask = nested_get(cfg_data, 'train.lambda_mask') if cfg_data else None
    cfg_lambda_semantic = nested_get(cfg_data, 'train.lambda_semantic') if cfg_data else None

    train_lsem_values = parse_float_from_text(args_text + '\n' + train_text, 'lambda_semantic')
    train_lmask_values = parse_float_from_text(args_text + '\n' + train_text, 'lambda_mask')
    train_exp_name = parse_exp_name_from_text(args_text + '\n' + train_text)

    selected_ckpt = best['selected_checkpoint'] if best else None
    selected_ckpt_path = None
    if selected_ckpt:
        selected_ckpt_path = os.path.join(
            out_dir,
            'snapshots',
            '%s_checkpoint_%s.pt' % (exp_name, selected_ckpt),
        )

    checks = []
    checks.append(('output_dir_exists', os.path.isdir(out_dir), out_dir))
    checks.append(('result_dir_exists', os.path.isdir(result_dir), result_dir))
    checks.append(('cfg_exp_name_matches', cfg_exp_name in (None, exp_name), cfg_exp_name))
    checks.append(('args_or_log_exp_name_matches', train_exp_name in (None, exp_name), train_exp_name))
    if expected:
        checks.append(('cfg_lambda_mask_expected', cfg_lambda_mask in (None, expected['lambda_mask']), cfg_lambda_mask))
        checks.append(('cfg_lambda_semantic_expected', cfg_lambda_semantic in (None, expected['lambda_semantic']), cfg_lambda_semantic))
        if train_lmask_values:
            checks.append(('logged_lambda_mask_expected', expected['lambda_mask'] in train_lmask_values, train_lmask_values))
        else:
            checks.append(('logged_lambda_mask_present', False, train_lmask_values))
        if train_lsem_values:
            checks.append(('logged_lambda_semantic_expected', expected['lambda_semantic'] in train_lsem_values, train_lsem_values))
        else:
            checks.append(('logged_lambda_semantic_present', False, train_lsem_values))
    checks.append(('best_checkpoint_csv_exists', best is not None, best_csv_path))
    checks.append(('selected_checkpoint_file_exists', selected_ckpt_path is None or os.path.exists(selected_ckpt_path), selected_ckpt_path))
    checks.append(('test_template_exists', bool(test_template), test_template.get('path')))
    if test_template:
        checks.append(('test_template_exp_name_matches', test_template.get('exp_name') == exp_name, test_template.get('exp_name')))
        if selected_ckpt:
            checks.append(('test_template_uses_selected_ckpt', test_template.get('snapshot') == str(selected_ckpt), test_template.get('snapshot')))
        if expected:
            checks.append(('test_template_lambda_semantic_expected', test_template.get('lambda_semantic') == expected['lambda_semantic'], test_template.get('lambda_semantic')))

    return {
        'exp_name': exp_name,
        'out_dir': out_dir,
        'result_dir': result_dir,
        'cfg_lambda_mask': cfg_lambda_mask,
        'cfg_lambda_semantic': cfg_lambda_semantic,
        'logged_lambda_mask_values': train_lmask_values,
        'logged_lambda_semantic_values': train_lsem_values,
        'best': best,
        'selected_ckpt_path': selected_ckpt_path,
        'test_template': test_template,
        'checks': checks,
    }


def print_report(results):
    print('Semantic detach run audit')
    print('')
    selected_paths = []
    for result in results:
        print('[%s]' % result['exp_name'])
        print('output_dir: %s' % result['out_dir'])
        print('result_dir: %s' % result['result_dir'])
        print('cfg lambda_mask: %s' % result['cfg_lambda_mask'])
        print('cfg lambda_semantic: %s' % result['cfg_lambda_semantic'])
        print('logged lambda_mask values: %s' % result['logged_lambda_mask_values'])
        print('logged lambda_semantic values: %s' % result['logged_lambda_semantic_values'])
        print('best checkpoint: %s' % (result['best'] or None))
        print('selected checkpoint path: %s' % result['selected_ckpt_path'])
        if result['selected_ckpt_path']:
            selected_paths.append(os.path.abspath(result['selected_ckpt_path']))
        print('test template: %s' % (result['test_template'] or None))
        for name, ok, detail in result['checks']:
            print('  %-40s %s %s' % (name, 'OK' if ok else 'FAIL', detail))
        print('')

    if len(selected_paths) > 1:
        duplicates = sorted({path for path in selected_paths if selected_paths.count(path) > 1})
        print('selected checkpoint path duplicates: %s' % (duplicates or 'none'))

    failures = [
        (result['exp_name'], name, detail)
        for result in results
        for name, ok, detail in result['checks']
        if not ok
    ]
    if failures:
        print('Audit finished with %d warning/failure item(s).' % len(failures))
    else:
        print('Audit finished with no detected independence problems.')


def main():
    parser = argparse.ArgumentParser(description='Audit whether semantic detach experiments used independent configs/checkpoints.')
    parser.add_argument('--exp_root', default='./outputs')
    parser.add_argument('--results_root', default='./results')
    parser.add_argument('--experiments', nargs='*', default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args()

    results = [audit_one(exp, args.exp_root, args.results_root) for exp in args.experiments]
    print_report(results)


if __name__ == '__main__':
    main()
