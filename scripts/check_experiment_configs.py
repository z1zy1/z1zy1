import argparse
import difflib
import glob
import hashlib
import json
import os
import pprint


KEY_SWITCHES = [
    'data.dataset',
    'model.enable_aux_mask',
    'train.use_semantic_aux',
    'train.use_aux_semantic',
    'train.use_semantic_cross_attention',
    'train.use_semantic_hard_gate',
    'train.use_semantic_partial_detach',
    'train.use_partial_detach',
    'train.semantic_detach_ratio',
    'train.use_feature_reweight',
    'train.lambda_mask',
    'train.lambda_semantic',
    'train.aux_warmup_start_ratio',
    'train.aux_warmup_end_ratio',
    'train.selection_strategy',
    'model.semantic_input_mode',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Compare experiment configs, selected checkpoints, and predictions.')
    parser.add_argument('experiments', nargs='*', default=[
        'levir_mci_card_mask_semantic_pd05',
        'levir_mci_card_mask_semantic_pd05_noreweight',
        'levir_mci_ours_weak_coupled_final',
    ])
    parser.add_argument('--experiments_root', default='experiments')
    parser.add_argument('--output', default=os.path.join('experiments', 'config_check', 'levir_mci_pd05_vs_noreweight_report.txt'))
    return parser.parse_args()


def resolve_exp(root, item):
    if os.path.isdir(item):
        return os.path.normpath(item)
    return os.path.normpath(os.path.join(root, item))


def file_sha256(path):
    if not path or not os.path.exists(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        return {'_error': str(exc)}


def read_config_hash(exp_dir):
    path = os.path.join(exp_dir, 'config_hash.txt')
    if not os.path.exists(path):
        return {}
    result = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                result[key] = value
    return result


def nested_get(payload, dotted):
    current = payload
    for part in dotted.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def find_prediction_file(exp_dir):
    patterns = [
        os.path.join(exp_dir, 'test_output', 'captions', '*', 'sc_results.json'),
        os.path.join(exp_dir, 'test_output', 'captions', 'sc_results.json'),
        os.path.join(exp_dir, 'eval_sents', '*', 'sc_results.json'),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        return ''
    return max(matches, key=lambda item: os.path.getmtime(item))


def find_checkpoint(best_payload):
    if not isinstance(best_payload, dict):
        return ''
    return best_payload.get('selected_checkpoint') or best_payload.get('selected_checkpoint_path') or best_payload.get('snapshot_path') or ''


def collect(exp_dir):
    resolved = load_json(os.path.join(exp_dir, 'resolved_config.json')) or {}
    best = load_json(os.path.join(exp_dir, 'best_checkpoint.json')) or {}
    prediction = find_prediction_file(exp_dir)
    checkpoint = find_checkpoint(best)
    return {
        'exp_dir': exp_dir,
        'exists': os.path.isdir(exp_dir),
        'resolved_config_path': os.path.join(exp_dir, 'resolved_config.json'),
        'resolved_config_exists': os.path.exists(os.path.join(exp_dir, 'resolved_config.json')),
        'config_hash': read_config_hash(exp_dir),
        'key_switches': {key: nested_get(resolved, key) for key in KEY_SWITCHES},
        'best_checkpoint_json_exists': os.path.exists(os.path.join(exp_dir, 'best_checkpoint.json')),
        'checkpoint': checkpoint,
        'checkpoint_sha256': file_sha256(checkpoint),
        'prediction_file': prediction,
        'prediction_sha256': file_sha256(prediction),
        'resolved_config': resolved,
    }


def comparable_config_text(info):
    payload = info.get('resolved_config') or {}
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop('exp_name', None)
        payload.pop('exp_dir', None)
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).splitlines()


def write_report(infos, output):
    lines = []
    lines.append('# LEVIR-MCI pd05 / noreweight / final config check')
    lines.append('')
    for info in infos:
        lines.append('## %s' % os.path.basename(info['exp_dir']))
        lines.append('exists: %s' % info['exists'])
        lines.append('resolved_config_exists: %s' % info['resolved_config_exists'])
        lines.append('config_hash: %s' % info['config_hash'])
        lines.append('key_switches:')
        lines.append(pprint.pformat(info['key_switches'], width=120))
        lines.append('best_checkpoint_json_exists: %s' % info['best_checkpoint_json_exists'])
        lines.append('checkpoint: %s' % info['checkpoint'])
        lines.append('checkpoint_sha256: %s' % info['checkpoint_sha256'])
        lines.append('prediction_file: %s' % info['prediction_file'])
        lines.append('prediction_sha256: %s' % info['prediction_sha256'])
        lines.append('')
    lines.append('## Pairwise conclusions')
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a = infos[i]
            b = infos[j]
            name = '%s vs %s' % (os.path.basename(a['exp_dir']), os.path.basename(b['exp_dir']))
            lines.append('### %s' % name)
            lines.append('same_full_config_hash: %s' % (a['config_hash'].get('full_config_hash') and a['config_hash'].get('full_config_hash') == b['config_hash'].get('full_config_hash')))
            lines.append('same_comparable_config_hash: %s' % (a['config_hash'].get('comparable_config_hash') and a['config_hash'].get('comparable_config_hash') == b['config_hash'].get('comparable_config_hash')))
            lines.append('same_checkpoint_path: %s' % (a['checkpoint'] and os.path.normcase(os.path.abspath(a['checkpoint'])) == os.path.normcase(os.path.abspath(b['checkpoint']))))
            lines.append('same_checkpoint_hash: %s' % (a['checkpoint_sha256'] and a['checkpoint_sha256'] == b['checkpoint_sha256']))
            lines.append('same_prediction_hash: %s' % (a['prediction_sha256'] and a['prediction_sha256'] == b['prediction_sha256']))
            changed_switches = {
                key: (a['key_switches'].get(key), b['key_switches'].get(key))
                for key in KEY_SWITCHES
                if a['key_switches'].get(key) != b['key_switches'].get(key)
            }
            lines.append('different_key_switches: %s' % pprint.pformat(changed_switches, width=120))
            if a['resolved_config_exists'] and b['resolved_config_exists']:
                diff = list(difflib.unified_diff(
                    comparable_config_text(a),
                    comparable_config_text(b),
                    fromfile=os.path.basename(a['exp_dir']),
                    tofile=os.path.basename(b['exp_dir']),
                    lineterm='',
                ))
                lines.append('resolved_config_diff:')
                lines.extend(diff[:400] if diff else ['<no diff after removing exp_name/exp_dir>'])
                if len(diff) > 400:
                    lines.append('<diff truncated at 400 lines>')
            else:
                lines.append('resolved_config_diff: unavailable because at least one resolved_config.json is missing')
            lines.append('')
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(line) for line in lines) + '\n')
    print(output)


def main():
    args = parse_args()
    infos = [collect(resolve_exp(args.experiments_root, item)) for item in args.experiments]
    write_report(infos, args.output)


if __name__ == '__main__':
    main()
