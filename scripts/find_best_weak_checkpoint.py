import argparse
import json
import os


DEFAULT_EXPERIMENTS = [
    'levir_mci_weak_pd08_lm003_ls001_noreweight',
    'levir_mci_weak_pd08_lm001_ls001_noreweight',
    'levir_mci_weak_pd05_lm003_ls0005_noreweight',
]
METRICS = ['Bleu_4', 'METEOR', 'CIDEr', 'SPICE']


def parse_args():
    parser = argparse.ArgumentParser(description='Pick the best weak LEVIR-MCI checkpoint for caption-only finetuning.')
    parser.add_argument('--experiments_root', default='experiments')
    parser.add_argument('--exp_names', nargs='*', default=DEFAULT_EXPERIMENTS)
    parser.add_argument('--output_json', default=os.path.join('experiments', 'levir_mci_caption_finetune_from_weak_best_init.json'))
    parser.add_argument('--output_txt', default=os.path.join('experiments', 'levir_mci_caption_finetune_from_weak_best_init.txt'))
    return parser.parse_args()


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def metric_value(metrics, key):
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def balanced_score(metrics):
    return sum(0.25 * metric_value(metrics, key) for key in METRICS)


def main():
    args = parse_args()
    candidates = []
    for exp_name in args.exp_names:
        exp_dir = os.path.join(args.experiments_root, exp_name)
        path = os.path.join(exp_dir, 'best_checkpoint.json')
        if not os.path.exists(path):
            candidates.append({'exp_name': exp_name, 'error': 'missing best_checkpoint.json', 'path': path})
            continue
        payload = load_json(path)
        metrics = payload.get('selected_val_metrics') or payload.get('selected_metrics') or {}
        checkpoint = payload.get('selected_checkpoint') or payload.get('selected_checkpoint_path')
        if not checkpoint:
            candidates.append({'exp_name': exp_name, 'error': 'missing selected checkpoint', 'path': path})
            continue
        candidates.append({
            'exp_name': exp_name,
            'best_checkpoint_json': path,
            'selected_checkpoint': checkpoint,
            'selected_val_metrics': metrics,
            'balanced_score': balanced_score(metrics),
        })
    valid = [item for item in candidates if 'error' not in item]
    if not valid:
        raise RuntimeError('No valid weak checkpoints found: %s' % candidates)
    best = sorted(valid, key=lambda item: (item['balanced_score'], metric_value(item['selected_val_metrics'], 'SPICE'), metric_value(item['selected_val_metrics'], 'CIDEr')), reverse=True)[0]
    output = {'selected': best, 'candidates': candidates}
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    with open(args.output_txt, 'w', encoding='utf-8') as f:
        f.write(best['selected_checkpoint'] + '\n')
        f.write(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
