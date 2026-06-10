import argparse
import csv
import json
import os


FIELDS = [
    'exp_name',
    'best_snapshot',
    'Bleu_4',
    'METEOR',
    'ROUGE_L',
    'CIDEr',
    'SPICE',
    'balanced_score',
    'all_above_eval_baseline',
    'test_Bleu_4',
    'test_METEOR',
    'test_ROUGE_L',
    'test_CIDEr',
    'test_SPICE',
    'all_above_test_baseline',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize SGC-CARD ablation validation/test results.')
    parser.add_argument('--exp_dir', default='./experiments')
    parser.add_argument('--exp_names', nargs='+', required=True)
    parser.add_argument('--output', default='./experiments/sgc_card_ablation_summary.csv')
    return parser.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    rows = []
    for exp_name in args.exp_names:
        exp_path = os.path.join(args.exp_dir, exp_name)
        best = load_json(os.path.join(exp_path, 'best_snapshot.json')) or {}
        test = load_json(os.path.join(exp_path, 'test_best_result.json')) or {}
        val_metrics = best.get('metrics', {})
        test_metrics = test.get('metrics', {})
        rows.append({
            'exp_name': exp_name,
            'best_snapshot': best.get('best_snapshot', ''),
            'Bleu_4': val_metrics.get('Bleu_4', ''),
            'METEOR': val_metrics.get('METEOR', ''),
            'ROUGE_L': val_metrics.get('ROUGE_L', ''),
            'CIDEr': val_metrics.get('CIDEr', ''),
            'SPICE': val_metrics.get('SPICE', ''),
            'balanced_score': best.get('balanced_score', ''),
            'all_above_eval_baseline': best.get('all_above_baseline', ''),
            'test_Bleu_4': test_metrics.get('Bleu_4', ''),
            'test_METEOR': test_metrics.get('METEOR', ''),
            'test_ROUGE_L': test_metrics.get('ROUGE_L', ''),
            'test_CIDEr': test_metrics.get('CIDEr', ''),
            'test_SPICE': test_metrics.get('SPICE', ''),
            'all_above_test_baseline': test.get('ALL_TEST_METRICS_ABOVE_BASELINE', ''),
        })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote %s' % args.output)


if __name__ == '__main__':
    main()
