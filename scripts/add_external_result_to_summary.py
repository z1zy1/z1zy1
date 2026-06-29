import argparse
import os
import subprocess
import sys


PENDING_NOTE = 'MModalCC external result pending; not included in main comparison.'
METRIC_KEYS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']


def parse_args():
    parser = argparse.ArgumentParser(description='Add a manually sourced external result to the paper summary CSV.')
    parser.add_argument('--summary_csv', default=os.path.join('experiments', 'paper_required_experiments_summary.csv'))
    parser.add_argument('--dataset', default='second_cc')
    parser.add_argument('--exp_name', default='second_cc_mmodalcc_external_comparison')
    parser.add_argument('--method_group', default='external_mmodalcc')
    parser.add_argument('--source', default='')
    parser.add_argument('--notes', default='')
    for key in METRIC_KEYS:
        parser.add_argument('--' + key, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = {key: getattr(args, key) for key in METRIC_KEYS if getattr(args, key) is not None}
    status = 'done' if metrics and args.source else 'pending_external'
    notes = args.notes or ('' if metrics and args.source else PENDING_NOTE)
    cmd = [
        sys.executable,
        os.path.join('scripts', 'update_paper_required_summary.py'),
        '--summary_csv', args.summary_csv,
        '--dataset', args.dataset,
        '--exp_name', args.exp_name,
        '--method_group', args.method_group,
        '--status', status,
        '--notes', notes,
        '--selection_strategy', 'external_reported' if metrics else '',
    ]
    if args.source:
        cmd.extend(['--set', 'checkpoint=' + args.source])
    for key, value in metrics.items():
        cmd.extend(['--set', '%s=%s' % (key, value)])
    subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
