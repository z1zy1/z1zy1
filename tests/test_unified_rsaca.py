import os
import subprocess
import csv
import json
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_unified_runner_dry_run_contains_all_three_datasets():
    command = [
        'bash', 'scripts/run_unified_rsaca_experiments.sh',
        '--stage', 'train', '--dry-run',
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert 'levir_cc' in result.stdout
    assert 'levir_mci' in result.stdout
    assert 'second_cc' in result.stdout
    assert result.stdout.count('DRY RUN:') == 9


def test_identity_residual_candidate_is_explicit_in_dry_run():
    env = os.environ.copy()
    env['RUN_ROOT'] = './experiments/unified_rsaca_identity_ablation'
    env['SEMANTIC_FUSION_NORM_MODE'] = 'context_pre_norm'
    result = subprocess.run(
        [
            'bash', 'scripts/run_unified_rsaca_experiments.sh',
            '--stage', 'train', '--dataset', 'levir_cc', '--seed', '1111', '--dry-run',
        ],
        cwd=ROOT, check=True, capture_output=True, text=True, env=env,
    )
    assert 'norm_mode=context_pre_norm' in result.stdout
    assert 'unified_rsaca_identity_ablation' in result.stdout


def test_unified_summary_script_is_executable():
    result = subprocess.run(
        ['python', 'scripts/summarize_unified_rsaca.py', '--help'],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert '--run_root' in result.stdout


def test_reliability_sparse_v1_runner_dry_run_is_isolated_and_enabled():
    result = subprocess.run(
        [
            'bash', 'scripts/run_reliability_sparse_rsaca_v1.sh',
            '--stage', 'train', '--dataset', 'levir_cc', '--seed', '1111', '--dry-run',
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert 'reliability_sparse_rsaca_v1' in result.stdout
    assert 'SEMANTIC_FUSION_SPARSE_CHANGE_TOKENS=1' in result.stdout
    assert 'SEMANTIC_FUSION_RELIABILITY_GATE=1' in result.stdout


def test_reliability_sparse_whole_adapter_runner_dry_run_is_enabled():
    result = subprocess.run(
        [
            'bash', 'scripts/run_reliability_sparse_rsaca_v1_whole_gate.sh',
            '--stage', 'train', '--dataset', 'second_cc', '--seed', '1111', '--dry-run',
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert 'reliability_sparse_rsaca_v1_whole_gate' in result.stdout
    assert 'SEMANTIC_FUSION_GLOBAL_TOKEN=1' in result.stdout
    assert 'SEMANTIC_FUSION_GATE_WHOLE_ADAPTER=1' in result.stdout


def test_reliability_sparse_prenorm_changed_global_runner_dry_run_is_enabled():
    result = subprocess.run(
        [
            'bash', 'scripts/run_reliability_sparse_rsaca_v1_prenorm_changed_global.sh',
            '--stage', 'train', '--dataset', 'levir_cc', '--seed', '1111', '--dry-run',
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert 'reliability_sparse_rsaca_v1_prenorm_changed_global' in result.stdout
    assert 'norm=context_pre_norm' in result.stdout
    assert 'global_token=changed_mean' in result.stdout
    assert 'gamma_max=0.1' in result.stdout
    assert 'gate_bias=-2.5' in result.stdout


def test_paper_selector_accepts_uppercase_validation_flag():
    with tempfile.TemporaryDirectory() as root:
        snapshots = os.path.join(root, 'snapshots')
        os.makedirs(snapshots)
        checkpoint = os.path.join(snapshots, 'candidate_checkpoint_100.pt')
        open(checkpoint, 'w').close()
        csv_path = os.path.join(root, 'val_metrics.csv')
        fields = [
            'iter', 'snapshot_path', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
            'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'ALL_ABOVE_BASELINE',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                'iter': 100, 'snapshot_path': checkpoint,
                'Bleu_1': 0.8, 'Bleu_2': 0.7, 'Bleu_3': 0.6, 'Bleu_4': 0.5,
                'METEOR': 0.4, 'ROUGE_L': 0.7, 'CIDEr': 1.4, 'SPICE': 0.3,
                'ALL_ABOVE_BASELINE': '1',
            })
        output = os.path.join(root, 'best_snapshot_for_paper.json')
        subprocess.run(
            [
                'python', 'scripts/select_best_snapshot_for_paper.py',
                '--exp_dir', root, '--csv', csv_path,
                '--output_json', output,
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        with open(output, encoding='utf-8') as handle:
            payload = json.load(handle)
        assert payload['best']['all_above_baseline'] is True
