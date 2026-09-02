import os
import subprocess
import csv
import json
import tempfile

import torch
from torch import nn

from models.CARD import SemanticCrossAttentionFusion


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_unified_runner_dry_run_contains_all_three_datasets():
    with tempfile.TemporaryDirectory() as root:
        env = os.environ.copy()
        env['RUN_ROOT'] = root
        command = [
            'bash', 'scripts/run_unified_rsaca_experiments.sh',
            '--stage', 'train', '--dry-run',
        ]
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, env=env)
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


def test_paired_card_rsaca_runner_dry_run_contains_both_arms():
    with tempfile.TemporaryDirectory() as root:
        env = os.environ.copy()
        env['PAIR_ROOT'] = root
        env['REQUIRE_CUDA'] = '0'
        result = subprocess.run(
            [
                'bash', 'scripts/run_paired_card_rsaca_matrix.sh',
                '--stage', 'train', '--dataset', 'levir_cc', '--seed', '1111', '--dry-run',
            ],
            cwd=ROOT, check=True, capture_output=True, text=True, env=env,
        )
        assert 'arm=card dataset=levir_cc seed=1111' in result.stdout
        assert 'arm=rsaca dataset=levir_cc seed=1111' in result.stdout
        assert result.stdout.count('DRY RUN: bash scripts/_run_paper_training.sh') == 2


def test_paired_summary_requires_matching_protocol_and_resolved_configs():
    with tempfile.TemporaryDirectory() as root:
        subprocess.run(
            [
                'python', 'scripts/lock_paired_card_rsaca_protocol.py',
                '--pair-root', root, '--git-commit', 'abc123', '--source-digest', 'source',
                '--source-status-digest', 'status', '--python', 'python',
                '--num-workers', '8', '--datasets', 'levir_cc', '--seeds', '1111',
                '--rsaca-arm', 'whole_adapter_gate_v1',
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        metrics = {
            'Bleu_1': 0.8, 'Bleu_2': 0.7, 'Bleu_3': 0.6, 'Bleu_4': 0.5,
            'METEOR': 0.4, 'ROUGE_L': 0.7, 'CIDEr': 1.4, 'SPICE': 0.3,
        }
        for arm in ('card', 'rsaca'):
            exp_dir = os.path.join(root, arm, '%s_levir_cc_seed1111' % arm)
            snapshots = os.path.join(exp_dir, 'snapshots')
            os.makedirs(snapshots)
            checkpoint = os.path.join(snapshots, '%s.pt' % arm)
            open(checkpoint, 'w').close()
            arm_metrics = dict(metrics)
            if arm == 'rsaca':
                arm_metrics = {key: value + 0.01 for key, value in arm_metrics.items()}
            with open(os.path.join(exp_dir, 'best_snapshot_for_paper.json'), 'w', encoding='utf-8') as handle:
                json.dump({'best_snapshot': checkpoint}, handle)
            with open(os.path.join(exp_dir, 'test_paired_locked_result.json'), 'w', encoding='utf-8') as handle:
                json.dump({'snapshot_path': checkpoint, 'metrics': arm_metrics}, handle)
            config = {
                'exp_dir': os.path.join(root, arm),
                'exp_name': '%s_levir_cc_seed1111' % arm,
                'data': {'dataset': 'levir_cc', 'num_workers': 8, 'use_semantic_maps': arm == 'rsaca'},
                'model': {'type': 'sgc_card', 'semantic_input_mode': 'cross_attention' if arm == 'rsaca' else 'none'},
                'train': {'seed': 1111, 'max_iter': 10000, 'semantic_detach_ratio': 0.5 if arm == 'rsaca' else 0.0},
            }
            with open(os.path.join(exp_dir, 'resolved_config.json'), 'w', encoding='utf-8') as handle:
                json.dump(config, handle)
        output = os.path.join(root, 'summary.json')
        subprocess.run(
            [
                'python', 'scripts/summarize_paired_card_rsaca_matrix.py',
                '--pair-root', root, '--datasets', 'levir_cc', '--seeds', '1111', '--output', output,
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        with open(output, encoding='utf-8') as handle:
            summary = json.load(handle)
        assert summary['acceptance_passed'] is True
        assert abs(summary['datasets']['levir_cc']['delta_rsaca_minus_card']['CIDEr'] - 0.01) < 1e-12

        rsaca_config = os.path.join(root, 'rsaca', 'rsaca_levir_cc_seed1111', 'resolved_config.json')
        with open(rsaca_config, encoding='utf-8') as handle:
            payload = json.load(handle)
        payload['train']['max_iter'] = 9000
        with open(rsaca_config, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        rejected = subprocess.run(
            [
                'python', 'scripts/summarize_paired_card_rsaca_matrix.py',
                '--pair-root', root, '--datasets', 'levir_cc', '--seeds', '1111', '--output', output,
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert rejected.returncode != 0
        assert 'outside the semantic-arm allowlist' in rejected.stderr


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


def test_detached_gate_runner_dry_run_is_isolated_and_enabled():
    with tempfile.TemporaryDirectory() as root:
        env = os.environ.copy()
        env['RUN_ROOT'] = root
        result = subprocess.run(
            [
                'bash', 'scripts/run_reliability_sparse_rsaca_v2_detached_gate.sh',
                '--stage', 'train', '--dataset', 'levir_cc', '--seed', '3333', '--dry-run',
            ],
            cwd=ROOT, check=True, capture_output=True, text=True, env=env,
        )
        assert root in result.stdout
        assert 'detached_gate_inputs=1' in result.stdout
        assert 'visual_gate=0 fallback=0 warmup=0' in result.stdout


def test_detached_reliability_gate_inputs_do_not_require_grad():
    class CaptureGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = None

        def forward(self, inputs):
            self.inputs = inputs
            return inputs.new_zeros(inputs.size(0), 1)

    fusion = SemanticCrossAttentionFusion(
        embed_dim=8,
        num_semantic_classes=7,
        num_heads=2,
        dropout=0.0,
        use_sparse_change_tokens=True,
        use_reliability_gate=True,
        detach_reliability_inputs=True,
    )
    capture_gate = CaptureGate()
    fusion.reliability_gate = capture_gate
    diff_feat = torch.randn(2, 8, 2, 2, requires_grad=True)
    semantic_diff = torch.tensor([
        [[0, 6], [6, 0]],
        [[6, 0], [0, 6]],
    ])

    output = fusion(diff_feat, semantic_diff=semantic_diff)
    output.mean().backward()

    assert capture_gate.inputs is not None
    assert not capture_gate.inputs.requires_grad
    assert diff_feat.grad is not None


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
                'ALL_ABOVE_BASELINE': '1.0',
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


def test_detached_gate_paired_control_dry_run_is_isolated_and_disabled():
    with tempfile.TemporaryDirectory() as root:
        env = os.environ.copy()
        env['RUN_ROOT'] = root
        result = subprocess.run(
            [
                'bash', 'scripts/run_reliability_sparse_rsaca_v2_paired_control.sh',
                '--stage', 'train', '--dataset', 'levir_cc', '--seed', '3333', '--dry-run',
            ],
            cwd=ROOT, check=True, capture_output=True, text=True, env=env,
        )
        assert root in result.stdout
        assert 'detached_gate_inputs=0' in result.stdout
        assert 'visual_gate=0 fallback=0 warmup=0' in result.stdout


def test_paired_validation_comparison_requires_all_non_spice_improvements():
    with tempfile.TemporaryDirectory() as root:
        metrics = {
            'Bleu_1': 0.8, 'Bleu_2': 0.7, 'Bleu_3': 0.6, 'Bleu_4': 0.5,
            'METEOR': 0.4, 'ROUGE_L': 0.7, 'CIDEr': 1.4, 'SPICE': 0.3,
        }
        for label, offset in (('candidate', 0.01), ('control', 0.0)):
            exp_dir = os.path.join(root, label, 'unified_rsaca_levir_cc_seed3333')
            snapshots = os.path.join(exp_dir, 'snapshots')
            os.makedirs(snapshots)
            checkpoint = os.path.join(snapshots, 'checkpoint.pt')
            open(checkpoint, 'w').close()
            selected = dict(metrics)
            for metric in ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'):
                selected[metric] += offset
            with open(os.path.join(exp_dir, 'best_snapshot_for_paper.json'), 'w', encoding='utf-8') as handle:
                json.dump({'best': {'snapshot_path': checkpoint, 'metrics': selected}}, handle)
        output = os.path.join(root, 'comparison.json')
        subprocess.run(
            [
                'python', 'scripts/compare_rsaca_validation_pairs.py',
                '--candidate-root', os.path.join(root, 'candidate'),
                '--control-root', os.path.join(root, 'control'),
                '--seeds', '3333', '--output', output,
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        with open(output, encoding='utf-8') as handle:
            payload = json.load(handle)
        assert payload['screen']['locked_test_recommended'] is True


def test_paper_selector_rejects_empty_snapshot_rows():
    with tempfile.TemporaryDirectory() as root:
        csv_path = os.path.join(root, 'val_metrics.csv')
        fields = ['iter', 'snapshot_path', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: '' for key in fields})
        result = subprocess.run(
            [
                'python', 'scripts/select_best_snapshot_for_paper.py',
                '--exp_dir', root, '--csv', csv_path,
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert 'No valid validation snapshots' in result.stderr
