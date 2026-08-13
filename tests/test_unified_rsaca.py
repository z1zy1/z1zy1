import os
import subprocess


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
