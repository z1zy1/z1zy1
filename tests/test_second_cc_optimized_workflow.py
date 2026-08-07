import json
import os
import subprocess
import sys
import tempfile
import unittest


METRICS = {
    'Bleu_1': 0.60,
    'Bleu_2': 0.43,
    'Bleu_3': 0.32,
    'Bleu_4': 0.24,
    'METEOR': 0.23,
    'ROUGE_L': 0.53,
    'CIDEr': 0.78,
    'SPICE': 0.22,
}


class SecondCcOptimizedWorkflowTest(unittest.TestCase):
    def test_lock_summary_and_prune_keep_only_locked_snapshots(self):
        with tempfile.TemporaryDirectory(prefix='second_cc_optimized_test_') as root:
            exp_root = os.path.join(root, 'experiments')
            run_root = os.path.join(exp_root, 'runs')
            data_root = os.path.join(root, 'data')
            os.makedirs(exp_root)
            os.makedirs(data_root)
            baseline_path = os.path.join(exp_root, 'baseline.json')
            baseline = {name: value - 0.01 for name, value in METRICS.items()}
            self.write_json(baseline_path, {'selected_val_metrics': baseline})

            for seed in (1111, 2222, 3333):
                name = 'second_cc_opt_lr20e5_seed%d' % seed
                exp_dir = os.path.join(run_root, name)
                snapshot_dir = os.path.join(exp_dir, 'snapshots')
                os.makedirs(snapshot_dir)
                selected = os.path.join(snapshot_dir, '%s_checkpoint_9000.pt' % name)
                unlocked = os.path.join(snapshot_dir, '%s_checkpoint_8000.pt' % name)
                self.write_bytes(selected, b'locked')
                self.write_bytes(unlocked, b'unlocked')
                self.write_json(os.path.join(exp_dir, 'resolved_config.json'), {})
                self.write_json(os.path.join(exp_dir, 'best_checkpoint.json'), {
                    'status': 'done',
                    'selection_strategy': 'val_baseline_stable_window',
                    'selection_uses_test_metrics': False,
                    'selection_metric_split': 'validation',
                    'stability_window_size': 3,
                    'expected_step_gap': 1000,
                    'stable_window_count': 1,
                    'selected_val_metrics': METRICS,
                    'selected_checkpoint': selected,
                    'selected_source_exp_dir': exp_dir,
                    'selected_step': '9000',
                })

            manifest_path = os.path.join(exp_root, 'lock.json')
            self.run_script(
                'scripts/build_second_cc_optimized_lock.py',
                '--exp_root', exp_root,
                '--run_root', run_root,
                '--dataset_root', data_root,
                '--baseline', baseline_path,
                '--specs', 'lr20e5',
                '--output', manifest_path,
            )
            self.run_script('scripts/build_second_cc_optimized_lock.py', '--verify', manifest_path)
            manifest = self.load_json(manifest_path)

            baseline_summary = os.path.join(exp_root, 'baseline_summary.json')
            test_baseline = {name: value - 0.02 for name, value in METRICS.items()}
            self.write_json(baseline_summary, {'results': [{'dataset': 'second_cc', **test_baseline}]})
            for lock in manifest['locks']:
                result = {
                    'snapshot_path': lock['checkpoint']['path'],
                    'metrics': METRICS,
                    'group_metrics': {
                        'change': {'metrics': METRICS},
                        'nochange': {'metrics': METRICS},
                    },
                }
                self.write_json(lock['test_result'], result)

            summary_path = os.path.join(exp_root, 'summary.json')
            self.run_script(
                'scripts/summarize_second_cc_optimized_tests.py',
                '--manifest', manifest_path,
                '--baseline_summary', baseline_summary,
                '--output_json', summary_path,
                '--output_csv', os.path.join(exp_root, 'summary.csv'),
            )
            self.assertTrue(self.load_json(summary_path)['acceptance_passed'])

            plan_path = os.path.join(exp_root, 'prune.json')
            self.run_script(
                'scripts/prune_second_cc_optimized_snapshots.py',
                '--manifest', manifest_path,
                '--output', plan_path,
                '--apply',
            )
            plan = self.load_json(plan_path)
            self.assertEqual(plan['delete_count'], 3)
            self.assertTrue(all(os.path.isfile(lock['checkpoint']['path']) for lock in manifest['locks']))
            self.assertTrue(all(not os.path.exists(path) for path in plan['delete_snapshots']))

    @staticmethod
    def run_script(script, *args):
        subprocess.check_call([sys.executable, script, *args])

    @staticmethod
    def write_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)

    @staticmethod
    def load_json(path):
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)

    @staticmethod
    def write_bytes(path, payload):
        with open(path, 'wb') as handle:
            handle.write(payload)


if __name__ == '__main__':
    unittest.main()
