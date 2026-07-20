import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


PROTECTED = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']


def metrics(spice=0.2, **updates):
    result = {key: (1.0 if key == 'CIDEr' else 0.5) for key in PROTECTED}
    result['SPICE'] = spice
    result.update(updates)
    return result


class ValidationBaselineParetoSelectorTest(unittest.TestCase):
    def make_exp(self, root, rows):
        snap_dir = os.path.join(root, 'snapshots')
        os.makedirs(snap_dir, exist_ok=True)
        fields = ['iter', 'snapshot_path'] + PROTECTED + ['SPICE']
        with open(os.path.join(root, 'val_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index, row_metrics in enumerate(rows, 1):
                checkpoint = os.path.join(
                    snap_dir,
                    'exp_checkpoint_%d.pt' % (index * 10),
                )
                open(checkpoint, 'w').close()
                writer.writerow(
                    dict(
                        {
                            'iter': index * 10,
                            'snapshot_path': checkpoint,
                        },
                        **row_metrics
                    )
                )

    def write_baseline(self, root, payload=None):
        path = os.path.join(root, 'baseline.json')
        default_payload = {
            'status': 'done',
            'selection_strategy': 'validation_best_cider',
            'selection_uses_test_metrics': False,
            'selection_metric_split': 'validation',
            'exp_name': 'card_test_baseline',
            'selected_val_metrics': metrics(),
        }
        if payload is not None:
            if any(key in payload for key in ('metrics', 'selected_metrics', 'selected_val_metrics')):
                for key in ('metrics', 'selected_metrics', 'selected_val_metrics'):
                    default_payload.pop(key, None)
                default_payload.update(payload)
            else:
                default_payload['selected_val_metrics'] = payload
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default_payload, f)
        return path

    def run_selector(self, root, baseline=None):
        command = [
            sys.executable,
            'scripts/select_best_checkpoint.py',
            '--exp_dir',
            root,
            '--strategy',
            'val_baseline_pareto',
        ]
        if baseline:
            command += ['--baseline_metrics', baseline]
        return subprocess.run(command, text=True, capture_output=True)

    def test_explicit_baseline_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_exp(root, [metrics()])
            result = self.run_selector(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--baseline_metrics is required', result.stderr)

    def test_selected_val_metrics_parsed_and_unsafe_high_spice_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_exp(
                root,
                [
                    metrics(spice=0.9, Bleu_4=0.49),
                    metrics(spice=0.25, CIDEr=1.01),
                ],
            )
            baseline = self.write_baseline(root)
            result = self.run_selector(root, baseline)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(root, 'best_checkpoint.json'), encoding='utf-8') as f:
                payload = json.load(f)
            self.assertEqual(payload['selected_epoch_or_step'], '20')
            self.assertEqual(payload['baseline_source'], baseline)
            self.assertFalse(payload['selection_uses_test_metrics'])

    def test_no_feasible_checkpoint_does_not_select_nearest(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_exp(root, [metrics(spice=0.9, Bleu_4=0.49)])
            result = self.run_selector(root, self.write_baseline(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(root, 'best_checkpoint.json'), encoding='utf-8') as f:
                payload = json.load(f)
            self.assertEqual(payload['status'], 'no_valid_checkpoint')
            self.assertEqual(
                payload['failure_reason'],
                'no_checkpoint_preserves_validation_baseline',
            )
            self.assertEqual(payload['selected_checkpoint'], '')

    def test_pareto_front_prefers_safe_high_spice(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_exp(
                root,
                [
                    metrics(spice=0.25),
                    metrics(
                        spice=0.24,
                        Bleu_1=0.52,
                        Bleu_2=0.52,
                        Bleu_3=0.52,
                        Bleu_4=0.52,
                        METEOR=0.52,
                        ROUGE_L=0.52,
                        CIDEr=1.02,
                    ),
                    metrics(spice=0.23),
                ],
            )
            result = self.run_selector(root, self.write_baseline(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(root, 'best_checkpoint.json'), encoding='utf-8') as f:
                payload = json.load(f)
            self.assertEqual(payload['selected_epoch_or_step'], '10')
            self.assertEqual(payload['pareto_front_count'], 2)
            self.assertGreater(payload['spice_gain'], 0)

    def test_missing_protected_baseline_metric_is_error(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_exp(root, [metrics()])
            incomplete = metrics()
            incomplete.pop('Bleu_3')
            result = self.run_selector(
                root,
                self.write_baseline(root, {'selected_metrics': incomplete}),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Bleu_3', result.stderr)


if __name__ == '__main__':
    unittest.main()
