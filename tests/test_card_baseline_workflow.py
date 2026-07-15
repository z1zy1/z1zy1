import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import build_card_baseline_manifest as manifest_tool
import summarize_card_baseline_tests as summary_tool


def metric_values(seed=0.1):
    return {
        metric: seed + index * 0.01
        for index, metric in enumerate(manifest_tool.REQUIRED_METRICS)
    }


class ValidationBestCiderTest(unittest.TestCase):
    def run_selector(self, exp_dir):
        output = exp_dir / 'baseline_best_checkpoint.json'
        completed = subprocess.run(
            [
                sys.executable, str(SCRIPTS / 'select_best_checkpoint.py'),
                '--exp_dir', str(exp_dir), '--strategy', 'validation_best_cider',
                '--output_json', str(output),
                '--output_txt', str(exp_dir / 'baseline_best_checkpoint.txt'),
            ],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(output.read_text(encoding='utf-8'))

    def test_excludes_incomplete_row_and_records_validation_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp_dir = Path(temporary) / 'baseline'
            snapshots = exp_dir / 'snapshots'
            snapshots.mkdir(parents=True)
            incomplete = snapshots / 'snapshot_1000.pth'
            complete = snapshots / 'snapshot_2000.pth'
            incomplete.write_bytes(b'incomplete')
            complete.write_bytes(b'complete')
            fields = ['iter', 'snapshot_path'] + list(manifest_tool.REQUIRED_METRICS)
            with (exp_dir / 'val_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({'iter': 1000, 'snapshot_path': str(incomplete), 'CIDEr': 9.0})
                row = {'iter': 2000, 'snapshot_path': str(complete)}
                row.update(metric_values(0.2))
                writer.writerow(row)
            payload = self.run_selector(exp_dir)
            self.assertEqual(payload['status'], 'done')
            self.assertEqual(os.path.normpath(payload['selected_checkpoint']), os.path.normpath(str(complete)))
            self.assertIs(payload['selection_uses_test_metrics'], False)
            self.assertEqual(payload['selection_metric_split'], 'validation')
            self.assertEqual(payload['baseline_source'], 'not_used')
            self.assertEqual(set(payload['selected_val_metrics']), set(manifest_tool.REQUIRED_METRICS))

    def test_fails_closed_when_no_row_has_all_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp_dir = Path(temporary) / 'baseline'
            snapshots = exp_dir / 'snapshots'
            snapshots.mkdir(parents=True)
            checkpoint = snapshots / 'snapshot_1000.pth'
            checkpoint.write_bytes(b'checkpoint')
            with (exp_dir / 'val_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['iter', 'snapshot_path', 'CIDEr'])
                writer.writeheader()
                writer.writerow({'iter': 1000, 'snapshot_path': str(checkpoint), 'CIDEr': 1.0})
            payload = self.run_selector(exp_dir)
            self.assertEqual(payload['status'], 'no_valid_checkpoint')
            self.assertEqual(payload['failure_reason'], 'no_checkpoint_with_complete_validation_metrics')
            self.assertIs(payload['selection_uses_test_metrics'], False)


class LockedBaselineTest(unittest.TestCase):
    def make_experiment(self, exp_root, dataset, exp_name):
        exp_dir = exp_root / exp_name
        snapshots = exp_dir / 'snapshots'
        features = exp_root / ('features_' + dataset)
        data_root = exp_root / ('data_' + dataset)
        snapshots.mkdir(parents=True)
        features.mkdir()
        data_root.mkdir()
        annotation = data_root / 'captions.json'
        annotation.write_text('{}', encoding='utf-8')
        checkpoint = snapshots / 'snapshot_10000.pth'
        checkpoint.write_bytes(('checkpoint-' + dataset).encode())
        metrics = metric_values(0.3)
        val_csv = exp_dir / 'val_metrics.csv'
        fields = ['iter', 'snapshot_path'] + list(manifest_tool.REQUIRED_METRICS)
        with val_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            row = {'iter': 10000, 'snapshot_path': str(checkpoint)}
            row.update(metrics)
            writer.writerow(row)
        cfg = {
            'model': {
                'type': 'card', 'semantic_input_mode': 'none',
                'enable_aux_mask': False, 'use_aux_mask': False,
                'semantic_fusion_gamma_init': 0.0, 'semantic_fusion_gamma_max': 0.0,
            },
            'data': {
                'dataset': dataset, 'data_root': str(data_root),
                'default_feature_dir': str(features), 'eval_anno_path': str(annotation),
                'use_change_mask': False, 'use_semantic_maps': False,
                'allow_missing_pseudo_mask': False, 'train': {'batch_size': 32},
            },
            'train': {
                'init_checkpoint': '', 'start_from': None, 'finetune_steps': 0,
                'max_iter': 10000, 'total_steps': 10000, 'snapshot_interval': 1000,
                'save_interval': 1000, 'eval_interval': 1000, 'seed': 1111,
                'lambda_mask': 0.0, 'lambda_semantic': 0.0,
                'use_aux_mask': False, 'use_semantic_aux': False,
                'use_aux_semantic': False, 'use_semantic_detach': False,
                'use_semantic_partial_detach': False, 'use_partial_detach': False,
                'use_semantic_cross_attention': False, 'use_semantic_hard_gate': False,
                'use_feature_reweight': False, 'detach_reweight_mask': False,
                'use_content_word_weight': False, 'use_content_word_weighted_ce': False,
                'normalize_content_word_weights': False, 'use_relation_aux': False,
                'use_weak_mask_prior': False, 'use_aux_warmup': False,
                'paper_selection_mode': False,
                'optim': {
                    'type': 'adam', 'lr': 0.0002, 'weight_decay': 0.0,
                    'step_size': 17, 'gamma': 0.1,
                },
            },
        }
        (exp_dir / 'resolved_config.json').write_text(json.dumps(cfg), encoding='utf-8')
        (exp_dir / 'resolved_config.yaml').write_text('model:\n  type: card\n', encoding='utf-8')
        selection = {
            'status': 'done', 'selection_strategy': 'validation_best_cider',
            'selection_uses_test_metrics': False, 'selection_metric_split': 'validation',
            'selected_source_exp_dir': str(exp_dir),
            'selected_checkpoint_path': str(checkpoint), 'metrics_file': str(val_csv),
            'selected_val_metrics': metrics,
        }
        (exp_dir / 'baseline_best_checkpoint.json').write_text(json.dumps(selection), encoding='utf-8')
        return exp_dir, checkpoint, cfg

    def make_manifest(self, root):
        exp_root = root / 'experiments'
        names = dict(manifest_tool.DEFAULT_EXPERIMENTS)
        entries = {}
        for dataset in manifest_tool.DATASET_ORDER:
            self.make_experiment(exp_root, dataset, names[dataset])
            entries[dataset] = manifest_tool.build_entry(str(exp_root), dataset, names[dataset])
        manifest = {
            'status': 'validation_locked', 'selection_uses_test_metrics': False,
            'datasets': entries,
        }
        path = exp_root / 'card_baseline_locked_manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path, manifest

    def test_manifest_rejects_enhanced_source_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp_root = root / 'experiments'
            exp_dir, _, cfg = self.make_experiment(
                exp_root, 'levir_mci', manifest_tool.DEFAULT_EXPERIMENTS['levir_mci']
            )
            cfg['train']['use_content_word_weight'] = True
            (exp_dir / 'resolved_config.json').write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'use_content_word_weight'):
                manifest_tool.build_entry(str(exp_root), 'levir_mci', exp_dir.name)

    def test_manifest_rejects_non_adam_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exp_root = root / 'experiments'
            exp_dir, _, cfg = self.make_experiment(
                exp_root, 'second_cc', manifest_tool.DEFAULT_EXPERIMENTS['second_cc']
            )
            cfg['train']['optim']['type'] = 'sgd'
            (exp_dir / 'resolved_config.json').write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'optim.type'):
                manifest_tool.build_entry(str(exp_root), 'second_cc', exp_dir.name)

    def test_summary_uses_only_nested_metrics_and_matches_locked_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest = self.make_manifest(Path(temporary))
            for dataset, entry in manifest['datasets'].items():
                payload = {
                    'snapshot_path': entry['selected_checkpoint'],
                    'metrics': metric_values(0.4),
                    'CIDEr': 999.0,
                    'deltas': {'CIDEr': 999.0},
                }
                result_path = Path(entry['test_result'])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload), encoding='utf-8')
            summary = summary_tool.build_summary(str(manifest_path))
            self.assertEqual(len(summary['results']), 3)
            self.assertAlmostEqual(summary['results'][0]['CIDEr'], metric_values(0.4)['CIDEr'])
            self.assertIs(summary['selection_uses_test_metrics'], False)

    def test_manifest_rejects_tampered_test_data_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest = self.make_manifest(Path(temporary))
            manifest['datasets']['second_cc']['data_root'] = str(Path(temporary) / 'wrong-data')
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'data_root'):
                manifest_tool.verify_manifest(str(manifest_path))

    def test_summary_fails_on_missing_core_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, manifest = self.make_manifest(Path(temporary))
            for dataset, entry in manifest['datasets'].items():
                metrics = metric_values(0.4)
                if dataset == 'second_cc': metrics.pop('SPICE')
                result_path = Path(entry['test_result'])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps({
                    'snapshot_path': entry['selected_checkpoint'], 'metrics': metrics,
                }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SPICE'):
                summary_tool.build_summary(str(manifest_path))

    def test_single_result_validator_rejects_wrong_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / 'result.json'
            result.write_text(json.dumps({
                'snapshot_path': str(Path(temporary) / 'wrong.pth'),
                'metrics': metric_values(0.4),
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                summary_tool.validate_locked_test_result(
                    str(result), str(Path(temporary) / 'locked.pth')
                )


if __name__ == '__main__':
    unittest.main()
