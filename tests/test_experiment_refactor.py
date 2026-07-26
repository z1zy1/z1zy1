import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import yaml

from datasets.datasets import resolve_dataset_module
from utils.attr_dict import AttrDict
from utils.checkpointing import (
    normalize_checkpoint_payload,
    split_model_states,
    strip_module_prefix,
)
from utils.experiment_runtime import (
    RunStateManager,
    StructuredMetricLogger,
    create_stage_logger,
    update_run_summary,
)
from utils.config_validation import validate_resolved_config
from utils.experiment_tracking import comparable_config_dict, save_resolved_config
from utils.metrics import MetricTracker, canonical_metric_name, normalize_metrics
from tools.summarize_experiments import summarize


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class _ScalarLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def minimal_config(exp_dir, exp_name='demo', seed=1111):
    return AttrDict({
        'exp_dir': exp_dir,
        'exp_name': exp_name,
        'logger': AttrDict({'display_id': 1}),
        'data': AttrDict({'dataset': 'levir_mci'}),
        'model': AttrDict({
            'enable_aux_mask': False,
            'semantic_input_mode': 'none',
            'semantic_fusion_gamma_max': 0.0,
        }),
        'train': AttrDict({
            'seed': seed,
            'lambda_mask': 0.0,
            'lambda_semantic': 0.0,
            'selection_strategy': 'validation_best_cider',
            'semantic_detach_ratio': 0.0,
        }),
    })


class MetricTrackerTest(unittest.TestCase):
    def test_aliases_and_stage_averages(self):
        self.assertEqual(canonical_metric_name('BLEU4'), 'Bleu_4')
        self.assertEqual(normalize_metrics({'rouge-l': 0.7})['ROUGE_L'], 0.7)
        tracker = MetricTracker()
        tracker.update('val', {'BLEU4': 0.4, 'loss': 2.0}, weight=1)
        tracker.update('val', {'Bleu_4': 0.8, 'loss': 1.0}, weight=3)
        self.assertAlmostEqual(tracker.averages('val')['Bleu_4'], 0.7)
        self.assertAlmostEqual(tracker.averages('val')['loss'], 1.25)
        best = tracker.update_best('val', {'Bleu_4': 0.8, 'loss': 1.0})
        self.assertEqual(best['Bleu_4'], 0.8)


class ConfigAndDatasetFactoryTest(unittest.TestCase):
    def test_config_validation_and_dataset_aliases(self):
        cfg = minimal_config('.')
        cfg.train.max_iter = 1
        cfg.train.snapshot_interval = 1
        self.assertIs(validate_resolved_config(cfg, phase='train'), cfg)
        cfg.train.use_semantic_partial_detach = True
        cfg.train.semantic_detach_ratio = 1.5
        with self.assertRaisesRegex(ValueError, 'semantic_detach_ratio'):
            validate_resolved_config(cfg, phase='train')
        canonical, module = resolve_dataset_module('second_cc')
        self.assertEqual(canonical, 'rcc_dataset_transformer_levir')
        self.assertEqual(module, 'datasets.rcc_dataset_transformer_levir')
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            resolve_dataset_module('not_registered')
        first = minimal_config('.', exp_name='first')
        first.experiment_group = 'paper'
        first.description = 'first note'
        first.tags = ['a']
        second = minimal_config('.', exp_name='second')
        second.experiment_group = 'debug'
        second.description = 'second note'
        second.tags = ['b']
        self.assertEqual(comparable_config_dict(first), comparable_config_dict(second))


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_legacy_fields_are_exposed_without_removal(self):
        payload = {
            'change_detector_state': {'module.x': 1},
            'speaker_state': {'y': 2},
            'model_cfg': {'name': 'legacy'},
        }
        normalized = normalize_checkpoint_payload(payload)
        self.assertIs(normalized['model_state_dict'], payload['change_detector_state'])
        self.assertEqual(normalized['config'], payload['model_cfg'])
        self.assertIn('change_detector_state', normalized)
        model, speaker = split_model_states(payload)
        self.assertEqual(model, {'module.x': 1})
        self.assertEqual(speaker, {'y': 2})
        self.assertEqual(strip_module_prefix(model), {'x': 1})


class RuntimeArtifactsTest(unittest.TestCase):
    def test_config_metrics_and_state_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'source.yaml')
            with open(source, 'w', encoding='utf-8') as handle:
                yaml.safe_dump({'exp_name': 'demo'}, handle)
            output = os.path.join(root, 'run')
            cfg = minimal_config(root)
            args = SimpleNamespace(cfg=source, sample='value')
            save_resolved_config(output, cfg, args=args, phase='train')
            required = (
                'config_original.yaml',
                'config_resolved.yaml',
                'resolved_config.json',
                'command.txt',
                'environment.txt',
                'git_info.txt',
            )
            for name in required:
                self.assertTrue(os.path.exists(os.path.join(output, name)), name)

            logger = StructuredMetricLogger(output)
            record = logger.log(
                'val',
                {'BLEU4': 0.5, 'loss_total': 1.2, 'mask_f1': 0.7},
                epoch=1,
                global_step=10,
            )
            self.assertEqual(record['Bleu_4'], 0.5)
            with open(logger.jsonl_path, encoding='utf-8') as handle:
                parsed = json.loads(handle.readline())
            self.assertEqual(parsed['stage'], 'val')
            with open(logger.csv_path, newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['Bleu_4'], '0.5')

            state = RunStateManager(
                output,
                experiment_name='demo',
                dataset='levir_mci',
                seed=1111,
            )
            state.start()
            state.complete(final_global_step=10)
            payload = update_run_summary(output, val_inference_completed=True)
            self.assertEqual(payload['status'], 'completed')
            self.assertTrue(payload['val_inference_completed'])

    def test_runtime_artifacts_are_strict_json(self):
        with tempfile.TemporaryDirectory() as output:
            logger = StructuredMetricLogger(output)
            record = logger.log(
                'train',
                {'loss_total': float('nan'), 'tensor_like': _ScalarLike(2.5)},
            )
            self.assertIsNone(record['loss'])
            self.assertEqual(record['extras']['tensor_like'], 2.5)
            self.assertNotIn('loss_total', record['extras'])
            state = RunStateManager(
                output,
                experiment_name='strict_json',
                dataset='levir_cc',
                seed=1111,
            )
            state.complete(best={'score': -float('inf')})
            with open(os.path.join(output, 'run_summary.json'), encoding='utf-8') as handle:
                raw = handle.read()
            self.assertNotIn('NaN', raw)
            self.assertNotIn('Infinity', raw)
            self.assertIsNone(json.loads(raw)['best']['score'])
            readable_logger = create_stage_logger(
                output,
                'train',
                additional_log_paths=(os.path.join(output, 'train_log.txt'),),
            )
            readable_logger.info('strict JSON test')
            self.assertTrue(os.path.exists(os.path.join(output, 'train.log')))
            self.assertTrue(os.path.exists(os.path.join(output, 'train_log.txt')))
            for handler in readable_logger.handlers:
                handler.close()
            readable_logger.handlers.clear()


class ToolEntryTest(unittest.TestCase):
    def write_config(self, root):
        path = os.path.join(root, 'config.yaml')
        with open(path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump({
                'exp_dir': os.path.join(root, 'experiments'),
                'exp_name': 'demo',
                'data': {'dataset': 'levir_mci'},
                'train': {'seed': 1111},
            }, handle)
        return path

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable] + list(args),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_train_eval_test_and_smoke_dry_runs(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.write_config(root)
            checkpoint = os.path.join(root, 'checkpoint.pt')
            train = self.run_tool(
                'tools/train.py', '--config', config, '--run-name', 'dry', '--dry-run'
            )
            self.assertIn('train_card_spot.py', train.stdout)
            for tool in ('tools/evaluate.py', 'tools/test.py'):
                result = self.run_tool(
                    tool, '--config', config, '--checkpoint', checkpoint, '--dry-run'
                )
                self.assertIn('test_card_spot.py', result.stdout)
            smoke = self.run_tool(
                'tools/smoke_test.py', '--config', config, '--dry-run'
            )
            self.assertIn('train_one_batch_and_validate', smoke.stdout)

    def test_wrappers_resolve_outputs_safely_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            config = os.path.join(root, 'relative.yaml')
            with open(config, 'w', encoding='utf-8') as handle:
                yaml.safe_dump({
                    'exp_dir': './experiments',
                    'exp_name': 'relative',
                    'data': {'dataset': 'levir_mci'},
                    'train': {'seed': 1111},
                }, handle)
            train = subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, 'tools', 'train.py'),
                    '--config', config,
                    '--run-name', 'wrapper_relative_dry',
                    '--dry-run',
                ],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            command = json.loads(train.stdout)['command']
            output_dir = command[command.index('--output_dir') + 1]
            self.assertEqual(
                output_dir,
                os.path.join(ROOT, 'experiments', 'wrapper_relative_dry'),
            )

            prediction = os.path.join(root, 'existing_predictions.json')
            with open(prediction, 'w', encoding='utf-8') as handle:
                handle.write('[]')
            refused = subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, 'tools', 'evaluate.py'),
                    '--config', config,
                    '--checkpoint', os.path.join(root, 'fake.pt'),
                    '--result-json', prediction,
                    '--dry-run',
                ],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Refusing to overwrite', refused.stderr)


class SelectionAndSummaryTest(unittest.TestCase):
    def _write_run(self, root, name, seed, metrics, status='completed'):
        run = os.path.join(root, name)
        snapshots = os.path.join(run, 'snapshots')
        os.makedirs(snapshots)
        checkpoint = os.path.join(snapshots, '%s_checkpoint_100.pt' % name)
        open(checkpoint, 'w').close()
        config = {
            'exp_name': name,
            'data': {'dataset': 'levir_mci'},
            'model': {'type': 'card', 'semantic_input_mode': 'aux'},
            'train': {
                'seed': seed,
                'lambda_mask': 0.003,
                'lambda_semantic': 0.005,
                'use_semantic_partial_detach': True,
                'use_feature_reweight': False,
            },
        }
        with open(os.path.join(run, 'resolved_config.json'), 'w', encoding='utf-8') as handle:
            json.dump(config, handle)
        with open(os.path.join(run, 'run_summary.json'), 'w', encoding='utf-8') as handle:
            json.dump({
                'experiment_name': name,
                'dataset': 'levir_mci',
                'run_id': name,
                'seed': seed,
                'status': status,
                'failure_reason': 'synthetic failure' if status == 'failed' else '',
            }, handle)
        if metrics:
            with open(os.path.join(run, 'test_metrics.csv'), 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['checkpoint_path'] + list(metrics))
                writer.writeheader()
                writer.writerow(dict({'checkpoint_path': checkpoint}, **metrics))
        return run, checkpoint

    def test_generic_selection_is_validation_only(self):
        with tempfile.TemporaryDirectory() as root:
            run, checkpoint = self._write_run(
                root,
                'selection_demo',
                1111,
                {name: 0.1 for name in ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')},
            )
            metrics_path = os.path.join(run, 'val_metrics.csv')
            later_checkpoint = os.path.join(
                run, 'snapshots', 'selection_demo_checkpoint_200.pt'
            )
            open(later_checkpoint, 'w').close()
            with open(metrics_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['iter', 'snapshot_path', 'CIDEr', 'SPICE'])
                writer.writeheader()
                writer.writerow({'iter': 100, 'snapshot_path': checkpoint, 'CIDEr': 1.2, 'SPICE': 0.3})
                writer.writerow({'iter': 200, 'snapshot_path': later_checkpoint, 'CIDEr': 1.2, 'SPICE': 0.3})
            subprocess.check_call([
                sys.executable,
                'tools/select_checkpoint.py',
                '--run-dir', run,
                '--primary-metric', 'CIDEr',
                '--secondary-metric', 'SPICE',
            ], cwd=ROOT)
            with open(os.path.join(run, 'checkpoint_selection.json'), encoding='utf-8') as handle:
                payload = json.load(handle)
            self.assertFalse(payload['selection_uses_test_metrics'])
            self.assertEqual(payload['selection_metric_split'], 'validation')
            self.assertEqual(payload['selected_checkpoint'], later_checkpoint)
            rows = summarize(root)
            run_row = next(row for row in rows if row['run_id'] == 'selection_demo')
            self.assertEqual(run_row['best_epoch'], 200)
            self.assertEqual(run_row['val_CIDEr'], 1.2)
            self.assertEqual(run_row['model_name'], 'card')

    def test_summary_keeps_failures_and_reports_sample_std(self):
        metrics_a = {name: 1.0 for name in ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')}
        metrics_b = {name: 3.0 for name in metrics_a}
        with tempfile.TemporaryDirectory() as root:
            self._write_run(root, 'method_seed1111', 1111, metrics_a)
            self._write_run(root, 'method_seed2222', 2222, metrics_b)
            self._write_run(root, 'failed_seed3333', 3333, {}, status='failed')
            rows = summarize(root)
            self.assertTrue(any(row['status'] == 'failed' for row in rows))
            std_rows = [row for row in rows if row['run_id'] == 'std']
            self.assertEqual(len(std_rows), 1)
            self.assertAlmostEqual(float(std_rows[0]['CIDEr']), 2 ** 0.5)

    def test_summary_falls_back_from_blank_legacy_csv_to_json_metrics(self):
        metrics = {
            name: 0.5
            for name in (
                'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE',
            )
        }
        with tempfile.TemporaryDirectory() as root:
            run, _ = self._write_run(root, 'json_fallback_seed1111', 1111, {})
            with open(
                os.path.join(run, 'test_metrics.csv'),
                'w',
                newline='',
                encoding='utf-8',
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=['checkpoint_path'] + list(metrics),
                )
                writer.writeheader()
                writer.writerow({name: '' for name in writer.fieldnames})
            with open(
                os.path.join(run, 'test_metrics.json'),
                'w',
                encoding='utf-8',
            ) as handle:
                json.dump({'stage': 'test', 'metrics': metrics}, handle)
            row = next(item for item in summarize(root) if item['run_id'] == 'json_fallback_seed1111')
            self.assertEqual(row['CIDEr'], 0.5)

    def test_summary_records_malformed_artifact_as_failure(self):
        with tempfile.TemporaryDirectory() as root:
            run = os.path.join(root, 'corrupt_seed1111')
            os.makedirs(run)
            with open(os.path.join(run, 'run_summary.json'), 'w', encoding='utf-8') as handle:
                handle.write('{not-json')
            rows = summarize(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'failed')
            self.assertIn('summary artifact error', rows[0]['failure_reason'])


if __name__ == '__main__':
    unittest.main()
