import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from build_7_6_locked_manifest import (
    audit_speaker_vocab_shapes,
    audit_vocab_file,
    directory_inventory,
)


METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']


def values(spice=0.30, **updates):
    result = {metric: (1.0 if metric == 'CIDEr' else 0.5) for metric in METRICS}
    result['SPICE'] = spice
    result.update(updates)
    return result


class StableValidationWindowTest(unittest.TestCase):
    def make_exp(self, root, name, rows):
        exp = os.path.join(root, name)
        snapshots = os.path.join(exp, 'snapshots')
        os.makedirs(snapshots)
        fields = ['iter', 'snapshot_path'] + METRICS
        with open(os.path.join(exp, 'val_metrics.csv'), 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step, metric_values in rows:
                checkpoint = os.path.join(snapshots, '%s_checkpoint_%d.pt' % (name, step))
                open(checkpoint, 'w').close()
                writer.writerow(dict({'iter': step, 'snapshot_path': checkpoint}, **metric_values))
        return exp

    def baseline(self, root, metric_values=None, name='baseline_best_checkpoint.json'):
        path = os.path.join(root, name)
        selected = metric_values or values(spice=0.20)
        payload = {
            'status': 'done',
            'selection_strategy': 'validation_best_cider',
            'selection_uses_test_metrics': False,
            'selection_metric_split': 'validation',
            'exp_name': 'second_cc_card_rgb_baseline',
            'selected_val_metrics': selected,
            'selected_metrics': selected,
        }
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        return path

    def run_selector(self, exp, baseline, *extra):
        return subprocess.run(
            [
                sys.executable, 'scripts/select_best_checkpoint.py',
                '--exp_dir', exp,
                '--strategy', 'val_baseline_stable_window',
                '--baseline_metrics', baseline,
                '--stability_window', '3',
                '--expected_step_gap', '1000',
                '--min_spice_gain', '0',
                '--require_audited_validation_baseline',
                *extra,
            ],
            text=True,
            capture_output=True,
        )

    def test_selects_centre_and_records_window_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root, 'second_cc_candidate', [
                (1000, values(spice=0.24)),
                (2000, values(spice=0.26)),
                (3000, values(spice=0.25)),
                (4000, values(spice=0.23)),
            ])
            result = self.run_selector(exp, self.baseline(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(exp, 'best_checkpoint.json'), encoding='utf-8') as handle:
                payload = json.load(handle)
            self.assertEqual(payload['status'], 'done')
            self.assertEqual(str(payload['selected_epoch_or_step']), '2000')
            window = payload['stability_window']
            self.assertEqual(window['member_steps'], [1000, 2000, 3000])
            self.assertTrue(window['all_members_complete_finite'])
            self.assertTrue(window['all_members_preserve_validation_baseline'])
            self.assertTrue(window['all_members_preserve_spice_gain'])
            self.assertIn('SPICE', window['metric_mean'])
            self.assertIn('SPICE', window['metric_sample_std'])
            self.assertIn('SPICE', window['metric_worst'])

    def test_step_gap_and_source_boundaries_are_not_crossed(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.make_exp(root, 'second_cc_first', [
                (1000, values(spice=0.25)),
                (3000, values(spice=0.25)),
            ])
            second = self.make_exp(root, 'second_cc_second', [(4000, values(spice=0.25))])
            result = self.run_selector(first, self.baseline(root), '--candidate_exp_dir', second)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(first, 'best_checkpoint.json'), encoding='utf-8') as handle:
                payload = json.load(handle)
            self.assertEqual(payload['status'], 'no_valid_checkpoint')
            self.assertEqual(payload['failure_reason'], 'no_stable_validation_window_preserves_baseline')

    def test_malformed_candidate_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root, 'second_cc_bad', [
                (1000, values(spice=0.25)),
                (2000, values(spice='bad')),
                (3000, values(spice=0.25)),
            ])
            result = self.run_selector(exp, self.baseline(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(exp, 'best_checkpoint.json'), encoding='utf-8') as handle:
                payload = json.load(handle)
            self.assertEqual(payload['status'], 'no_valid_checkpoint')

    def test_window_member_below_required_spice_gain_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root, 'second_cc_spice_drop', [
                (1000, values(spice=0.19)),
                (2000, values(spice=0.40)),
                (3000, values(spice=0.40)),
            ])
            result = self.run_selector(exp, self.baseline(root))
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(exp, 'best_checkpoint.json'), encoding='utf-8') as handle:
                payload = json.load(handle)
            self.assertEqual(payload['status'], 'no_valid_checkpoint')

    def test_invalid_or_test_baseline_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root, 'second_cc_candidate', [
                (1000, values()), (2000, values()), (3000, values()),
            ])
            bad_metrics = values()
            bad_metrics['SPICE'] = 'NaN'
            invalid = self.baseline(root, bad_metrics)
            result = self.run_selector(exp, invalid)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Invalid validation baseline metric SPICE', result.stderr)

            test_json = self.baseline(root, values(), name='test_result.json')
            result = self.run_selector(exp, test_json)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Test result JSON', result.stderr)

    def test_nonfinite_threshold_arguments_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root, 'second_cc_candidate', [
                (1000, values()), (2000, values()), (3000, values()),
            ])
            baseline = self.baseline(root)
            for option in ('--baseline_tolerance', '--min_spice_gain'):
                result = self.run_selector(exp, baseline, option, 'NaN')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('must be finite', result.stderr)


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


class VocabularyCheckpointAuditTest(unittest.TestCase):
    def test_vocab_indices_and_speaker_dimensions_must_agree(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'vocab.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump({'<NULL>': 0, '<UNK>': 1, 'change': 2}, handle)
            audit = audit_vocab_file(path, 3)
            self.assertEqual(audit['size'], 3)
            state = {
                'core.embed.weight': FakeTensor(3, 8),
                'logit.weight': FakeTensor(3, 16),
            }
            shape_audit = audit_speaker_vocab_shapes(state, 3, False)
            self.assertEqual(shape_audit['embedding_shape'], [3, 8])

    def test_fake_checkpoint_vocab_shape_mismatch_is_rejected(self):
        state = {
            'core.embed.weight': FakeTensor(3, 8),
            'logit.weight': FakeTensor(4, 16),
        }
        with self.assertRaisesRegex(ValueError, 'Output vocabulary dimension mismatch'):
            audit_speaker_vocab_shapes(state, 3, False)

    def test_noncontiguous_vocab_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'vocab.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump({'<NULL>': 0, '<UNK>': 2}, handle)
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                audit_vocab_file(path, 2)

    def test_directory_inventory_cache_returns_independent_records(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'feature.npy')
            with open(path, 'wb') as handle:
                handle.write(b'feature')
            first = directory_inventory(root)
            first['role'] = 'mutated'
            second = directory_inventory(root)
            self.assertNotIn('role', second)
            self.assertEqual(second['file_count'], 1)


class LockedTestImmutabilityTest(unittest.TestCase):
    def test_force_overwrite_are_rejected_and_existing_results_are_immutable(self):
        script = os.path.join('scripts', 'run_7_6_followup_test_locked.sh')
        with open(script, encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn('--force|--overwrite)', source)
        self.assertIn('locked tests refuse --force/--overwrite', source)
        rejection = source.split('--force|--overwrite)', 1)[1].split(';;', 1)[0]
        self.assertIn('exit 2', rejection)
        self.assertNotIn('FORCE=', source)
        self.assertNotIn('[ "$FORCE"', source)
        self.assertIn('if [ -s "$result" ]; then', source)
        self.assertIn('validate_result "$result" "$checkpoint"', source)

    def test_manifest_and_runtime_bind_exact_provenance_and_data_layout(self):
        with open('scripts/build_7_6_locked_manifest.py', encoding='utf-8') as handle:
            builder = handle.read()
        self.assertIn("init_checkpoint != expected_init", builder)
        self.assertIn("'levir_cc_init_provenance'", builder)
        self.assertIn("'semantic_map_directories'", builder)
        self.assertIn("'train.semantic_tag_file'", builder)
        self.assertIn('INVENTORY_CACHE.clear()', builder)

        with open('scripts/run_7_6_followup_test_locked.sh', encoding='utf-8') as handle:
            runner = handle.read()
        self.assertIn("resolved_paths['default_feature_dir']", runner)
        self.assertIn('validate_runtime_binding "$dataset" "$source_cfg"', runner)
        self.assertNotIn("features[0]['path']", runner)


if __name__ == '__main__':
    unittest.main()
