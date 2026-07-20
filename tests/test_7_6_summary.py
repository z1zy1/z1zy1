import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import summarize_7_6_locked_tests as summary_tool


def metrics(offset=0.0):
    return {
        metric: 0.10 + index * 0.02 + offset
        for index, metric in enumerate(summary_tool.REQUIRED_METRICS)
    }


class LockedSummaryTest(unittest.TestCase):
    def build_summary(self, manifest, baseline):
        def load_fixture(path):
            return json.loads(Path(path).read_text(encoding='utf-8'))
        with mock.patch.object(
            summary_tool, 'verify_7_6_manifest', side_effect=load_fixture
        ):
            return summary_tool.build_summary(str(manifest), str(baseline))

    def write_result(self, path, checkpoint, metric_values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'snapshot_path': str(checkpoint),
            'metrics': metric_values,
        }), encoding='utf-8')

    def make_fixture(self, root):
        baseline_rows = []
        for dataset in summary_tool.DATASET_ORDER:
            source = root / ('card_baseline_' + dataset)
            source.mkdir()
            checkpoint = source / 'baseline_checkpoint.pt'
            checkpoint.write_bytes(b'baseline')
            result = source / 'test_result.json'
            metric_values = metrics()
            self.write_result(result, checkpoint, metric_values)
            row = {
                'dataset': dataset,
                'exp_name': source.name,
                'selected_checkpoint': str(checkpoint),
                'test_result': str(result),
            }
            row.update(metric_values)
            baseline_rows.append(row)
        baseline_summary = root / 'card_baseline_test_summary.json'
        baseline_summary.write_text(json.dumps({
            'status': 'done',
            'selection_uses_test_metrics': False,
            'source_manifest': str(root / 'card_baseline_locked_manifest.json'),
            'results': baseline_rows,
        }), encoding='utf-8')

        specs = [
            ('levir_cc', None, 0.01),
            ('levir_mci', 1111, 0.01),
            ('levir_mci', 2222, 0.02),
            ('levir_mci', 3333, 0.03),
            ('second_cc', None, 0.01),
        ]
        locks = []
        for dataset, seed, offset in specs:
            suffix = dataset if seed is None else '%s_seed%d' % (dataset, seed)
            source = root / ('source_' + suffix)
            source.mkdir()
            checkpoint = source / 'selected_checkpoint.pt'
            checkpoint.write_bytes(b'candidate')
            target = root / ('target_' + suffix)
            result = target / 'test_7_6_locked_result.json'
            self.write_result(result, checkpoint, metrics(offset))
            selection = target / 'best_checkpoint.json'
            selection.write_text(json.dumps({
                'selection_uses_test_metrics': False,
                'selected_checkpoint': str(checkpoint),
            }), encoding='utf-8')
            locks.append({
                'lock_id': suffix,
                'dataset': dataset,
                'seed': seed,
                'target_exp': target.name,
                'selected_checkpoint': str(checkpoint),
                'selected_source_exp_dir': str(source),
                'selection_json': str(selection),
                'test_result': str(result),
                'method_signature': {'method': 'mask_semantic'} if dataset == 'levir_mci' else {},
            })
        manifest = root / '7_6_locked_manifest.json'
        manifest.write_text(json.dumps({
            'status': 'validation_locked',
            'selection_uses_test_metrics': False,
            'card_baseline_manifest': {
                'path': str(root / 'card_baseline_locked_manifest.json'),
            },
            'locks': locks,
        }), encoding='utf-8')
        return manifest, baseline_summary, locks

    def test_builds_raw_deltas_and_mci_sample_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, baseline, _ = self.make_fixture(root)
            summary = self.build_summary(manifest, baseline)
            self.assertEqual(len(summary['raw_results']), 5)
            mci = summary['dataset_aggregates']['levir_mci']
            self.assertEqual(mci['n'], 3)
            self.assertEqual(mci['seeds'], [1111, 2222, 3333])
            self.assertAlmostEqual(mci['metric_mean']['SPICE'], metrics(0.02)['SPICE'])
            self.assertAlmostEqual(mci['metric_sample_std']['SPICE'], 0.01)
            self.assertTrue(mci['all_mean_metrics_strictly_above_baseline'])
            self.assertTrue(all(
                row['all_metrics_strictly_above_baseline']
                for row in summary['raw_results']
            ))

            output_json = root / 'outputs' / 'summary.json'
            output_csv = root / 'csv' / 'summary.csv'
            summary_tool.write_summary(summary, str(output_json), str(output_csv))
            self.assertTrue(output_json.is_file())
            with output_csv.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertEqual(
                len([row for row in rows if row['row_type'] == 'dataset_sample_std']),
                1,
            )

    def test_rejects_test_result_for_a_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, baseline, locks = self.make_fixture(root)
            lock = locks[0]
            result_path = Path(lock['test_result'])
            payload = json.loads(result_path.read_text(encoding='utf-8'))
            payload['snapshot_path'] = locks[-1]['selected_checkpoint']
            result_path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'does not match locked checkpoint'):
                self.build_summary(manifest, baseline)

    def test_rejects_fractional_or_reused_mci_seed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, baseline, locks = self.make_fixture(root)
            locks[1]['seed'] = 1111.5
            manifest.write_text(json.dumps({
                'status': 'validation_locked',
                'selection_uses_test_metrics': False,
                'card_baseline_manifest': {
                    'path': str(root / 'card_baseline_locked_manifest.json'),
                },
                'locks': locks,
            }), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'MCI seed set|finite integer'):
                self.build_summary(manifest, baseline)

    def test_rejects_baseline_summary_from_another_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, baseline, _ = self.make_fixture(root)
            payload = json.loads(baseline.read_text(encoding='utf-8'))
            payload['source_manifest'] = str(root / 'another_card_manifest.json')
            baseline.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source_manifest'):
                self.build_summary(manifest, baseline)


if __name__ == '__main__':
    unittest.main()
