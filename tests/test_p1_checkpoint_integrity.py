import csv
import json
import os
import subprocess
import tempfile
import unittest

import torch

from utils.checkpoint_integrity import atomic_copy_checkpoint, atomic_save_checkpoint, validate_checkpoint_file


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ('Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE')


class P1CheckpointIntegrityTest(unittest.TestCase):
    def test_atomic_checkpoint_is_regular_immutable_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, 'snapshots', 'run_checkpoint_1000.pt')
            digest = atomic_save_checkpoint(
                {'weight': torch.tensor([1.0, 2.0])}, checkpoint,
                immutable=True, write_checksum=True,
            )
            self.assertFalse(os.path.islink(checkpoint))
            self.assertEqual(digest, validate_checkpoint_file(checkpoint, require_checksum=True))
            with self.assertRaises(FileExistsError):
                atomic_save_checkpoint({}, checkpoint, immutable=True, write_checksum=True)
            loaded = torch.load(checkpoint, map_location='cpu')
            self.assertTrue(torch.equal(loaded['weight'], torch.tensor([1.0, 2.0])))

    def test_checkpoint_metadata_step_is_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, 'run_checkpoint_1000.pt')
            atomic_save_checkpoint(
                {'global_step': 1000}, checkpoint, immutable=True,
                write_checksum=True, expected_step=1000,
            )
            validate_checkpoint_file(
                checkpoint, require_checksum=True, require_metadata=True,
                expected_step=1000,
            )
            with self.assertRaisesRegex(ValueError, 'step mismatch'):
                validate_checkpoint_file(
                    checkpoint, require_checksum=True, require_metadata=True,
                    expected_step=2000,
                )

    def test_atomic_checkpoint_refuses_symlink_destination(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, 'target.pt')
            torch.save({'kept': True}, target)
            destination = os.path.join(root, 'checkpoint.pt')
            os.symlink(target, destination)
            with self.assertRaisesRegex(RuntimeError, 'symlink'):
                atomic_save_checkpoint({}, destination)
            self.assertEqual(torch.load(target, map_location='cpu'), {'kept': True})

    def test_selected_alias_is_independent_and_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'run_checkpoint_1000.pt')
            alias = os.path.join(root, 'best_for_paper.pth')
            digest = atomic_save_checkpoint({'value': 7}, source, immutable=True, write_checksum=True)
            self.assertEqual(atomic_copy_checkpoint(source, alias, require_checksum=True), digest)
            self.assertFalse(os.path.islink(alias))
            self.assertNotEqual(os.stat(source).st_ino, os.stat(alias).st_ino)
            with open(alias, 'ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                validate_checkpoint_file(alias, require_checksum=True)
            self.assertEqual(digest, validate_checkpoint_file(source, require_checksum=True))

    def test_copy_refuses_source_destination_same_inode(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'run_checkpoint_1000.pt')
            alias = os.path.join(root, 'best_for_paper.pth')
            atomic_save_checkpoint({'value': 7}, source, immutable=True, write_checksum=True)
            os.link(source, alias)
            with self.assertRaisesRegex(ValueError, 'same inode'):
                atomic_copy_checkpoint(source, alias, require_checksum=True)
            self.assertTrue(os.path.isfile(source))

    def test_copy_refuses_destination_equal_to_source(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, 'run_checkpoint_1000.pt')
            digest = atomic_save_checkpoint({'value': 7}, source, immutable=True, write_checksum=True)
            with self.assertRaisesRegex(ValueError, 'identical'):
                atomic_copy_checkpoint(source, source, require_checksum=True)
            self.assertEqual(digest, validate_checkpoint_file(source, require_checksum=True))

    def _write_grid(self, root, omit_step=None):
        snapshots = os.path.join(root, 'snapshots')
        os.makedirs(snapshots)
        csv_path = os.path.join(root, 'val_metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=('iter', 'snapshot_path') + METRICS)
            writer.writeheader()
            for step in range(1000, 10001, 1000):
                if step == omit_step:
                    continue
                checkpoint = os.path.join(snapshots, 'run_checkpoint_%d.pt' % step)
                atomic_save_checkpoint({'step': step}, checkpoint, immutable=True, write_checksum=True)
                values = {name: 1.0 + step / 100000.0 for name in METRICS}
                writer.writerow(dict({'iter': step, 'snapshot_path': checkpoint}, **values))
        return csv_path

    def test_p1_selector_requires_complete_grid_and_records_integrity(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = self._write_grid(root)
            output = os.path.join(root, 'best.json')
            alias = os.path.join(root, 'best.pth')
            subprocess.run(
                ['python', 'scripts/select_best_snapshot_p1.py', '--exp-dir', root,
                 '--csv', csv_path, '--output-json', output, '--copy-path', alias],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            with open(output, encoding='utf-8') as handle:
                payload = json.load(handle)
            self.assertEqual(payload['best']['iter'], 10000)
            self.assertEqual(
                payload['best_snapshot_sha256'],
                validate_checkpoint_file(payload['best_snapshot'], require_checksum=True),
            )
            self.assertEqual(payload['best_snapshot_size_bytes'], os.path.getsize(payload['best_snapshot']))
            self.assertEqual(validate_checkpoint_file(alias, require_checksum=True), payload['best_snapshot_sha256'])

    def test_p1_selector_rejects_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = self._write_grid(root, omit_step=5000)
            result = subprocess.run(
                ['python', 'scripts/select_best_snapshot_p1.py', '--exp-dir', root, '--csv', csv_path],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('1000..10000 exactly once', result.stderr)

    def test_p1_selector_rejects_step_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = self._write_grid(root)
            with open(csv_path, newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            rows[0]['iter'] = '2000'
            with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=('iter', 'snapshot_path') + METRICS)
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                ['python', 'scripts/select_best_snapshot_p1.py', '--exp-dir', root, '--csv', csv_path],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no valid checksummed snapshot for iter 2000', result.stderr)

    def test_p1_selector_rejects_checkpoint_without_checksum(self):
        with tempfile.TemporaryDirectory() as root:
            snapshots = os.path.join(root, 'snapshots')
            os.makedirs(snapshots)
            checkpoint = os.path.join(snapshots, 'run_checkpoint_1000.pt')
            torch.save({'step': 1000}, checkpoint)
            csv_path = os.path.join(root, 'val_metrics.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=('iter', 'snapshot_path') + METRICS)
                writer.writeheader()
                writer.writerow(dict({'iter': 1000, 'snapshot_path': checkpoint}, **{name: 1.0 for name in METRICS}))
            result = subprocess.run(
                ['python', 'scripts/select_best_snapshot_p1.py', '--exp-dir', root, '--csv', csv_path],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no valid checksummed snapshot', result.stderr)


if __name__ == '__main__':
    unittest.main()
