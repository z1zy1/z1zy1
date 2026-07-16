import json
import os
import tempfile
import unittest

from scripts.resolve_experiment_config import resolve_experiment_config


def valid_config(marker):
    return {
        'model': {'type': marker},
        'data': {'dataset': 'levir_cc'},
        'train': {'lambda_mask': 0.0},
    }


class ResolveExperimentConfigTest(unittest.TestCase):
    def write_json(self, root, name, payload):
        path = os.path.join(root, name)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        return path

    def test_prefers_resolved_config(self):
        with tempfile.TemporaryDirectory() as root:
            expected = self.write_json(root, 'resolved_config.json', valid_config('resolved'))
            self.write_json(root, 'cfg.json', valid_config('legacy'))
            path, payload, artifact = resolve_experiment_config(root)
            self.assertEqual(os.path.normpath(path), os.path.normpath(expected))
            self.assertEqual(payload['model']['type'], 'resolved')
            self.assertEqual(artifact, 'resolved_config.json')

    def test_falls_back_to_legacy_cfg_json(self):
        with tempfile.TemporaryDirectory() as root:
            expected = self.write_json(root, 'cfg.json', valid_config('legacy'))
            path, payload, artifact = resolve_experiment_config(root)
            self.assertEqual(os.path.normpath(path), os.path.normpath(expected))
            self.assertEqual(payload['model']['type'], 'legacy')
            self.assertEqual(artifact, 'cfg.json')

    def test_bad_resolved_config_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, 'resolved_config.json'), 'w', encoding='utf-8') as handle:
                handle.write('{broken')
            self.write_json(root, 'cfg.json', valid_config('legacy'))
            with self.assertRaisesRegex(ValueError, 'Invalid JSON config'):
                resolve_experiment_config(root)

    def test_incomplete_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_json(root, 'cfg.json', {'model': {}, 'data': {}})
            with self.assertRaisesRegex(ValueError, 'train'):
                resolve_experiment_config(root)

    def test_missing_artifacts_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, 'neither'):
                resolve_experiment_config(root)


if __name__ == '__main__':
    unittest.main()
