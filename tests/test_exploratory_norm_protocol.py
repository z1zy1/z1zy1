import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_exploratory_initialization import compare
from utils.config_validation import validate_resolved_config
from utils.semantic_controls import EXPLORATORY_NORM_PROTOCOL, P1_PROTOCOLS


def minimal_config(protocol_id):
    model = SimpleNamespace(
        semantic_input_mode='none', semantic_fusion_gamma_init=0.01,
        semantic_fusion_gamma_max=0.5, semantic_fusion_norm_mode='legacy_post_norm',
        semantic_fusion_reliability_gate_bias=-1.5,
    )
    data = SimpleNamespace(dataset='levir_cc', num_semantic_classes=0,
                           semantic_diff_only=False, semantic_diff_root='',
                           semantic_unknown_change_class=6, semantic_diff_binary=False)
    train = SimpleNamespace(protocol_id=protocol_id, semantic_detach_ratio=0.5,
                            lambda_mask=0.0, lambda_semantic=0.0,
                            aux_warmup_start_ratio=0.0, aux_warmup_end_ratio=1.0,
                            max_iter=10000, snapshot_interval=1000,
                            use_semantic_detach=False, use_semantic_partial_detach=False)
    return SimpleNamespace(exp_name='test', data=data, model=model, train=train)


class ExploratoryNormProtocolTest(unittest.TestCase):
    def test_registered_arm_configs_differ_only_in_norm_mode(self):
        root = Path(__file__).resolve().parents[1]
        config_root = root / 'experiments/exploratory_norm_context_pre_20261015/configs'
        with (config_root / 'C0.json').open(encoding='utf-8') as handle:
            c0 = json.load(handle)
        with (config_root / 'N1.json').open(encoding='utf-8') as handle:
            n1 = json.load(handle)

        def flatten(value, prefix=''):
            result = {}
            for key, item in value.items():
                path = prefix + key
                if isinstance(item, dict):
                    result.update(flatten(item, path + '.'))
                else:
                    result[path] = item
            return result

        left, right = flatten(c0), flatten(n1)
        self.assertEqual(set(left), set(right))
        self.assertEqual(left['arm'], 'C0')
        self.assertEqual(right['arm'], 'N1')
        left.pop('arm')
        right.pop('arm')
        self.assertEqual(
            [key for key in left if left[key] != right[key]],
            ['model.semantic_fusion_norm_mode'],
        )

    def test_protocol_is_registered_for_immutable_validation_artifacts(self):
        self.assertIn(EXPLORATORY_NORM_PROTOCOL, P1_PROTOCOLS)

    def test_protocol_is_rejected_by_formal_test_validation(self):
        with self.assertRaisesRegex(ValueError, 'formal test is prohibited'):
            validate_resolved_config(minimal_config(EXPLORATORY_NORM_PROTOCOL), phase='test')

    def test_old_followup_formal_test_guard_remains_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'formal test is prohibited'):
            validate_resolved_config(
                minimal_config('semantic_controls_followup_20261005'), phase='test'
            )

    def test_existing_p1_protocol_is_not_reclassified_as_exploratory(self):
        cfg = minimal_config('p1_rsaca_20260913')
        self.assertIs(validate_resolved_config(cfg, phase='test'), cfg)

    def test_smoke_runner_has_a_hard_100_step_cap_and_separate_output_root(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / 'scripts/run_exploratory_norm_pair.sh').read_text(encoding='utf-8')
        self.assertIn('RUN_ROOT="${EXPLORATORY_OUTPUT_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}/smoke/$ARM/$DATASET"', script)
        self.assertIn('TRAIN_STEPS=100', script)
        self.assertIn('SNAPSHOT_STEPS=100', script)
        self.assertIn('WARMUP_STEPS=50', script)
        self.assertIn('WARMUP_STEPS=500', script)

    def test_initialization_comparison_detects_any_tensor_difference(self):
        left = {
            'protocol_id': EXPLORATORY_NORM_PROTOCOL, 'seed': 2222,
            'change_detector': {'p': {'shape': [2], 'dtype': 'float32', 'sha256': 'abc'}},
            'speaker': {'q': {'shape': [1], 'dtype': 'float32', 'sha256': 'def'}},
        }
        right = json.loads(json.dumps(left))
        self.assertTrue(compare(left, right)['matched'])
        right['speaker']['q']['sha256'] = 'changed'
        result = compare(left, right)
        self.assertFalse(result['matched'])
        self.assertEqual(result['differences'], ['speaker.q differs'])


if __name__ == '__main__':
    unittest.main()
