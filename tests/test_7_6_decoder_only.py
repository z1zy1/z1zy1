import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DecoderOnlyRegressionTest(unittest.TestCase):
    def test_frozen_auxiliary_loss_is_not_backpropagated_separately(self):
        source = (ROOT / 'train_card_spot.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        matching = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if isinstance(node.test, ast.Name) and node.test.id == 'finetune_decoder_only':
                calls = [
                    child for statement in node.body for child in ast.walk(statement)
                    if isinstance(child, ast.Call)
                ]
                attributes = [
                    call.func for call in calls if isinstance(call.func, ast.Attribute)
                ]
                if any(
                    isinstance(attribute.value, ast.Name)
                    and attribute.value.id == 'main_loss'
                    and attribute.attr == 'backward'
                    for attribute in attributes
                ):
                    matching.append(attributes)
        self.assertTrue(matching, 'decoder-only backward branch must call main_loss.backward()')
        for attributes in matching:
            self.assertFalse(any(
                isinstance(attribute.value, ast.Name)
                and attribute.value.id == 'weighted_semantic_loss'
                and attribute.attr == 'backward'
                for attribute in attributes
            ))

    def test_levir_runner_keeps_source_forward_switches(self):
        source = (ROOT / 'scripts' / 'run_7_6_followup_train.sh').read_text(encoding='utf-8')
        required = (
            'ENABLE_AUX_MASK=1', 'USE_AUX_SEMANTIC=1',
            'USE_SEMANTIC_PARTIAL_DETACH=1', 'SEMANTIC_DETACH_RATIO=0.5',
            'USE_FEATURE_REWEIGHT=1', 'REWEIGHT_ALPHA=0.2',
            'ALLOW_MISSING_PSEUDO_MASK=1', 'FINETUNE_DECODER_ONLY=1',
            'resolve_7_6_levir_cc_source.py',
        )
        for token in required:
            self.assertIn(token, source)


if __name__ == '__main__':
    unittest.main()
