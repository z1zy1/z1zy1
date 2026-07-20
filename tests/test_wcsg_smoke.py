import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.attr_dict import AttrDict
from utils.experiment_tracking import key_switch_summary, save_resolved_config, sync_wcsg_config_aliases


class WCSGSmokeTest(unittest.TestCase):
    def test_key_switch_summary_allows_missing_train_optim(self):
        cfg = AttrDict({
            'exp_name': 'minimal',
            'data': AttrDict({'dataset': 'levir_cc'}),
            'model': AttrDict({'semantic_input_mode': 'none'}),
            'train': AttrDict({}),
        })
        summary = key_switch_summary(cfg)
        self.assertEqual(summary['learning_rate'], 0.0)
        self.assertEqual(summary['dataset_name'], 'levir_cc')
        self.assertIs(summary['finetune_decoder_only'], False)

    @unittest.skipUnless(torch is not None, 'PyTorch is required for model shape smoke tests.')
    def test_semantic_cross_attention_fusion_class_id_shape(self):
        from models.CARD import SemanticCrossAttentionFusion
        fusion = SemanticCrossAttentionFusion(embed_dim=16, num_semantic_classes=8, num_heads=4, gamma_init=0.1)
        diff = torch.randn(2, 49, 16)
        sem_before = torch.randint(0, 7, (2, 14, 14))
        sem_after = torch.randint(0, 7, (2, 14, 14))
        out = fusion(diff, sem_before, sem_after, spatial_size=(7, 7), detach_ratio=0.5)
        self.assertEqual(tuple(out.shape), tuple(diff.shape))

    @unittest.skipUnless(torch is not None, 'PyTorch is required for model shape smoke tests.')
    def test_semantic_cross_attention_fusion_probability_map_shape(self):
        from models.CARD import SemanticCrossAttentionFusion
        fusion = SemanticCrossAttentionFusion(embed_dim=16, num_semantic_classes=8, num_heads=4, gamma_init=0.1)
        diff = torch.randn(2, 16, 7, 7)
        sem_before = torch.rand(2, 8, 14, 14)
        sem_after = torch.rand(2, 8, 14, 14)
        out = fusion(diff, sem_before, sem_after, detach_ratio=0.5)
        self.assertEqual(tuple(out.shape), tuple(diff.shape))

    @unittest.skipUnless(torch is not None, 'PyTorch is required for model tests.')
    def test_semantic_fusion_gamma_max_clamps_without_state_dict_change(self):
        from models.CARD import SemanticCrossAttentionFusion

        torch.manual_seed(7)
        bounded = SemanticCrossAttentionFusion(
            8, 4, num_heads=2, dropout=0.0, gamma_init=0.2, gamma_max=0.05
        )
        reference = SemanticCrossAttentionFusion(
            8, 4, num_heads=2, dropout=0.0, gamma_init=0.2
        )
        legacy = SemanticCrossAttentionFusion(
            8, 4, num_heads=2, dropout=0.0, gamma_init=0.2
        )
        reference.load_state_dict(bounded.state_dict())
        legacy.load_state_dict(bounded.state_dict())
        with torch.no_grad():
            reference.gamma.fill_(0.05)
        for module in (bounded, reference, legacy):
            module.eval()
        diff = torch.randn(1, 4, 8)
        before = torch.randint(0, 4, (1, 2, 2))
        after = torch.randint(0, 4, (1, 2, 2))
        bounded_out = bounded(diff, before, after, spatial_size=(2, 2))
        reference_out = reference(diff, before, after, spatial_size=(2, 2))
        legacy_out = legacy(diff, before, after, spatial_size=(2, 2))
        self.assertEqual(set(bounded.state_dict()), set(legacy.state_dict()))
        self.assertTrue(torch.allclose(bounded_out, reference_out, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(bounded_out, legacy_out, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(torch is not None, 'PyTorch is required for model tests.')
    def test_semantic_fusion_gamma_max_default_is_legacy_compatible(self):
        from models.CARD import SemanticCrossAttentionFusion

        implicit = SemanticCrossAttentionFusion(
            8, 4, num_heads=2, dropout=0.0, gamma_init=0.2
        )
        explicit = SemanticCrossAttentionFusion(
            8, 4, num_heads=2, dropout=0.0, gamma_init=0.2, gamma_max=0.0
        )
        explicit.load_state_dict(implicit.state_dict())
        implicit.eval()
        explicit.eval()
        diff = torch.randn(1, 4, 8)
        before = torch.randint(0, 4, (1, 2, 2))
        after = torch.randint(0, 4, (1, 2, 2))
        self.assertTrue(
            torch.allclose(
                implicit(diff, before, after, spatial_size=(2, 2)),
                explicit(diff, before, after, spatial_size=(2, 2)),
                atol=1e-7,
                rtol=1e-7,
            )
        )

    @unittest.skipUnless(torch is not None, 'PyTorch is required for loss tests.')
    def test_normalized_content_word_weighted_ce_exact_denominator(self):
        import torch.nn as nn
        import torch.nn.functional as F

        from models.transformer_decoder import DynamicSpeaker

        speaker = DynamicSpeaker.__new__(DynamicSpeaker)
        nn.Module.__init__(speaker)
        speaker.use_content_word_weight = True
        speaker.content_word_token_ids = [1]
        speaker.content_word_weight = 2.0
        speaker.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        scores = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
        targets = torch.tensor([0, 1, -1])
        per_token = F.cross_entropy(scores, targets, ignore_index=-1, reduction='none')
        speaker.normalize_content_word_weights = False
        legacy = speaker._caption_loss(scores, targets)
        speaker.normalize_content_word_weights = True
        normalized = speaker._caption_loss(scores, targets)
        self.assertTrue(torch.allclose(legacy, (per_token[0] + 2.0 * per_token[1]) / 2.0))
        self.assertTrue(torch.allclose(normalized, (per_token[0] + 2.0 * per_token[1]) / 3.0))

    @unittest.skipUnless(torch is not None, 'PyTorch is required for model shape smoke tests.')
    def test_partial_detach_preserves_shape(self):
        from models.CARD import partial_detach_feature
        feat = torch.randn(2, 49, 16, requires_grad=True)
        out = partial_detach_feature(feat, 0.5)
        self.assertEqual(tuple(out.shape), tuple(feat.shape))

    def test_config_aliases_disable_hard_gate_and_reweight(self):
        cfg = AttrDict({
            'exp_name': 'second_cc_wcsg_card_final',
            'exp_dir': './experiments',
            'logger': AttrDict({'display_id': 1}),
            'data': AttrDict({'dataset': 'second_cc'}),
            'model': AttrDict({
                'enable_aux_mask': False,
                'use_aux_mask': False,
                'semantic_input_mode': 'cross_attention',
            }),
            'train': AttrDict({
                'use_aux_mask': False,
                'use_semantic_aux': True,
                'use_aux_semantic': True,
                'use_semantic_cross_attention': True,
                'use_semantic_hard_gate': False,
                'use_feature_reweight': False,
                'use_semantic_partial_detach': True,
                'use_partial_detach': True,
                'semantic_detach_ratio': 0.5,
                'lambda_mask': 0.0,
                'lambda_semantic': 0.005,
                'use_mask_warmup': False,
                'use_aux_warmup': True,
                'use_semantic_warmup': False,
            }),
        })
        sync_wcsg_config_aliases(cfg)
        self.assertTrue(cfg.train.use_semantic_cross_attention)
        self.assertFalse(cfg.train.use_semantic_hard_gate)
        self.assertFalse(cfg.train.use_feature_reweight)

    def test_resolved_config_hashes_differ(self):
        base = AttrDict({
            'exp_name': 'a',
            'exp_dir': '',
            'logger': AttrDict({'display_id': 1}),
            'data': AttrDict({'dataset': 'levir_mci'}),
            'model': AttrDict({'enable_aux_mask': True, 'use_aux_mask': True, 'semantic_input_mode': 'aux'}),
            'train': AttrDict({
                'use_semantic_aux': True,
                'use_aux_semantic': True,
                'use_semantic_cross_attention': False,
                'use_semantic_hard_gate': False,
                'use_feature_reweight': False,
                'use_semantic_partial_detach': True,
                'semantic_detach_ratio': 0.5,
                'lambda_mask': 0.003,
                'lambda_semantic': 0.005,
                'use_mask_warmup': False,
                'use_aux_warmup': True,
                'use_semantic_warmup': False,
            }),
        })
        other = AttrDict(json.loads(json.dumps(base)))
        other.exp_name = 'b'
        other.train['use_feature_reweight'] = True
        with tempfile.TemporaryDirectory() as tmp:
            base.exp_dir = tmp
            other.exp_dir = tmp
            a = save_resolved_config(os.path.join(tmp, 'a'), base)
            b = save_resolved_config(os.path.join(tmp, 'b'), other)
        self.assertNotEqual(a['comparable_config_hash'], b['comparable_config_hash'])

    def test_checkpoint_selector_outputs_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap_dir = os.path.join(tmp, 'snapshots')
            os.makedirs(snap_dir)
            ckpt1 = os.path.join(snap_dir, 'exp_checkpoint_100.pt')
            ckpt2 = os.path.join(snap_dir, 'exp_checkpoint_200.pt')
            open(ckpt1, 'w').close()
            open(ckpt2, 'w').close()
            csv_path = os.path.join(tmp, 'val_metrics.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['iter', 'snapshot_path', 'Bleu_4', 'METEOR', 'CIDEr', 'SPICE'])
                writer.writeheader()
                writer.writerow({'iter': 100, 'snapshot_path': ckpt1, 'Bleu_4': 0.40, 'METEOR': 0.30, 'CIDEr': 1.00, 'SPICE': 0.20})
                writer.writerow({'iter': 200, 'snapshot_path': ckpt2, 'Bleu_4': 0.42, 'METEOR': 0.31, 'CIDEr': 1.02, 'SPICE': 0.22})
            baseline = os.path.join(tmp, 'baseline.json')
            with open(baseline, 'w', encoding='utf-8') as f:
                json.dump({'Bleu_4': 0.39, 'METEOR': 0.30, 'CIDEr': 1.00, 'SPICE': 0.20}, f)
            subprocess.check_call([
                sys.executable,
                'scripts/select_best_checkpoint.py',
                '--exp_dir', tmp,
                '--baseline_metrics', baseline,
            ])
            with open(os.path.join(tmp, 'best_checkpoint.json'), encoding='utf-8') as f:
                payload = json.load(f)
            self.assertEqual(os.path.normpath(payload['selected_checkpoint_path']), os.path.normpath(ckpt2))

    def test_loss_keys_present_in_training_stats(self):
        with open('train_card_spot.py', encoding='utf-8') as f:
            source = f.read()
        for key in ('loss_caption', 'loss_mask', 'loss_semantic', 'loss_total'):
            self.assertIn("stats['%s']" % key, source)


if __name__ == '__main__':
    unittest.main()
