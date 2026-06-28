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
from utils.experiment_tracking import save_resolved_config, sync_wcsg_config_aliases


class WCSGSmokeTest(unittest.TestCase):
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