from types import SimpleNamespace
import unittest

import torch

from utils.utils import build_lr_scheduler, build_transfer_parameter_groups


def cfg(max_iter=100, warmup=10, min_ratio=0.1, scale=0.1):
    optim = SimpleNamespace(
        lr=2e-4,
        scheduler='warmup_cosine',
        warmup_steps=warmup,
        min_lr_ratio=min_ratio,
        pretrained_lr_scale=scale,
    )
    return SimpleNamespace(train=SimpleNamespace(max_iter=max_iter, optim=optim))


class TransferOptimizationTest(unittest.TestCase):
    def test_warmup_cosine_scheduler_is_per_iteration_and_preserves_group_ratio(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([
            {'params': [first], 'lr': 1.0},
            {'params': [second], 'lr': 0.1},
        ])
        scheduler, per_iteration = build_lr_scheduler(optimizer, cfg())

        self.assertTrue(per_iteration)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 0.01)
        for _ in range(9):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1.0)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 0.1)
        for _ in range(90):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 0.01)

    def test_transfer_groups_assign_missing_parameters_to_target_lr(self):
        change = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        speaker = torch.nn.Linear(2, 2)
        groups, summary = build_transfer_parameter_groups(
            change,
            speaker,
            cfg(),
            init_checkpoint='source.pt',
            change_missing=('1.weight', '1.bias'),
            speaker_missing=(),
        )

        self.assertEqual([group['group_name'] for group in groups], ['pretrained', 'target'])
        self.assertEqual([group['lr'] for group in groups], [2e-5, 2e-4])
        target_count = change[1].weight.numel() + change[1].bias.numel()
        self.assertEqual(summary['target_parameters'], target_count)
        self.assertGreater(summary['pretrained_parameters'], summary['target_parameters'])

    def test_warmup_cosine_rejects_invalid_warmup(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        with self.assertRaisesRegex(ValueError, 'warmup_steps'):
            build_lr_scheduler(optimizer, cfg(max_iter=10, warmup=10))


if __name__ == '__main__':
    unittest.main()
